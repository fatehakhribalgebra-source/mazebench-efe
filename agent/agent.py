"""The agent: perception -> memory -> EFE planning -> action, plus a supervisor.

Two tiers, for the reason set out in ``env.actions``.

The **planner** chooses among movement and camera rotation by minimising
expected free energy. That is the learned, model-based part.

The **supervisor** owns the control actions, which a Markov latent model cannot
represent honestly:

* *death* -- undo, else reset the room. Gem score is monotonic in MazeBench, so
  recovery never costs progress.
* *stuck* -- when the room stops yielding new patterns, the local EFE gradient
  is flat and more planning will not help. Jump to the best frontier room with
  ``goto_level``. This is a third, room-level tier of the hierarchy: the same
  novelty logic as the planner, applied over the room graph instead of tiles.
* *budget* -- stop when the action ceiling is reached.

Keeping these out of the search space is deliberate. Letting the planner treat
``reset`` as an ordinary move invites a reset loop that farms "novel" states,
and ``undo`` depends on history the latent state does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from agent.efe import EFE, EFEConfig
from agent.memory import MOVE_DELTAS, EpisodicMemory, window_hash
from agent.planner import CEMPlanner, PlanResult, PlannerConfig
from env.mazebench_env import MazeBenchEnv, PlayerDead
from env.obs import encode


#: Ablation arms. The research question is whether an action-conditioned world
#: model *with persistent episodic memory* beats a **memoryless** baseline at
#: reward-free exploration -- so "beats uniform random" is not an answer to it.
#: Random shows the planner works at all; `memoryless` is the arm that isolates
#: what memory contributes; `no_epistemic` isolates what the curiosity half of
#: EFE contributes over a purely pragmatic planner.
#:
#:   full            everything
#:   memoryless      same world model and planner, but no persistent record of
#:                   anything seen: cell / pattern / box / abstract novelty all
#:                   zeroed and the frontier-jump supervisor disabled. Ensemble
#:                   disagreement stays, because model uncertainty is not
#:                   episodic memory -- that is the point of the comparison.
#:   no_epistemic    pragmatic terms only; disagreement and all novelty off.
#:   pragmatic_only  as no_epistemic, and ambiguity off too: plain reward
#:                   maximisation through the world model, no EFE at all.
ARMS = ("full", "memoryless", "no_epistemic", "pragmatic_only")


def apply_arm(efe_config, arm: str):
    """Return a copy of ``efe_config`` with the arm's terms switched off."""
    from agent.efe import EFEConfig

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    out = EFEConfig(**dict(efe_config.__dict__))
    if arm in ("memoryless", "no_epistemic", "pragmatic_only"):
        out.w_cell_novelty = 0.0
        out.w_pattern_novelty = 0.0
        out.w_abstract_novelty = 0.0
        out.w_box_novelty = 0.0
        out.use_pattern_novelty = False
    if arm in ("no_epistemic", "pragmatic_only"):
        out.w_disagreement = 0.0
    if arm == "pragmatic_only":
        out.w_ambiguity = 0.0
    return out


@dataclass
class AgentConfig:
    window_size: int = 15
    exploit: float = 1.0            # 0 = reward-free exploration protocol
    arm: str = "full"
    stuck_patience: int = 24        # steps without a new pattern before jumping
    min_room_novelty: float = 0.06  # below this, a room counts as exhausted
    jump_cooldown: int = 20
    device: str = "cpu"
    safety_mask: bool = True        # refuse to step onto a visible hazard glyph
    efe: EFEConfig = field(default_factory=EFEConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)


@dataclass
class StepRecord:
    """One decision, enough to replay and audit it afterwards."""

    step: int
    kind: str                      # "plan" | "recover" | "jump"
    action: int | None
    action_name: str
    room: str
    gem_count: int
    board: str
    efe: float | None = None
    terms: dict[str, float] = field(default_factory=dict)
    plan: list[int] = field(default_factory=list)
    imagined: list[np.ndarray] = field(default_factory=list)
    real_window: np.ndarray | None = None
    novelty: dict[str, float] = field(default_factory=dict)
    note: str = ""


class EFEAgent:
    def __init__(self, model, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device).eval()
        self.memory = EpisodicMemory()
        self.efe = EFE(
            self.memory,
            apply_arm(self.config.efe.scaled(self.config.exploit), self.config.arm),
        )
        self.planner = CEMPlanner(
            self.model, self.efe, self.config.planner, device=self.config.device
        )
        self._steps_since_new_pattern = 0
        self._steps_since_jump = 10 ** 9
        self._last_action: int | None = None
        self._last_window: np.ndarray | None = None
        self._last_room: str = ""
        self._blocked: set[int] = set()
        self._escape_steps = 0
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ util
    def reset(self) -> None:
        self.memory = EpisodicMemory()
        self.efe.memory = self.memory
        self.planner.reset()
        self._steps_since_new_pattern = 0
        self._steps_since_jump = 10 ** 9
        self._last_action = None
        self._last_window = None
        self._last_room = ""
        self._blocked = set()
        self._escape_steps = 0

    def _tensors(self, encoded) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(encoded.window)[None].to(self.device)
        scalars = torch.from_numpy(encoded.scalars)[None].to(self.device)
        return window, scalars

    def _derive_moved(self, window: np.ndarray) -> bool | None:
        """Self-derived movement signal: identical window means the move failed.

        The benchmark never reports a blocked move, so this is the only honest
        way to know. It is also why the world model has a ``moved`` head.
        """
        if self._last_action is None or self._last_action >= 4 or self._last_window is None:
            return None
        return not np.array_equal(window, self._last_window)

    def _action_mask(self, window: np.ndarray) -> torch.Tensor | None:
        """Which primitives the planner may consider. Two independent guards.

        **Blocked moves.** Nothing in the EFE objective stops the planner
        re-proposing a move the world refuses: the `moved` head says it will
        work, it does not, and the next decision starts from the same state and
        reaches the same conclusion. One observed run jammed at a 0.98 blocked
        rate for 150 actions, visiting 4 distinct states. So a move that
        produced no change to the window is masked until something changes.
        The signal is the agent's own percept -- the same one the `moved` head
        is trained on -- not the engine's evaluator-only `moved` flag.

        **Hazards.** The learned death head sees death in ~2% of transitions,
        and relying on it alone the agent died more often than uniform random.
        Hazard glyphs are visible in the agent's own board, so they are read
        directly.

        Movement is screen-relative and the board renders in screen space, so
        the cell "above" the player is simply the one above the window centre.
        No yaw correction belongs here; ``test_movement_is_screen_relative``
        pins that down.

        The hazard half only holds under canonical glyphs -- ``hide_names``
        permutes everything but player and gem, so set ``safety_mask=False``
        there and let the model learn it. The blocked half is glyph-agnostic
        and always applies.
        """

        from env.glyphs import HAZARD_IDS

        size = window.shape[0]
        centre = size // 2
        mask = torch.ones(8, dtype=torch.bool, device=self.device)

        for action in self._blocked:
            if action < 4:
                mask[action] = False

        if self.config.safety_mask:
            for action, (dr, dc) in enumerate(MOVE_DELTAS):
                r, c = centre + dr, centre + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                if int(window[r, c]) in HAZARD_IDS:
                    mask[action] = False

        if not mask[:4].any():
            # Every move is blocked or lethal. Rotations stay available, but
            # they cannot free the agent, so clear the blocked set and let it
            # retry rather than spin. The escape burst handles the rest.
            self._blocked.clear()
            mask[:4] = True
        return mask

    # ---------------------------------------------------------------- ingest
    def perceive(self, obs: dict) -> tuple[object, dict[str, float]]:
        encoded = encode(obs, self.config.window_size)
        moved = self._derive_moved(encoded.window)
        novelty = self.memory.observe(
            room_id=encoded.room_id,
            window=encoded.window,
            gem_count=encoded.gem_count,
            yaw=int(obs.get("yaw") or 0),
            moved=moved,
            last_action=self._last_action,
            died=bool(obs.get("player_dead")),
        )
        if self._last_room and self._last_room != encoded.room_id:
            self.memory.note_room_link(self._last_room, encoded.room_id, self._last_action)
        self._last_room = encoded.room_id

        # Track which moves the world is currently refusing, from our own
        # percept. Cleared the moment anything actually changes, so the set
        # never outlives the situation that produced it.
        if self._last_action is not None and self._last_action < 4:
            if moved is False:
                self._blocked.add(self._last_action)
            elif moved:
                self._blocked.clear()
        if novelty["room_changed"]:
            self._blocked.clear()

        if novelty["new_pattern"]:
            self._steps_since_new_pattern = 0
        else:
            self._steps_since_new_pattern += 1
        self._steps_since_jump += 1
        self._last_window = encoded.window.copy()

        # record the real box layout, so the planner's box-novelty term is
        # scored against layouts the agent has actually produced
        with torch.no_grad():
            window, scalars = self._tensors(encoded)
            layout = int(self.efe.box_layout_keys(window)[0])
            novelty["new_box_layout"] = float(
                self.memory.record_box_layout(encoded.room_id, layout)
            )
            # feed the level-2 coverage memory
            _, h = self.model.encode(window, scalars)
            self.memory.add_abstract_all(
                [state[0].cpu().numpy() for state in self.model.lift_all(h)]
            )
        return encoded, novelty

    # ------------------------------------------------------------- decisions
    def _should_jump(self, obs: dict) -> tuple[bool, str]:
        if self.config.arm != "full":
            # The frontier jump reads the room graph, which is episodic memory.
            # Every ablation arm below `full` must do without it, or the
            # comparison measures the planner and the supervisor together.
            return False, ""
        if self._steps_since_jump < self.config.jump_cooldown:
            return False, ""
        if self._steps_since_new_pattern < self.config.stuck_patience:
            return False, ""
        record = self.memory.rooms.get(self._last_room)
        if record is None or record.novelty_rate > self.config.min_room_novelty:
            return False, ""
        best = self.memory.best_frontier_room(exclude=self._last_room)
        if best is None or best[1] <= 0:
            # Stuck with nowhere known to jump to -- the case that produced the
            # 150-action lock-up, because the starting room is the only room.
            # Fall back to a short forced-random burst, which is enough to
            # break a planner that has settled into a refused action.
            self._escape_steps = max(self._escape_steps, 8)
            self._steps_since_new_pattern = 0
            return False, ""
        return True, best[0]

    # ------------------------------------------------------------- main loop
    def run_episode(
        self,
        env: MazeBenchEnv,
        max_steps: int = 256,
        on_step=None,
    ) -> list[StepRecord]:
        """Play one rollout. Returns a record per decision."""
        self.reset()
        obs = env.reset()
        history: list[StepRecord] = []

        for step in range(max_steps):
            encoded, novelty = self.perceive(obs)

            # --- supervisor: death ---
            if obs.get("player_dead"):
                obs, done, _ = env.recover()
                record = StepRecord(
                    step=step, kind="recover", action=None, action_name="undo/reset",
                    room=encoded.room_id, gem_count=encoded.gem_count,
                    board=str(obs.get("level") or ""), novelty=novelty,
                    note="died; recovered",
                )
                history.append(record)
                if on_step:
                    on_step(record)
                self._last_action = None
                if done:
                    break
                continue

            # --- supervisor: room exhausted ---
            jump, target_room = self._should_jump(obs)
            if jump:
                x, y = target_room.replace("level_", "").split("x")
                try:
                    obs, done, _ = env.control("goto_level", x=x, y=y)
                    note = f"room exhausted; jumped to {target_room}"
                except Exception as error:                      # unvisited / rejected
                    obs, done, _ = env.step(0)
                    note = f"jump to {target_room} refused ({error})"
                self._steps_since_jump = 0
                self._steps_since_new_pattern = 0
                record = StepRecord(
                    step=step, kind="jump", action=None, action_name="goto_level",
                    room=encoded.room_id, gem_count=encoded.gem_count,
                    board=str(obs.get("level") or ""), novelty=novelty, note=note,
                )
                history.append(record)
                if on_step:
                    on_step(record)
                self._last_action = None
                if done:
                    break
                continue

            # --- escape burst: break out of a refused action ---
            mask = self._action_mask(encoded.window)
            if self._escape_steps > 0:
                self._escape_steps -= 1
                allowed = [a for a in range(8) if mask is None or bool(mask[a])]
                action = int(self._rng.choice(allowed))
                try:
                    obs, done, _ = env.step(action)
                except PlayerDead:
                    obs, done, _ = env.recover()
                from env.actions import ACTION_NAMES as _NAMES
                record = StepRecord(
                    step=step, kind="escape", action=action, action_name=_NAMES[action],
                    room=encoded.room_id, gem_count=encoded.gem_count,
                    board=str(obs.get("level") or ""), novelty=novelty,
                    note="stuck with no frontier room; forced escape",
                )
                history.append(record)
                if on_step:
                    on_step(record)
                self._last_action = action
                if done:
                    break
                continue

            # --- planner ---
            window, scalars = self._tensors(encoded)
            with torch.no_grad():
                z, _ = self.model.encode(window, scalars)
            plan = self.planner.plan(
                z0=z,
                start_pos=self.memory.est_pos,
                start_room=encoded.room_id,
                start_yaw=int(obs.get("yaw") or 0),
                action_mask=mask,
            )

            try:
                obs, done, _ = env.step(plan.action)
            except PlayerDead:
                obs, done, _ = env.recover()
                done = done or False

            from env.actions import ACTION_NAMES
            record = StepRecord(
                step=step, kind="plan", action=plan.action,
                action_name=ACTION_NAMES[plan.action], room=encoded.room_id,
                gem_count=encoded.gem_count, board=str(obs.get("level") or ""),
                efe=plan.efe, terms=plan.term_breakdown,
                plan=plan.sequence.tolist(), imagined=plan.imagined,
                real_window=encoded.window, novelty=novelty,
            )
            history.append(record)
            if on_step:
                on_step(record)
            self._last_action = plan.action
            if done:
                break

        return history


class RandomAgent:
    """Uniform-random baseline over the same plannable action set.

    The control that makes an exploration number mean anything. Shares the
    supervisor's death recovery so the comparison is not just "the baseline
    fell in a hole".
    """

    def __init__(self, seed: int = 0, window_size: int = 15):
        self.rng = np.random.default_rng(seed)
        self.window_size = window_size

    def run_episode(self, env: MazeBenchEnv, max_steps: int = 256, on_step=None) -> list[StepRecord]:
        from env.actions import ACTION_NAMES, N_ACTIONS

        obs = env.reset()
        history: list[StepRecord] = []
        for step in range(max_steps):
            if obs.get("player_dead"):
                obs, done, _ = env.recover()
                if done:
                    break
                continue
            action = int(self.rng.integers(N_ACTIONS))
            try:
                obs, done, _ = env.step(action)
            except PlayerDead:
                obs, done, _ = env.recover()
            record = StepRecord(
                step=step, kind="plan", action=action, action_name=ACTION_NAMES[action],
                room=str(obs.get("current_room") or ""), gem_count=int(obs.get("gem_count") or 0),
                board=str(obs.get("level") or ""),
            )
            history.append(record)
            if on_step:
                on_step(record)
            if done:
                break
        return history
