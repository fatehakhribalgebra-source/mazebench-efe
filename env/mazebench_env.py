"""Gym-style MazeBench wrapper with a hard observation firewall.

The firewall is the point of this file. ``maze-bridge.js`` hands back the full
authoritative snapshot, including fields the benchmark explicitly withholds from
the model: ``board_state_hash``, ``player`` coordinates, ``collected_gems`` ids,
``push_count``, ``novel_push_count`` and the scorecard. Those are *evaluator*
fields. If the policy ever reads one, any exploration result is worthless,
because the agent was handed the state it was supposed to infer.

So ``step()`` returns a pair:

* ``obs``  -- only ``AGENT_VISIBLE_FIELDS``. This is all the agent ever sees.
* ``info`` -- the privileged remainder, for logging and scoring only.

``EvalLog`` keeps the privileged stream for post-hoc metrics, mirroring how the
real evaluator scores a rollout after the model exits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import PLANNABLE
from .bridge import BridgeConfig, MazeBridge
from .obs import AGENT_VISIBLE_FIELDS

PRIVILEGED_FIELDS = frozenset({
    "board_state_hash", "board_state_hash_version", "player", "collected_gems",
    "push_count", "novel_push_count", "scorecard", "_render_state",
    "pushes_this_action", "novel_pushes_this_action", "collected_this_action",
    "moved", "room_changed", "undone", "reset",
})


def split_observation(snapshot: dict) -> tuple[dict, dict]:
    """Return (agent_visible, privileged)."""
    visible = {k: v for k, v in snapshot.items() if k in AGENT_VISIBLE_FIELDS}
    privileged = {k: v for k, v in snapshot.items() if k not in AGENT_VISIBLE_FIELDS}
    return visible, privileged


class PlayerDead(RuntimeError):
    """The player died. Only ``undo`` / ``reset_level`` are accepted now.

    Death is recoverable in MazeBench and gem score stays monotonic, so this is
    not the end of an episode -- it is a control decision for the supervisor.
    """


@dataclass
class EvalLog:
    """Evaluator-side record. Never read by the agent."""

    steps: int = 0
    gem_count: int = 0
    visited_rooms: set[str] = field(default_factory=set)
    novel_push_count: int = 0
    push_count: int = 0
    unique_board_states: set[str] = field(default_factory=set)
    deaths: int = 0
    rotations: int = 0
    moves: int = 0
    blocked_moves: int = 0
    room_changes: int = 0
    first_gem_step: int | None = None
    gem_steps: list[int] = field(default_factory=list)

    def record(self, privileged: dict, action: int | None) -> None:
        self.steps += 1
        prev_gems = self.gem_count
        self.gem_count = max(self.gem_count, int(privileged.get("gem_count") or 0))
        if self.gem_count > prev_gems:
            self.gem_steps.append(self.steps)
            if self.first_gem_step is None:
                self.first_gem_step = self.steps
        for room in privileged.get("visited_levels") or []:
            self.visited_rooms.add(str(room))
        self.novel_push_count = max(self.novel_push_count, int(privileged.get("novel_push_count") or 0))
        self.push_count = max(self.push_count, int(privileged.get("push_count") or 0))
        if privileged.get("board_state_hash"):
            self.unique_board_states.add(str(privileged["board_state_hash"]))
        if privileged.get("player_dead"):
            self.deaths += 1
        if privileged.get("room_changed"):
            self.room_changes += 1
        if action is not None:
            if action < 4:
                self.moves += 1
                if privileged.get("moved") is False:
                    self.blocked_moves += 1
            else:
                self.rotations += 1

    def reward(self, gem_w: float = 1.0, room_w: float = 0.1, push_w: float = 0.05) -> dict[str, float]:
        """The benchmark's three reward components, same weights as the env."""
        rooms = max(0, len(self.visited_rooms) - 1)
        return {
            "gem_score": gem_w * self.gem_count,
            "room_exploration_score": room_w * rooms,
            "block_progress_score": push_w * self.novel_push_count,
            "total": gem_w * self.gem_count + room_w * rooms + push_w * self.novel_push_count,
        }

    def summary(self) -> dict[str, Any]:
        out = {
            "steps": self.steps,
            "gems": self.gem_count,
            "rooms_visited": len(self.visited_rooms),
            "novel_block_positions": self.novel_push_count,
            "block_pushes": self.push_count,
            "unique_board_states": len(self.unique_board_states),
            "state_novelty_rate": (len(self.unique_board_states) / self.steps) if self.steps else 0.0,
            "deaths": self.deaths,
            "moves": self.moves,
            "blocked_moves": self.blocked_moves,
            "blocked_move_rate": (self.blocked_moves / self.moves) if self.moves else 0.0,
            "rotations": self.rotations,
            "room_changes": self.room_changes,
            "first_gem_step": self.first_gem_step,
            "actions_per_room": (self.steps / max(1, len(self.visited_rooms))),
        }
        out.update(self.reward())
        return out


class MazeBenchEnv:
    """Discrete-action MazeBench with the observation firewall applied."""

    def __init__(
        self,
        bridge_config: BridgeConfig | None = None,
        max_actions: int | None = 256,
        bridge: Any | None = None,
    ):
        self.bridge = bridge if bridge is not None else MazeBridge(bridge_config)
        self.max_actions = max_actions
        self.log = EvalLog()
        self._last_privileged: dict = {}
        self._done = False

    # -- core loop ----------------------------------------------------------
    def reset(self, new_run: bool = True) -> dict:
        if new_run:
            snapshot = self.bridge.reset_run()
        else:
            snapshot = self.bridge.observe()
        self.log = EvalLog()
        self._done = False
        visible, privileged = split_observation(snapshot)
        self._last_privileged = privileged
        return visible

    def step(self, action: int) -> tuple[dict, bool, dict]:
        """Apply a plannable primitive. Returns (visible_obs, done, privileged)."""
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        if self.dead:
            raise PlayerDead("player is dead; call recover()")
        command, kwargs = PLANNABLE[action]
        snapshot = self.bridge.call(command, **kwargs)
        visible, privileged = split_observation(snapshot)
        self.log.record({**visible, **privileged}, action)
        self._last_privileged = privileged
        self._done = self._is_done(visible, privileged)
        return visible, self._done, privileged

    def control(self, command: str, **kwargs) -> tuple[dict, bool, dict]:
        """Apply a control action (undo / reset_level / goto_level / quit)."""
        snapshot = self.bridge.call(command, **kwargs)
        visible, privileged = split_observation(snapshot)
        self.log.record({**visible, **privileged}, None)
        self._last_privileged = privileged
        self._done = self._is_done(visible, privileged)
        return visible, self._done, privileged

    def _is_done(self, visible: dict, privileged: dict) -> bool:
        if visible.get("game_won") or privileged.get("game_won"):
            return True
        if privileged.get("game_lost"):
            return True
        if self.max_actions is not None and self.log.steps >= self.max_actions:
            return True
        return False

    @property
    def dead(self) -> bool:
        return bool(self._last_privileged.get("player_dead"))

    def recover(self) -> tuple[dict, bool, dict]:
        """Get out of a death. Try ``undo`` first, fall back to ``reset_level``.

        ``undo`` costs one action and rewinds the fatal step; ``reset_level``
        costs one action and rewinds the whole room. Gems already collected
        stay collected either way, so recovery never loses score.
        """
        visible, done, privileged = self.control("undo")
        if privileged.get("player_dead"):
            visible, done, privileged = self.control("reset_level")
        return visible, done, privileged

    # -- evaluator side -----------------------------------------------------
    def scorecard(self) -> dict:
        """Authoritative final score. Call AFTER the agent has stopped acting."""
        return self.bridge.scorecard().get("scorecard", {})

    def close(self) -> None:
        self.bridge.close()

    def __enter__(self) -> "MazeBenchEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
