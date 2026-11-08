"""Planning by minimising expected free energy.

Cross-entropy method over discrete action sequences, rolled out entirely inside
the world model. No engine calls -- that is the whole point of having a world
model, and it is what "imagine a state where the box sits on the gem" means
operationally: the tile head decodes every imagined step, so a candidate plan
can be *rendered and read*, not just scored.

Receding horizon: plan ``horizon`` steps, execute only the first action, replan.
The previous plan is shifted by one and used as the prior for the next, so
consecutive decisions are consistent instead of jittering.

The hierarchy enters twice. Level 1 drives the primitive rollout. Level 2 then
extends the horizon past it in coarse chunks, scoring where a plan *leads*
rather than only where it ends -- 4 extra level-2 calls buy 16 more steps of
lookahead for a fraction of the cost of simulating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from agent.efe import EFE, advance_positions, advance_yaws
from env.actions import N_ACTIONS

UNKNOWN_ROOM = "?"      # a room the plan believes it has entered but never seen


@dataclass
class PlannerConfig:
    horizon: int = 8
    n_samples: int = 256
    n_elite: int = 32
    n_iters: int = 4
    temperature: float = 1.0
    prior_weight: float = 0.3     # how much of the shifted previous plan to keep
    elite_smoothing: float = 0.1  # Laplace smoothing when refitting logits
    keep_top_k_traces: int = 3    # imagined rollouts returned for inspection


@dataclass
class PlanResult:
    action: int
    efe: float
    sequence: np.ndarray                       # (horizon,) the chosen plan
    term_breakdown: dict[str, float]           # first-step terms of the chosen plan
    imagined: list[np.ndarray] = field(default_factory=list)   # decoded windows, step by step
    candidate_efe: np.ndarray | None = None    # (n_samples,) final-iteration scores
    logits: np.ndarray | None = None           # (horizon, N_ACTIONS) fitted plan


class CEMPlanner:
    def __init__(self, model, efe: EFE, config: PlannerConfig | None = None, device: str = "cpu"):
        self.model = model
        self.efe = efe
        self.config = config or PlannerConfig()
        self.device = torch.device(device)
        self._prior: Tensor | None = None       # (horizon, N_ACTIONS)

    # ------------------------------------------------------------------ util
    def reset(self) -> None:
        self._prior = None

    def _initial_logits(self, mask: Tensor | None) -> Tensor:
        t, a = self.config.horizon, N_ACTIONS
        logits = torch.zeros(t, a, device=self.device)
        if self._prior is not None:
            shifted = torch.zeros_like(logits)
            shifted[:-1] = self._prior[1:]
            logits = self.config.prior_weight * shifted
        if mask is not None:
            logits = logits.masked_fill(~mask[None, :], -1e9)
        return logits

    # --------------------------------------------------------------- rollout
    @torch.no_grad()
    def _rollout(
        self,
        z0: Tensor,
        actions: Tensor,
        start_pos: tuple[int, int],
        start_room: str,
        start_yaw: int,
        collect_traces: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor], list[Tensor]]:
        """Score ``actions`` (T, N) under the world model. Returns (G, first-step terms, traces)."""
        cfg, efe_cfg = self.config, self.efe.config
        t_steps, n = actions.shape
        z = z0.expand(n, -1, -1, -1).contiguous()

        positions = np.zeros((n, 2), dtype=np.int64)
        positions[:] = start_pos
        yaws = np.full(n, int(start_yaw), dtype=np.int64)
        rooms = [start_room] * n

        total = torch.zeros(n, device=z.device)
        first_terms: dict[str, Tensor] = {}
        traces: list[Tensor] = []

        for t in range(t_steps):
            action_t = actions[t]
            prediction = self.model.imagine_step(z, action_t)
            decoded = prediction["tile_logits"].argmax(dim=1)

            contribution, terms = self.efe.step_terms(prediction, positions, rooms, decoded)
            total = total + (efe_cfg.discount ** t) * contribution
            if t == 0:
                first_terms = {k: v.detach() for k, v in terms.items()}
            if collect_traces:
                traces.append(decoded[:collect_traces].detach().to("cpu", torch.int16))

            # advance the agent's belief about where it is
            moved_prob = torch.sigmoid(prediction["moved_logit"]).cpu().numpy()
            room_prob = torch.sigmoid(prediction["room_change_logit"]).cpu().numpy()
            acts = action_t.cpu().numpy()
            positions = advance_positions(positions, acts, moved_prob, yaws)
            yaws = advance_yaws(yaws, acts)
            for i in range(n):
                if room_prob[i] >= 0.5 and rooms[i] != UNKNOWN_ROOM:
                    rooms[i] = UNKNOWN_ROOM        # believed new room: maximally novel
                    positions[i] = (0, 0)

            z = prediction["z"]

        # --- hierarchical terminal value ---
        total = total + self._terminal(z, actions, efe_cfg.discount ** t_steps)
        return total, first_terms, traces

    @torch.no_grad()
    def _terminal(self, z_final: Tensor, actions: Tensor, discount: float) -> Tensor:
        """Extend the horizon with every abstract level and score where plans lead.

        Each level runs at its own timescale, so the stack covers progressively
        longer futures for a near-constant cost: with ``terminal_chunks=4`` and
        strides 4/16/64, a 2-level model looks 16 primitive steps past the
        planning horizon, a 3-level model 64, a 4-level model 256. That is the
        entire case for adding a level -- the lookahead grows geometrically
        while the work per decision grows linearly.

        Each level's novelty is scored against *its own* coverage store, and
        discounted by how far ahead in primitive time it reaches, so a coarse
        level cannot shout down a fine one just by looking further.
        """
        efe_cfg = self.efe.config
        n = z_final.shape[0]
        if efe_cfg.w_abstract_novelty == 0.0 or efe_cfg.terminal_chunks <= 0:
            return torch.zeros(n, device=z_final.device)

        levels = self.model.abstract
        if len(levels) == 0:
            return torch.zeros(n, device=z_final.device)

        value = torch.zeros(n, device=z_final.device)
        state = self.model.encoder.pool(z_final)
        reach = 0                        # primitive steps past the horizon

        for level_index, level in enumerate(levels):
            state = level.abstract(state)
            # The plan's tail, cycled, as this level's action chunk. The exact
            # continuation matters far less than whether the plan heads
            # somewhere new.
            span = level.stride
            tail = actions[-span:] if actions.shape[0] >= span else actions
            chunk = tail.transpose(0, 1).contiguous()
            if chunk.shape[1] < span:
                pad = chunk[:, -1:].expand(n, span - chunk.shape[1])
                chunk = torch.cat([chunk, pad], dim=1)

            value = value + self.efe.terminal_value(state, level=level_index) * (
                discount * (efe_cfg.discount ** reach)
            )
            rolled = state
            for step in range(1, efe_cfg.terminal_chunks + 1):
                rolled = level(rolled, chunk)
                reach += span
                value = value + self.efe.terminal_value(rolled, level=level_index) * (
                    discount * (efe_cfg.discount ** reach)
                )
        return value

    # ------------------------------------------------------------------ plan
    @torch.no_grad()
    def plan(
        self,
        z0: Tensor,
        start_pos: tuple[int, int],
        start_room: str,
        start_yaw: int,
        action_mask: Tensor | None = None,
    ) -> PlanResult:
        cfg = self.config
        logits = self._initial_logits(action_mask)

        elite_seqs: Tensor | None = None
        scores: Tensor | None = None
        for iteration in range(cfg.n_iters):
            probs = torch.softmax(logits / cfg.temperature, dim=-1)        # (T, A)
            samples = torch.multinomial(probs, cfg.n_samples, replacement=True)  # (T, N)
            scores, _, _ = self._rollout(z0, samples, start_pos, start_room, start_yaw)

            n_elite = min(cfg.n_elite, cfg.n_samples)
            elite_idx = torch.topk(-scores, n_elite).indices            # minimise G
            elite_seqs = samples[:, elite_idx]                          # (T, n_elite)

            counts = torch.zeros_like(logits)
            counts.scatter_add_(
                1, elite_seqs, torch.ones_like(elite_seqs, dtype=counts.dtype)
            )
            freq = (counts + cfg.elite_smoothing) / (n_elite + cfg.elite_smoothing * N_ACTIONS)
            logits = torch.log(freq)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask[None, :], -1e9)

        self._prior = logits.detach()

        # re-score the fitted plan deterministically, and keep its imagination
        best = logits.argmax(dim=-1)                                     # (T,)
        best_batch = best[:, None].expand(-1, max(1, cfg.keep_top_k_traces)).contiguous()
        final_scores, first_terms, traces = self._rollout(
            z0, best_batch, start_pos, start_room, start_yaw,
            collect_traces=cfg.keep_top_k_traces,
        )

        breakdown = {k: float(v[0]) for k, v in first_terms.items()}
        return PlanResult(
            action=int(best[0]),
            efe=float(final_scores[0]),
            sequence=best.cpu().numpy(),
            term_breakdown=breakdown,
            imagined=[t[0].numpy() for t in traces],
            candidate_efe=None if scores is None else scores.cpu().numpy(),
            logits=logits.cpu().numpy(),
        )
