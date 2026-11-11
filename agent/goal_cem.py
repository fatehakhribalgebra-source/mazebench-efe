"""CEM planning to a goal screenshot, flat or coarse-to-fine over the hierarchy.

This is the evaluation protocol the papers use: give the agent a goal image,
let it plan with a learned world model, measure how often it arrives. DINO-WM
reports 81.6% on PointMaze and 64.1% on Wall; jepa-wms reports 83.9% and 78.8%.
Those are the numbers to beat on MazeBench.

**What the hierarchy actually buys.** Not a smaller action space -- a level-3
"action" spans 16 primitive moves, and 4^16 is not a space you sample from. It
buys *cheap evaluation*. A 64-step plan costs 4 predictor calls at stride 16
and 64 at stride 1, so the top level can rank hundreds of candidate plans for
the price of a handful at the bottom. Candidates are always sampled in
primitive action space; the levels filter them coarse-to-fine, and the bottom
level scores the finalists exactly.

With ``n_levels=1`` this degrades to ordinary flat CEM, which is the published
baseline. Keep that arm working -- it is how we find out whether the hierarchy
earns its place, and it is the comparison a reader will ask for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from model.hjepa import N_ACTIONS, goal_cost


@dataclass
class GoalCEMConfig:
    horizon: int = 64          # primitive steps planned over
    n_samples: int = 512       # candidates sampled per iteration
    n_iters: int = 3           # CEM refinement rounds
    elite_frac: float = 0.125  # top fraction kept to refit the distribution
    coarse_keep: int = 64      # survivors after the top level
    mid_keep: int = 16         # survivors after the middle level
    temperature: float = 1.0
    smoothing: float = 0.1     # Laplace smoothing when refitting logits
    replan_every: int = 1      # primitive steps between replans
    terminal_weight: float = 0.1   # see _score_at_level; 1.0 = terminal-only
    seed: int | None = None


class GoalCEM:
    """Plans primitive action sequences toward a goal latent."""

    def __init__(self, model, config: GoalCEMConfig | None = None, device: str = "cpu"):
        self.model = model
        self.config = config or GoalCEMConfig()
        self.device = device
        self.n_levels = len(model.strides)
        self.generator = None
        if self.config.seed is not None:
            self.generator = torch.Generator(device=device).manual_seed(self.config.seed)
        self._logits = None

    # -- internals ---------------------------------------------------------

    def _fresh_logits(self) -> Tensor:
        return torch.zeros(self.config.horizon, N_ACTIONS, device=self.device)

    def _sample(self, logits: Tensor, n: int) -> Tensor:
        probs = torch.softmax(logits / self.config.temperature, dim=-1)
        flat = probs.unsqueeze(0).expand(n, -1, -1).reshape(-1, N_ACTIONS)
        drawn = torch.multinomial(flat, 1, generator=self.generator)
        return drawn.view(n, self.config.horizon)

    def _score_at_level(self, context: Tensor, plans: Tensor, goal: Tensor, level: int) -> Tensor:
        """Cost of each plan, evaluated at ``level``.

        ``plans`` ``(n, horizon)``. The horizon is cut into ``stride``-sized
        chunks, each chunk being one level-step. A plan shorter than one chunk
        is scored at the level below.

        The cost is the **closest approach** along the trajectory, not the
        terminal distance. Terminal-only cost asks the agent to be standing on
        the goal at exactly step ``horizon``, so a plan that arrives at step 4
        and drifts away by step 8 is scored as a failure -- and under MPC, where
        only the first action is executed and the horizon slides forward every
        step, the agent never commits to arriving at all. Scoring the minimum
        says "get there and it does not matter when", which is the actual task.
        ``terminal_weight`` keeps a little pressure on the endpoint as a
        tie-break between plans that both touch the goal.
        """
        stride = self.model.strides[level]
        n, horizon = plans.shape
        chunks = horizon // stride
        if chunks == 0:
            return self._score_at_level(context, plans, goal, max(level - 1, 0))

        used = plans[:, : chunks * stride].reshape(n, chunks, stride)
        ctx = context.unsqueeze(0).expand(n, -1, -1, -1)
        rolled = self.model.rollout(ctx, used, level=level)

        per_step = goal_cost(rolled, goal)          # (n, chunks)
        closest = per_step.min(dim=1).values
        weight = self.config.terminal_weight
        return (1.0 - weight) * closest + weight * per_step[:, -1]

    def _filter(self, context: Tensor, plans: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        """Coarse-to-fine: rank at the top level, re-rank survivors further down."""
        keep = (self.config.coarse_keep, self.config.mid_keep)
        for depth, level in enumerate(range(self.n_levels - 1, 0, -1)):
            costs = self._score_at_level(context, plans, goal, level)
            n_keep = min(keep[min(depth, len(keep) - 1)], plans.shape[0])
            plans = plans[costs.argsort()[:n_keep]]
        costs = self._score_at_level(context, plans, goal, 0)
        return plans, costs

    # -- public ------------------------------------------------------------

    def reset(self) -> None:
        self._logits = None

    @torch.no_grad()
    def plan(self, context: Tensor, goal: Tensor) -> dict:
        """``context`` ``(W, N, D)``, ``goal`` ``(N, D)``.

        Returns the chosen action, the full best plan, and its cost. The
        distribution is carried across calls (shifted by one step), so
        replanning refines the previous plan instead of starting cold -- the
        same trick the paper's MPC loop uses, and it matters more at long
        horizons than the sample count does.
        """
        config = self.config
        logits = self._logits if self._logits is not None else self._fresh_logits()

        best_plan, best_cost = None, None
        for _ in range(config.n_iters):
            plans = self._sample(logits, config.n_samples)
            survivors, costs = self._filter(context, plans, goal)

            order = costs.argsort()
            n_elite = max(2, int(len(order) * config.elite_frac))
            elites = survivors[order[:n_elite]]

            if best_cost is None or costs[order[0]] < best_cost:
                best_cost = costs[order[0]].item()
                best_plan = survivors[order[0]].clone()

            counts = torch.zeros_like(logits)
            counts.scatter_add_(
                1,
                elites.t(),
                torch.ones(config.horizon, n_elite, device=self.device),
            )
            logits = torch.log(counts + config.smoothing)

        # Carry the plan forward one step for the next call.
        self._logits = torch.cat([logits[1:], self._fresh_logits()[:1]], dim=0)
        return {
            "action": int(best_plan[0].item()),
            "plan": best_plan,
            "cost": best_cost,
        }
