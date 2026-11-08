"""Expected Free Energy.

For a candidate action sequence (a "policy" in the active-inference sense) the
expected free energy over a horizon is

    G(pi) = sum_t gamma^t [ risk_t + ambiguity_t - epistemic_t ]

and the agent acts to **minimise** it. The three parts, and how each is
estimated here:

**risk** -- ``-ln P(o_t | C)``, the divergence between what the agent expects to
observe and what it prefers to observe. ``C`` is the preference model: gems
collected, new rooms entered, novel block configurations, staying alive. Since
the normaliser of ``P(o|C)`` is constant across candidate policies, risk is
computed as the negative of a utility ``U``. Gems alone are far too sparse to
steer an 8-step horizon, so ``U`` also carries a dense proximity term read off
the *imagined* board -- how close the imagined player is to a visible gem
glyph. That term is what turns "prefer gems" into something a short-horizon
planner can act on.

**ambiguity** -- ``E_q[ H[P(o|s)] ]``, the expected entropy of observations
given the state. Penalised: among two states with equal information gain, the
agent prefers the one whose observations it can actually predict. Estimated as
the mean per-tile predictive entropy of the tile head. This term is what stops
the epistemic drive degenerating into seeking noise.

**epistemic** -- expected information gain, rewarded. Two components, matching
the two things the agent is uncertain about:

* *parameters* -- how much this transition would teach the world model,
  estimated by disagreement across the predictor ensemble (Plan2Explore /
  latent disagreement, the standard tractable proxy for Bayesian surprise).
* *state* -- how much it would learn about where it is, estimated by
  count-based novelty from episodic memory over dead-reckoned cells and
  egocentric patterns.

A terminal term from the level-2 model adds coarse long-horizon novelty beyond
the primitive horizon, so the hierarchy contributes to the objective rather
than just existing.

Every quantity is derived from agent-visible observations, the agent's own
actions, or its own model. Nothing here reads an evaluator-only field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from agent.memory import MOVE_DELTAS, EpisodicMemory, rotate_delta, window_hash
from env.glyphs import GEM_IDS, HAZARD_IDS, PUSHABLE_IDS


@dataclass
class EFEConfig:
    horizon: int = 8
    discount: float = 0.95

    # --- preference model (risk term) ---
    w_gem: float = 10.0            # expected gems collected this step
    w_gem_proximity: float = 1.0   # dense shaping toward a visible gem
    w_room_change: float = 2.0     # entering a room at all
    w_box_novelty: float = 0.5     # novel block configuration
    w_death: float = 20.0          # strongly dispreferred
    w_hazard_proximity: float = 0.3

    # --- ambiguity term ---
    w_ambiguity: float = 0.5

    # --- epistemic terms ---
    w_disagreement: float = 2.0
    w_cell_novelty: float = 3.0
    w_pattern_novelty: float = 2.0

    # --- hierarchical terminal value ---
    w_abstract_novelty: float = 1.0
    terminal_chunks: int = 4

    # --- practical switches ---
    use_pattern_novelty: bool = True
    gem_proximity_cap: float = 8.0

    def scaled(self, exploit: float) -> "EFEConfig":
        """Tilt the balance between pragmatic and epistemic drive.

        ``exploit`` in [0, 1]: 0 is pure exploration (reward-free protocol),
        1 leaves the preference weights alone and damps curiosity. Used to run
        the *same* agent under both the benchmark's gem objective and the
        reward-free exploration protocol, which is the comparison that makes an
        exploration claim meaningful.
        """
        out = EFEConfig(**{k: v for k, v in self.__dict__.items()})
        out.w_gem *= exploit
        out.w_gem_proximity *= exploit
        out.w_disagreement *= (1.0 - 0.5 * exploit)
        out.w_cell_novelty *= (1.0 - 0.5 * exploit)
        out.w_pattern_novelty *= (1.0 - 0.5 * exploit)
        return out


def _distance_to_centre(size: int, device: torch.device) -> Tensor:
    """Chebyshev distance from each cell of the window to the player at centre."""
    idx = torch.arange(size, device=device)
    dr = (idx[:, None] - size // 2).abs()
    dc = (idx[None, :] - size // 2).abs()
    return torch.maximum(dr, dc).float()


class EFE:
    """Scores imagined transitions. Stateless apart from the memory it reads."""

    def __init__(self, memory: EpisodicMemory, config: EFEConfig | None = None):
        self.memory = memory
        self.config = config or EFEConfig()
        self._dist_cache: dict[tuple[int, torch.device], Tensor] = {}
        self._gem_ids: Tensor | None = None
        self._hazard_ids: Tensor | None = None
        self._push_ids: Tensor | None = None
        self._zobrist: dict[tuple[int, torch.device], Tensor] = {}

    def _zobrist_weights(self, size: int, device: torch.device) -> Tensor:
        """Fixed random per-cell weights, for hashing a box layout on GPU.

        Extracting per-candidate box coordinates would mean a CPU round trip per
        candidate per step. A Zobrist-style weighted sum over the pushable mask
        gives one integer key per candidate with no sync, which is all the
        count lookup needs.
        """
        key = (size, device)
        if key not in self._zobrist:
            generator = torch.Generator(device="cpu").manual_seed(0xB07E5)
            weights = torch.randint(
                1, 2 ** 40, (size, size), generator=generator, dtype=torch.int64
            )
            self._zobrist[key] = weights.to(device)
        return self._zobrist[key]

    def box_layout_keys(self, decoded: Tensor) -> Tensor:
        """(N, S, S) imagined glyphs -> (N,) int64 layout hashes."""
        device = decoded.device
        if self._push_ids is None or self._push_ids.device != device:
            self._push_ids = torch.tensor(sorted(PUSHABLE_IDS), device=device)
        mask = torch.isin(decoded, self._push_ids).to(torch.int64)
        weights = self._zobrist_weights(decoded.shape[-1], device)
        return (mask * weights[None]).flatten(1).sum(dim=1)

    def _dist_map(self, size: int, device: torch.device) -> Tensor:
        key = (size, device)
        if key not in self._dist_cache:
            self._dist_cache[key] = _distance_to_centre(size, device)
        return self._dist_cache[key]

    def _ids(self, device: torch.device) -> tuple[Tensor, Tensor]:
        if self._gem_ids is None or self._gem_ids.device != device:
            self._gem_ids = torch.tensor(sorted(GEM_IDS), device=device)
            self._hazard_ids = torch.tensor(sorted(HAZARD_IDS), device=device)
        return self._gem_ids, self._hazard_ids

    # ---------------------------------------------------------------- terms
    @staticmethod
    def expected_gem_delta(gem_delta_logits: Tensor) -> Tensor:
        """E[number of gems collected], from the gem-delta head."""
        probs = torch.softmax(gem_delta_logits, dim=-1)
        counts = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        return (probs * counts).sum(-1)

    @staticmethod
    def tile_entropy(tile_logits: Tensor) -> Tensor:
        """Mean per-tile predictive entropy -> the ambiguity term. (N,)"""
        log_probs = torch.log_softmax(tile_logits, dim=1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=1).mean(dim=(1, 2))

    def proximity(self, decoded: Tensor, target_ids: Tensor, cap: float) -> Tensor:
        """Closeness of the nearest target glyph to the player, in [0, 1].

        ``decoded`` is (N, S, S) imagined glyph ids. Returns 1.0 when a target
        sits on the player's own cell and 0.0 when none is visible.
        """
        size = decoded.shape[-1]
        dist = self._dist_map(size, decoded.device)
        mask = torch.isin(decoded, target_ids)
        big = torch.full_like(dist, float(cap) + 1.0)
        per_cell = torch.where(mask, dist[None].expand_as(mask.float()).clone(), big[None].expand_as(mask.float()).clone())
        nearest = per_cell.flatten(1).min(dim=1).values
        return (1.0 - (nearest / cap).clamp(0.0, 1.0))

    # ------------------------------------------------------------ per step
    def step_terms(
        self,
        prediction: dict[str, Tensor],
        positions: np.ndarray,
        room_ids: list[str],
        decoded: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """EFE contribution of one imagined step, for N candidates at once.

        ``prediction`` is the output of ``HJEPA.imagine_step``.
        ``positions`` is (N, 2) dead-reckoned cell estimates.
        ``room_ids`` is the believed room per candidate.
        Returns (per-candidate EFE contribution, term breakdown).
        """
        c = self.config
        device = prediction["z"].device
        if decoded is None:
            decoded = prediction["tile_logits"].argmax(dim=1)
        gem_ids, hazard_ids = self._ids(device)

        # --- risk: negative utility of the predicted observation ---
        e_gems = self.expected_gem_delta(prediction["gem_delta_logits"])
        p_room = torch.sigmoid(prediction["room_change_logit"])
        p_death = torch.sigmoid(prediction["death_logit"])
        gem_prox = self.proximity(decoded, gem_ids, c.gem_proximity_cap)
        hazard_prox = self.proximity(decoded, hazard_ids, 3.0)

        # Box layouts: the preference the user's "imagine the cube on top of
        # the diamond" case needs. The tile head decodes the imagined board, so
        # a plan that pushes a block somewhere new is visible *as a layout* and
        # scored against layouts already produced -- matching the benchmark's
        # block_progress_score, which pays for novel positions and not for
        # pushing back and forth.
        if c.w_box_novelty != 0.0:
            keys = self.box_layout_keys(decoded).cpu().numpy()
            box_nov = torch.tensor(
                [self.memory.box_novelty(room_ids[i], int(keys[i])) for i in range(len(room_ids))],
                device=device, dtype=torch.float32,
            )
        else:
            box_nov = torch.zeros_like(gem_prox)

        utility = (
            c.w_gem * e_gems
            + c.w_gem_proximity * gem_prox
            + c.w_room_change * p_room
            + c.w_box_novelty * box_nov
            - c.w_death * p_death
            - c.w_hazard_proximity * hazard_prox
        )
        risk = -utility

        # --- ambiguity ---
        ambiguity = self.tile_entropy(prediction["tile_logits"])

        # --- epistemic: parameter information gain ---
        disagreement = prediction["disagreement"]

        # --- epistemic: state information gain, from episodic memory ---
        cell_nov = torch.tensor(
            [self.memory.cell_novelty(room_ids[i], (int(positions[i, 0]), int(positions[i, 1])))
             for i in range(len(room_ids))],
            device=device, dtype=torch.float32,
        )
        if c.use_pattern_novelty and c.w_pattern_novelty != 0.0:
            windows = decoded.detach().to("cpu", torch.int16).numpy()
            pattern_nov = torch.tensor(
                [self.memory.pattern_novelty(window_hash(w)) for w in windows],
                device=device, dtype=torch.float32,
            )
        else:
            pattern_nov = torch.zeros_like(cell_nov)

        epistemic = (
            c.w_disagreement * disagreement
            + c.w_cell_novelty * cell_nov
            + c.w_pattern_novelty * pattern_nov
        )

        contribution = risk + c.w_ambiguity * ambiguity - epistemic
        terms = {
            "risk": risk,
            "utility": utility,
            "ambiguity": ambiguity,
            "epistemic": epistemic,
            "e_gems": e_gems,
            "gem_proximity": gem_prox,
            "p_room_change": p_room,
            "p_death": p_death,
            "disagreement": disagreement,
            "cell_novelty": cell_nov,
            "pattern_novelty": pattern_nov,
            "box_novelty": box_nov,
        }
        return contribution, terms

    # ------------------------------------------------------- terminal value
    def terminal_value(self, state: Tensor, level: int = 0) -> Tensor:
        """Coarse novelty of where a rollout ends, at one abstract level.

        Scored against that level's own coverage store, since levels run at
        different timescales and their distances are not comparable. Returned
        as a *cost* (negated novelty), so it adds to G like every other term.
        """
        if self.config.w_abstract_novelty == 0.0:
            return torch.zeros(state.shape[0], device=state.device)
        novelty = self.memory.abstract_novelty(state.detach().cpu().numpy(), level=level)
        return -self.config.w_abstract_novelty * torch.tensor(
            novelty, device=state.device, dtype=torch.float32
        )


# ------------------------------------------------------------------ helpers
def advance_positions(
    positions: np.ndarray,
    actions: np.ndarray,
    moved_prob: np.ndarray,
    yaws: np.ndarray,
    move_threshold: float = 0.5,
) -> np.ndarray:
    """Dead-reckon N candidate positions one step.

    Only move actions displace the agent, and only when the model expects the
    move to succeed. Camera rotations change ``yaw`` instead, which is why the
    deltas are rotated: movement is screen-relative.
    """
    out = positions.copy()
    for i in range(len(positions)):
        a = int(actions[i])
        if a < 4 and moved_prob[i] >= move_threshold:
            dr, dc = rotate_delta(MOVE_DELTAS[a], int(yaws[i]))
            out[i, 0] += dr
            out[i, 1] += dc
    return out


def advance_yaws(yaws: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Camera rotation left/right changes yaw; pitch changes do not."""
    out = yaws.copy()
    for i in range(len(yaws)):
        a = int(actions[i])
        if a == 6:      # rotate_camera left
            out[i] = (out[i] - 1) % 4
        elif a == 7:    # rotate_camera right
            out[i] = (out[i] + 1) % 4
    return out
