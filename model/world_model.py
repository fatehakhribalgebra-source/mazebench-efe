"""Hierarchical latent world model for MazeBench.

Design, and why it is shaped this way.

MazeBench is a grid of discrete tiles, so the latent is a **grid** too:
``z`` has shape ``(B, C, S, S)``, one feature vector per visible tile, at full
window resolution with no downsampling. A pooled vector ``h`` is derived from it
for the things that want a compact state (level-2 dynamics, disagreement,
novelty). Two consequences matter:

* The one-step predictor is a *residual* conv net. Almost nothing in a grid
  world changes when you take one step, so predicting a delta is far easier
  than predicting the next grid outright.
* A 1x1 conv tile head decodes ``z`` back to glyph logits. That is what makes
  imagination inspectable: a rollout can be rendered as ASCII and read, so
  "imagine the box pushed onto the gem" is a thing you can literally print.
  It also removes any worry about representational collapse at level 1 -- a
  collapsed latent cannot reconstruct the board.

Level 2 is the hierarchy. It runs on pooled latents at a coarser timescale:
one level-2 step consumes a *chunk* of ``chunk_len`` primitive actions. The
planner uses it as a terminal value beyond its primitive horizon, which is
where the hierarchy earns its keep -- scoring 32 steps ahead costs 8 level-2
calls instead of 32 level-1 calls. Level 2 has no decoder, so it is trained
JEPA-style against an EMA target with a VICReg variance/covariance term to stop
it collapsing.

Honest labelling: this is *not* a pure JEPA. A pure JEPA predicts only in
representation space and never reconstructs. Level 1 here is a latent
predictive model with a reconstruction head, which is a deliberate trade --
reconstruction is what gives the EFE planner readable, groundable outcomes on a
tile world, and it makes collapse impossible rather than merely discouraged.
Level 2 is JEPA-style. See README for the full argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from env.actions import N_ACTIONS
from env.glyphs import VOCAB_SIZE
from env.obs import SCALAR_DIM


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


def vicreg_terms(h: Tensor, eps: float = 1e-4) -> tuple[Tensor, Tensor]:
    """Variance and covariance regularizers (Bardes et al., VICReg).

    Variance pushes every feature dimension to keep unit standard deviation
    across the batch; covariance decorrelates dimensions. Together they make
    the trivial constant solution a bad one, which is what a predictive loss
    with a learned target otherwise drifts towards.
    """
    h = h - h.mean(dim=0, keepdim=True)
    std = torch.sqrt(h.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    n, d = h.shape
    if n < 2:
        return var_loss, h.new_zeros(())
    cov = (h.T @ h) / (n - 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss, cov_loss


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------
class GridEncoder(nn.Module):
    """Egocentric glyph window -> latent grid ``z`` and pooled vector ``h``."""

    def __init__(self, channels: int = 64, embed_dim: int = 32, pooled_dim: int = 256, n_blocks: int = 3):
        super().__init__()
        self.channels = channels
        self.pooled_dim = pooled_dim
        self.tile_embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.scalar_proj = nn.Linear(SCALAR_DIM, 16)
        self.stem = nn.Conv2d(embed_dim + 16, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(8, channels)
        self.pool_proj = nn.Sequential(
            nn.Linear(channels * 2, pooled_dim), nn.LayerNorm(pooled_dim)
        )

    def forward(self, window: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        # window: (B, S, S) int64; scalars: (B, SCALAR_DIM) float
        emb = self.tile_embed(window).permute(0, 3, 1, 2)             # (B, E, S, S)
        s = self.scalar_proj(scalars)[:, :, None, None]
        s = s.expand(-1, -1, emb.shape[2], emb.shape[3])              # (B, 16, S, S)
        x = self.stem(torch.cat([emb, s], dim=1))
        z = self.out_norm(self.blocks(x))                             # (B, C, S, S)
        return z, self.pool(z)

    def pool(self, z: Tensor) -> Tensor:
        mean = z.mean(dim=(2, 3))
        peak = z.amax(dim=(2, 3))
        return self.pool_proj(torch.cat([mean, peak], dim=1))         # (B, pooled_dim)


# --------------------------------------------------------------------------
# level 1: one primitive action
# --------------------------------------------------------------------------
class Level1Predictor(nn.Module):
    """``z, a -> z'``, residual. One ensemble member."""

    def __init__(self, channels: int = 64, action_dim: int = 32, n_blocks: int = 2):
        super().__init__()
        self.action_embed = nn.Embedding(N_ACTIONS, action_dim)
        self.stem = nn.Conv2d(channels + action_dim, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.head = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)   # start as identity: z' == z

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        a = self.action_embed(action)[:, :, None, None].expand(-1, -1, z.shape[2], z.shape[3])
        h = self.stem(torch.cat([z, a], dim=1))
        return z + self.head(self.blocks(h))


class PredictorEnsemble(nn.Module):
    """K level-1 predictors over a shared encoder.

    Spread across members is the epistemic signal the EFE planner uses: where
    the members disagree, the world model does not yet know what happens, so
    going there is informative. This is the Plan2Explore / latent-disagreement
    estimator of expected parameter information gain.
    """

    def __init__(self, n_members: int = 5, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([Level1Predictor(**kwargs) for _ in range(n_members)])

    @property
    def n_members(self) -> int:
        return len(self.members)

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        """-> (K, B, C, S, S)"""
        return torch.stack([m(z, action) for m in self.members], dim=0)

    def mean_and_disagreement(self, z: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """-> (mean z' (B,C,S,S), disagreement (B,))"""
        stacked = self(z, action)
        mean = stacked.mean(dim=0)
        if stacked.shape[0] == 1:
            return mean, z.new_zeros(z.shape[0])
        # variance per feature, averaged -> scalar per batch element
        disagreement = stacked.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
        return mean, disagreement


# --------------------------------------------------------------------------
# decoders and grounded heads
# --------------------------------------------------------------------------
class TileHead(nn.Module):
    """Latent grid -> per-tile glyph logits. Makes imagination renderable."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.proj = nn.Conv2d(channels, VOCAB_SIZE, 1)

    def forward(self, z: Tensor) -> Tensor:
        return self.proj(F.gelu(self.norm(z)))        # (B, VOCAB, S, S)


MAX_GEM_DELTA = 4   # classes 0..3 gems in one action; 3 means "3 or more"


class OutcomeHeads(nn.Module):
    """Grounded outcome predictions from a pooled latent.

    Every target here is derived from an *agent-visible* field, so training on
    them leaks nothing. ``gem_delta`` comes from the public ``gem_count``,
    ``room_change`` from the public ``current_room``, ``player_visible`` from
    whether the agent can find ``P`` in its own board.
    """

    def __init__(self, pooled_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.GELU())
        self.gem_delta = nn.Linear(hidden, MAX_GEM_DELTA)
        self.room_change = nn.Linear(hidden, 1)
        self.player_visible = nn.Linear(hidden, 1)
        self.death = nn.Linear(hidden, 1)
        # `moved` is NOT the engine's evaluator-only flag. The benchmark never
        # reports a blocked move, but the agent can derive it from its own
        # percept: an identical egocentric window after a move means blocked.
        # That self-derived label trains this head, and the head is what lets
        # the planner dead-reckon position through an imagined rollout.
        self.moved = nn.Linear(hidden, 1)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        t = self.trunk(h)
        return {
            "gem_delta_logits": self.gem_delta(t),
            "room_change_logit": self.room_change(t).squeeze(-1),
            "player_visible_logit": self.player_visible(t).squeeze(-1),
            "death_logit": self.death(t).squeeze(-1),
            "moved_logit": self.moved(t).squeeze(-1),
        }


# --------------------------------------------------------------------------
# level 2: a chunk of actions
# --------------------------------------------------------------------------
class AbstractLevel(nn.Module):
    """One coarse-timescale JEPA level over the latent of the level below.

    ``abstract`` lifts the level below's latent into this level's space;
    ``forward`` advances it by one chunk of this level's own timescale.
    Trained to match an EMA target of ``abstract`` applied to the true future
    latent, with VICReg to stop it collapsing. No decoder by design -- at a
    coarse timescale the exact board is not predictable and should not be.

    ``stride`` is how many *primitive* actions one step of this level spans:
    4 for the first abstract level, 16 for the second, and so on. Every level
    conditions on the primitive actions inside its span, summarised by a GRU,
    so a level never has to interpret the level below's chunk encoding.
    """

    def __init__(
        self,
        input_dim: int,
        abstract_dim: int = 128,
        chunk_dim: int = 64,
        chunk_len: int = 4,
        stride: int = 4,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.stride = stride
        self.abstract_dim = abstract_dim
        self.lift = nn.Sequential(
            nn.Linear(input_dim, abstract_dim), nn.GELU(),
            nn.Linear(abstract_dim, abstract_dim), nn.LayerNorm(abstract_dim),
        )
        self.action_embed = nn.Embedding(N_ACTIONS, 32)
        self.chunk_rnn = nn.GRU(32, chunk_dim, batch_first=True)
        self.step = nn.Sequential(
            nn.Linear(abstract_dim + chunk_dim, 256), nn.GELU(),
            nn.Linear(256, abstract_dim),
        )

    def encode_chunk(self, actions: Tensor) -> Tensor:
        """actions: (B, k) int64 primitive actions -> (B, chunk_dim)"""
        _, hidden = self.chunk_rnn(self.action_embed(actions))
        return hidden[-1]

    def abstract(self, h: Tensor) -> Tensor:
        return self.lift(h)

    def forward(self, state: Tensor, actions: Tensor) -> Tensor:
        return state + self.step(torch.cat([state, self.encode_chunk(actions)], dim=-1))


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------
@dataclass
class WorldModelConfig:
    window_size: int = 15
    channels: int = 64
    pooled_dim: int = 256
    abstract_dim: int = 128
    n_members: int = 5
    encoder_blocks: int = 3
    predictor_blocks: int = 2
    chunk_len: int = 4
    #: Total levels INCLUDING the primitive one. Only 2, 3 and 4 are
    #: supported -- a fixed small stack, not an arbitrary depth. Default 2 is
    #: the configuration already validated; 3 and 4 are the experiment.
    n_levels: int = 2
    ema_momentum: float = 0.99
    # loss weights
    w_tile: float = 1.0
    w_latent: float = 1.0
    w_gem: float = 2.0          # gems are rare; weight their head up
    w_room: float = 0.5
    w_aux: float = 0.2
    w_level2: float = 1.0
    w_vicreg_var: float = 1.0
    w_vicreg_cov: float = 0.04


class HJEPA(nn.Module):
    def __init__(self, config: WorldModelConfig | None = None):
        super().__init__()
        self.config = config or WorldModelConfig()
        c = self.config
        self.encoder = GridEncoder(c.channels, pooled_dim=c.pooled_dim, n_blocks=c.encoder_blocks)
        self.ensemble = PredictorEnsemble(
            n_members=c.n_members, channels=c.channels, n_blocks=c.predictor_blocks
        )
        self.tile_head = TileHead(c.channels)
        self.outcome_heads = OutcomeHeads(c.pooled_dim)
        # The hierarchy is a fixed small stack, the way H-JEPA is drawn -- not
        # an arbitrary depth. `n_levels` counts the primitive level too, and
        # the only supported choices are 2, 3 and 4, so that "does another
        # level help?" is a three-way experiment rather than a free parameter.
        #
        #   n_levels=2   primitive + level 2            (spans 4 actions)
        #   n_levels=3   + level 3                      (spans 16)
        #   n_levels=4   + level 4                      (spans 64)
        #
        # Each level conditions on the primitive actions inside its own span,
        # so a level never has to interpret the one below's chunk encoding.
        self.abstract = nn.ModuleList()
        input_dim, stride = c.pooled_dim, 1
        for _ in range(c.n_levels - 1):
            stride *= c.chunk_len
            self.abstract.append(AbstractLevel(
                input_dim=input_dim, abstract_dim=c.abstract_dim,
                chunk_len=c.chunk_len, stride=stride,
            ))
            input_dim = c.abstract_dim

        # EMA target encoder, for the JEPA latent term and the abstract levels
        self.target_encoder = GridEncoder(
            c.channels, pooled_dim=c.pooled_dim, n_blocks=c.encoder_blocks
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    # -- hierarchy access ---------------------------------------------------
    @property
    def level2(self) -> "AbstractLevel":
        """The first abstract level. The planner and agent speak this name."""
        return self.abstract[0]

    @property
    def level3(self) -> "AbstractLevel | None":
        return self.abstract[1] if len(self.abstract) > 1 else None

    @property
    def strides(self) -> list[int]:
        """Primitive actions spanned by one step of each abstract level."""
        return [level.stride for level in self.abstract]

    def lift_all(self, h: Tensor) -> list[Tensor]:
        """Pooled level-1 latent -> its state at every abstract level."""
        out, current = [], h
        for level in self.abstract:
            current = level.abstract(current)
            out.append(current)
        return out

    # -- EMA ---------------------------------------------------------------
    @torch.no_grad()
    def update_targets(self) -> None:
        m = self.config.ema_momentum
        for tgt, src in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tgt.mul_(m).add_(src.detach(), alpha=1.0 - m)
        for tgt, src in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            tgt.copy_(src)

    # -- inference ---------------------------------------------------------
    def encode(self, window: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(window, scalars)

    @torch.no_grad()
    def imagine_step(self, z: Tensor, action: Tensor) -> dict[str, Tensor]:
        """One imagined step. Returns mean latent, disagreement, and outcomes."""
        z_next, disagreement = self.ensemble.mean_and_disagreement(z, action)
        h_next = self.encoder.pool(z_next)
        outcomes = self.outcome_heads(h_next)
        tile_logits = self.tile_head(z_next)
        return {
            "z": z_next,
            "h": h_next,
            "disagreement": disagreement,
            "tile_logits": tile_logits,
            **outcomes,
        }

    @torch.no_grad()
    def render_imagined(self, tile_logits: Tensor) -> Tensor:
        """(B, VOCAB, S, S) -> (B, S, S) predicted glyph ids."""
        return tile_logits.argmax(dim=1)

    # -- training ----------------------------------------------------------
    def loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """One-step training loss.

        Expected keys:
          window      (B, S, S)   int64   current egocentric window
          scalars     (B, D)      float   current public scalars
          action      (B,)        int64
          next_window (B, S, S)   int64
          next_scalars(B, D)      float
          gem_delta   (B,)        int64   clamped 0..MAX_GEM_DELTA-1
          room_change (B,)        float   0/1
          player_vis  (B,)        float   0/1
          death       (B,)        float   0/1
          moved       (B,)        float   0/1, self-derived (window changed?)
          member_mask (K, B)      float   bootstrap mask, optional
        """
        c = self.config
        z, h = self.encoder(batch["window"], batch["scalars"])

        with torch.no_grad():
            z_target, _ = self.target_encoder(batch["next_window"], batch["next_scalars"])

        predictions = self.ensemble(z, batch["action"])            # (K, B, C, S, S)
        k, b = predictions.shape[0], predictions.shape[1]
        mask = batch.get("member_mask")
        if mask is None:
            mask = predictions.new_ones((k, b))

        # JEPA latent term, per member, bootstrap-masked
        latent_err = (predictions - z_target[None]).pow(2).mean(dim=(2, 3, 4))   # (K, B)
        latent_loss = (latent_err * mask).sum() / mask.sum().clamp(min=1.0)

        # tile reconstruction from the ensemble mean
        z_mean = predictions.mean(dim=0)
        tile_logits = self.tile_head(z_mean)
        tile_loss = F.cross_entropy(tile_logits, batch["next_window"])

        # grounded outcome heads from the pooled predicted latent
        h_next = self.encoder.pool(z_mean)
        outcomes = self.outcome_heads(h_next)
        gem_loss = F.cross_entropy(outcomes["gem_delta_logits"], batch["gem_delta"])
        room_loss = F.binary_cross_entropy_with_logits(
            outcomes["room_change_logit"], batch["room_change"]
        )
        aux_loss = (
            F.binary_cross_entropy_with_logits(outcomes["player_visible_logit"], batch["player_vis"])
            + F.binary_cross_entropy_with_logits(outcomes["death_logit"], batch["death"])
            + F.binary_cross_entropy_with_logits(outcomes["moved_logit"], batch["moved"])
        )

        var_loss, cov_loss = vicreg_terms(h)

        total = (
            c.w_latent * latent_loss
            + c.w_tile * tile_loss
            + c.w_gem * gem_loss
            + c.w_room * room_loss
            + c.w_aux * aux_loss
            + c.w_vicreg_var * var_loss
            + c.w_vicreg_cov * cov_loss
        )

        with torch.no_grad():
            tile_acc = (tile_logits.argmax(1) == batch["next_window"]).float().mean()
            changed = batch["window"] != batch["next_window"]
            changed_acc = (
                (tile_logits.argmax(1) == batch["next_window"])[changed].float().mean()
                if changed.any() else torch.zeros((), device=total.device)
            )
            gem_pred = outcomes["gem_delta_logits"].argmax(1)
            gem_acc = (gem_pred == batch["gem_delta"]).float().mean()
            # Accuracy is near-useless at a ~2% gem base rate -- "always 0"
            # scores 98%. Recall on the positive class is the honest number.
            positive = batch["gem_delta"] > 0
            gem_recall = (
                (gem_pred[positive] > 0).float().mean()
                if positive.any() else torch.zeros((), device=total.device)
            )
            predicted_positive = gem_pred > 0
            gem_precision = (
                (batch["gem_delta"][predicted_positive] > 0).float().mean()
                if predicted_positive.any() else torch.zeros((), device=total.device)
            )

        stats = {
            "loss": float(total),
            "latent": float(latent_loss),
            "tile": float(tile_loss),
            "tile_acc": float(tile_acc),
            "changed_tile_acc": float(changed_acc),
            "gem": float(gem_loss),
            "gem_acc": float(gem_acc),
            "gem_recall": float(gem_recall),
            "gem_precision": float(gem_precision),
            "gem_pos_rate": float(positive.float().mean()),
            "room": float(room_loss),
            "aux": float(aux_loss),
            "vicreg_var": float(var_loss),
            "vicreg_cov": float(cov_loss),
        }
        return total, stats

    def hierarchy_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """JEPA loss for every abstract level, each at its own timescale.

        Level ``k`` predicts where it lands after ``stride_k`` primitive
        actions, matched against the same lift applied to the *true* future
        latent. Targets are computed under ``no_grad`` from the EMA target
        encoder, so no level can cheat by moving its own target.

        Expected keys, for each level index ``k``:
            window, scalars                    state at t
            chunk_actions_{k}  (B, stride_k)   the primitive actions spanned
            future_window_{k}, future_scalars_{k}   state at t + stride_k

        Each level is scored separately so the report shows whether a third or
        fourth level is actually learning anything, or just sitting there
        adding parameters. A level whose prediction loss never falls below its
        VICReg variance term is not earning its place.
        """
        c = self.config
        with torch.no_grad():
            _, h_now = self.target_encoder(batch["window"], batch["scalars"])

        total = h_now.new_zeros(())
        stats: dict[str, float] = {}
        state_now = h_now

        for k, level in enumerate(self.abstract):
            future_window = batch.get(f"future_window_{k}")
            if future_window is None:
                break                       # this level has no samples this batch
            with torch.no_grad():
                _, h_future = self.target_encoder(future_window, batch[f"future_scalars_{k}"])
                target = h_future
                for below in self.abstract[: k + 1]:
                    target = below.abstract(target)

            state_at_level = level.abstract(state_now)
            predicted = level(state_at_level, batch[f"chunk_actions_{k}"])

            pred_loss = F.mse_loss(predicted, target)
            var_loss, cov_loss = vicreg_terms(predicted)
            level_loss = (
                c.w_level2 * pred_loss
                + c.w_vicreg_var * var_loss
                + c.w_vicreg_cov * cov_loss
            )
            total = total + level_loss

            name = f"l{k + 2}"              # level 2, 3, 4 in the report
            stats[f"{name}_pred"] = float(pred_loss)
            stats[f"{name}_var"] = float(var_loss)
            stats[f"{name}_cov"] = float(cov_loss)
            stats[f"{name}_stride"] = float(level.stride)

            state_now = state_at_level      # feed this level's state upward

        stats["l_total"] = float(total)
        return total, stats

    # Kept so older callers and scripts keep working.
    level2_loss = hierarchy_loss
