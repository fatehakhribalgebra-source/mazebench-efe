"""Hierarchical JEPA over MazeBench's 3D frames.

The goal this file exists to serve: reproduce the published goal-reaching
numbers (DINO-WM 81.6% on PointMaze, JEPA-WM 83.9%) on MazeBench, planning to a
goal *screenshot* with CEM. Nothing here knows about gems, rooms, expected free
energy or exploration. That comes later and only if this works first.

Recipe follows ``facebookresearch/jepa-wms`` (arXiv 2512.24497), navigation
configuration: a **frozen** DINOv2 ViT-S/14 at 224, and a small
action-conditioned predictor on top. The encoder is 1.7B-images worth of prior
that we never touch; the predictor is the only thing that learns MazeBench.

Three design choices worth knowing, because they are not the obvious ones:

1. **Patch tokens, not a pooled vector.** The encoder returns ``(B, 256, 384)``
   and every level predicts in that space. Pooling to 256 numbers throws away
   where things are, and where things are is the entire task.

2. **The hierarchy is temporal, not representational.** Level *k* predicts
   ``stride[k]`` steps ahead in the *same* latent space, given the actions that
   span the gap. So levels do not need their own anti-collapse machinery, a
   goal distance means the same thing at every level, and a coarse plan can be
   refined into a fine one without translation. The earlier version of this
   project gave each level its own abstract space and the top level quietly
   collapsed; this avoids that failure mode by construction.

3. **Multistep rollout loss, not teacher forcing alone.** The jepa-wms ablation
   found this to be one of the main drivers of planning success: a model that
   only ever predicts one step from ground truth drifts badly when the planner
   asks it for eight. ``rollout_steps`` controls it.

Action space is **four moves**. Camera rotation, undo, reset and level jumps are
deliberately excluded: rotation does not change world state (so a goal image
pins the camera, and letting the planner rotate makes the goal unreachable in
latent space for no benefit), and undo/reset are history-dependent, which a
Markov latent model cannot represent honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Four moves. See the module docstring for why rotation and reset are out.
ACTIONS = ("up", "down", "left", "right")
N_ACTIONS = len(ACTIONS)

# Not gated, unlike DINOv3, so the other machine needs no licence click.
# ViT-S/14 at 224 gives a 16x16 patch grid and 384 channels.
DINOV2_SMALL = "facebook/dinov2-small"
DINOV2_REVISION = "ed5e3cfc84a1e30e95a0f8d9ce4e4f0b2a4b0e66"


@dataclass
class HJEPAConfig:
    # encoder
    backbone: str = DINOV2_SMALL
    revision: str | None = None       # None = whatever is cached / latest
    image_size: int = 224
    patch_dim: int = 384              # ViT-S hidden size
    n_patches: int = 256              # (224 / 14) ** 2
    offline: bool = False             # require a local cache, never hit network

    # predictor, per jepa-wms navigation config
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    action_dim: int = 128

    # hierarchy
    n_levels: int = 3                 # 1 = flat DINO-WM-style baseline
    strides: tuple[int, ...] = (1, 4, 16)

    # training
    context: int = 2                  # W_p = 2 in the paper
    rollout_steps: int = 4            # multistep rollout loss
    dropout: float = 0.0

    def level_strides(self) -> tuple[int, ...]:
        if not 1 <= self.n_levels <= len(self.strides):
            raise ValueError(f"n_levels must be 1..{len(self.strides)}, got {self.n_levels}")
        return self.strides[: self.n_levels]


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------


class FrozenEncoder(nn.Module):
    """Frozen DINOv2. Returns patch tokens, CLS dropped.

    Kept as a module (rather than precomputed features) so evaluation can encode
    a goal screenshot on the fly. For training, precompute once with
    ``train/encode_frames.py`` -- running a ViT over the same frames every epoch
    is the single easiest way to waste a GPU.
    """

    def __init__(self, config: HJEPAConfig):
        super().__init__()
        self.config = config
        try:
            from transformers import AutoModel
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "The vision encoder needs transformers:\n"
                "    pip install 'transformers>=4.46' pillow"
            ) from error

        kwargs: dict = {"local_files_only": config.offline}
        if config.revision:
            kwargs["revision"] = config.revision
        self.backbone = AutoModel.from_pretrained(config.backbone, **kwargs)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """``(B, 3, H, W)`` normalised images -> ``(B, N, D)`` patch tokens."""
        out = self.backbone(pixel_values=images).last_hidden_state
        return out[:, 1:, :]  # drop CLS

    def train(self, mode: bool = True):  # noqa: D102 - stays frozen either way
        super().train(mode)
        self.backbone.eval()
        return self


# --------------------------------------------------------------------------
# predictor
# --------------------------------------------------------------------------


class AdaLNBlock(nn.Module):
    """Transformer block whose norms are modulated by the action embedding.

    AdaLN conditioning rather than concatenating the action as a token: the
    action then reaches every patch at every layer, which is what you want when
    one action moves the whole scene.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        # six modulation vectors: shift/scale/gate for attention and for MLP
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return x + gate2.unsqueeze(1) * self.mlp(h)


class LevelPredictor(nn.Module):
    """One level of the hierarchy: latents + the actions spanning a stride -> next latents.

    Returns ``z + delta`` with the output projection zero-initialised, so the
    model starts as the identity. In a world where one move changes a small part
    of the frame, "nothing changes" is a strong prior and this skips the epochs
    otherwise spent discovering it.
    """

    def __init__(self, config: HJEPAConfig, stride: int):
        super().__init__()
        self.stride = stride
        dim = config.patch_dim

        self.action_embed = nn.Embedding(N_ACTIONS, config.action_dim)
        # A level with stride s consumes s primitive actions; order matters, so
        # they are concatenated in sequence rather than summed.
        self.action_proj = nn.Sequential(
            nn.Linear(config.action_dim * stride, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.context_proj = nn.Linear(dim * config.context, dim)
        self.pos = nn.Parameter(torch.zeros(1, config.n_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList(
            [AdaLNBlock(dim, config.heads, config.mlp_ratio, config.dropout)
             for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context: Tensor, actions: Tensor) -> Tensor:
        """``context`` ``(B, W, N, D)``, ``actions`` ``(B, stride)`` -> ``(B, N, D)``."""
        batch, window, n_patches, dim = context.shape
        x = self.context_proj(context.permute(0, 2, 1, 3).reshape(batch, n_patches, window * dim))
        x = x + self.pos

        cond = self.action_proj(self.action_embed(actions).reshape(batch, -1))
        for block in self.blocks:
            x = block(x, cond)
        return context[:, -1] + self.head(self.norm(x))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class HJEPA(nn.Module):
    """``n_levels`` predictors over one shared patch-latent space."""

    def __init__(self, config: HJEPAConfig | None = None):
        super().__init__()
        self.config = config or HJEPAConfig()
        self.strides = self.config.level_strides()
        self.levels = nn.ModuleList(
            [LevelPredictor(self.config, stride) for stride in self.strides]
        )

    def predict(self, context: Tensor, actions: Tensor, level: int = 0) -> Tensor:
        """One step of ``level``, which advances ``strides[level]`` primitive steps."""
        return self.levels[level](context, actions)

    def rollout(self, context: Tensor, action_seq: Tensor, level: int = 0) -> Tensor:
        """Roll ``level`` forward over ``(B, T, stride)`` actions.

        Returns ``(B, T, N, D)``: the predicted latent after each level-step. The
        context window slides over predictions, so errors compound exactly as
        they will during planning -- which is the point of training on this.
        """
        window = context
        out = []
        for step in range(action_seq.shape[1]):
            nxt = self.predict(window, action_seq[:, step], level=level)
            out.append(nxt)
            window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)

    def loss(self, context: Tensor, action_seq: Tensor, targets: Tensor, level: int = 0) -> dict:
        """Multistep rollout loss in latent space.

        ``targets`` ``(B, T, N, D)`` are the encoder's own outputs for the
        frames the rollout should land on. Plain L2, as the paper uses for its
        CEM cost -- training and planning then agree on what "close" means.
        """
        pred = self.rollout(context, action_seq, level=level)
        per_step = F.mse_loss(pred, targets, reduction="none").mean(dim=(0, 2, 3))
        return {
            "loss": per_step.mean(),
            "per_step": per_step.detach(),
            "first": per_step[0].detach(),
            "last": per_step[-1].detach(),
        }


def goal_cost(latents: Tensor, goal: Tensor) -> Tensor:
    """L2 distance to a goal latent, averaged over patches.

    ``latents`` ``(..., N, D)``, ``goal`` ``(N, D)`` or ``(1, N, D)``.
    This is the planner's cost and the paper's cost. Keep them identical.
    """
    return (latents - goal).pow(2).mean(dim=(-1, -2))
