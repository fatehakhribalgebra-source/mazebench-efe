"""Transfer learning: frozen pretrained vision backbones as the encoder.

The default tile encoder in ``world_model.GridEncoder`` is the right choice for
a tile world — direct glyph access, trains in minutes, nothing a frozen image
model can add to it. So why this file?

Because "can it still work from pixels" is a different and harder question than
"can it work". MazeBench has a ``vision`` observation mode that returns a
perspective PNG instead of an ASCII board, and the win condition is identical.
An agent that works in both is doing something more than reading a grid. This
module is the pixel path, and it keeps the output contract of ``GridEncoder``
exactly — ``(z, h)`` with ``z`` of shape ``(B, C, S, S)`` — so the predictor,
tile head, planner and EFE code are all unchanged.

Both backbones stay **frozen**. Only a small adapter trains: patch tokens are
reshaped to their native spatial grid, bilinearly resampled to ``S x S``, and
projected to ``C`` channels. That is the whole transfer surface. Freezing is not
timidity — with a few thousand transitions, fine-tuning a ViT-L would memorise
the buffer, and the earlier run of this project already found frozen-backbone
prediction heads to be the only arm that trained stably at this data scale.

Revisions are pinned to the two checkpoints the earlier package in this project
actually validated on an RTX 2050, so results stay comparable:

* ``facebook/vjepa2-vitl-fpc64-256`` @ ``b3c1679b7c34d3255ef3547f27c7b226aefab26f``
  V-JEPA 2's frozen video encoder over a short frame stack. Note this is a
  short-context adaptation of a 64-frame pretraining configuration, and it is
  *not* V-JEPA 2-AC — no robot action conditioning is involved.
* ``google/siglip2-base-patch16-224`` @ ``75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2``
  A frozen image encoder. Spatial features only, no temporal or action
  dynamics. A Google visual backbone — not Genie, not SIMA.

Needs ``transformers`` and ``pillow``, and MazeBench's vision observation mode
(which needs Playwright). Not wired into the default pipeline; opt in with
``--encoder vision`` once those are present.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PINNED = {
    "vjepa2": ("facebook/vjepa2-vitl-fpc64-256", "b3c1679b7c34d3255ef3547f27c7b226aefab26f"),
    "siglip2": ("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
}


@dataclass
class VisionConfig:
    backbone: str = "siglip2"          # "siglip2" | "vjepa2"
    channels: int = 64                 # must match WorldModelConfig.channels
    window_size: int = 15              # must match WorldModelConfig.window_size
    pooled_dim: int = 256
    n_frames: int = 4                  # vjepa2 only: frames in the stack
    offline: bool = False              # require local cache, never hit the network
    dtype: str = "float32"


def _load_backbone(config: VisionConfig):
    """Load a frozen backbone at its pinned revision."""
    try:
        from transformers import AutoImageProcessor, AutoModel, AutoVideoProcessor
    except ImportError as error:                       # pragma: no cover
        raise ImportError(
            "The vision encoder needs `transformers` and `pillow`:\n"
            "    pip install 'transformers>=4.46' pillow\n"
            "The default tile encoder needs neither."
        ) from error

    if config.backbone not in PINNED:
        raise ValueError(f"backbone must be one of {sorted(PINNED)}, got {config.backbone!r}")
    name, revision = PINNED[config.backbone]
    kwargs = {"revision": revision, "local_files_only": config.offline}

    model = AutoModel.from_pretrained(name, **kwargs)
    if config.backbone == "vjepa2":
        try:
            processor = AutoVideoProcessor.from_pretrained(name, **kwargs)
        except Exception:                               # older transformers
            processor = AutoImageProcessor.from_pretrained(name, **kwargs)
    else:
        processor = AutoImageProcessor.from_pretrained(name, **kwargs)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor, name, revision


def _patch_grid(tokens: Tensor, drop_prefix: int = 0, side: int | None = None) -> Tensor:
    """(B, N, D) patch tokens -> (B, D, G, G).

    ``side`` is the spatial grid width, which the caller derives from the
    backbone config (image size / patch size): 224/16 = 14 for SigLIP 2,
    256/16 = 16 per frame for V-JEPA 2.

    It must be passed for any video backbone. A video model returns
    ``N = T' * G * G`` tokens, and that factorises more than one way -- V-JEPA
    2's 2048 tokens are ``8 x 16 x 16``, but a square-root heuristic happily
    reads them as ``2 x 32 x 32`` and silently builds a 32x32 grid of
    half-temporal features. Guessing here is a bug, so guessing is only allowed
    when ``N`` is itself a perfect square and no side was given.

    For a video backbone the temporal axis is averaged out: the world model
    carries its own dynamics and does not want the backbone's.
    """
    if drop_prefix:
        tokens = tokens[:, drop_prefix:, :]
    b, n, d = tokens.shape

    if side is None:
        root = int(round(n ** 0.5))
        if root * root != n:
            raise ValueError(
                f"{n} patch tokens is not a perfect square, so the spatial grid is "
                "ambiguous. Pass `side` derived from the backbone's image/patch size."
            )
        side = root

    cells = side * side
    if n % cells:
        raise ValueError(f"{n} tokens is not divisible by a {side}x{side} grid")
    frames = n // cells
    if frames > 1:
        tokens = tokens.view(b, frames, cells, d).mean(dim=1)
    return tokens.reshape(b, side, side, -1).permute(0, 3, 1, 2)


class FrozenVisionEncoder(nn.Module):
    """Drop-in replacement for ``GridEncoder`` that reads pixels.

    Returns the same ``(z, h)`` pair with the same shapes, so everything
    downstream — predictor ensemble, tile head, planner, EFE — is untouched.

    The tile head still decodes ``z`` to glyph logits. That is not a
    contradiction: in vision mode there is no ASCII board to supervise it, so
    the tile head is simply switched off (``w_tile=0``) and the latent is
    trained by the JEPA term and the grounded outcome heads alone. Imagination
    is then no longer renderable as ASCII, which is a real loss of
    interpretability and the main practical cost of the pixel path.
    """

    def __init__(self, config: VisionConfig | None = None):
        super().__init__()
        self.config = config or VisionConfig()
        self.backbone, self.processor, self.name, self.revision = _load_backbone(self.config)
        self.pooled_dim = self.config.pooled_dim

        width = self._feature_width()
        self.patch_side = self._patch_side()
        self.adapter = nn.Sequential(
            nn.Conv2d(width, self.config.channels, 1),
            nn.GroupNorm(8, self.config.channels),
            nn.GELU(),
            nn.Conv2d(self.config.channels, self.config.channels, 3, padding=1),
        )
        self.out_norm = nn.GroupNorm(8, self.config.channels)
        self.pool_proj = nn.Sequential(
            nn.Linear(self.config.channels * 2, self.pooled_dim),
            nn.LayerNorm(self.pooled_dim),
        )

    def _feature_width(self) -> int:
        cfg = getattr(self.backbone, "config", None)
        for attr in ("hidden_size", "vision_hidden_size"):
            if cfg is not None and getattr(cfg, attr, None):
                return int(getattr(cfg, attr))
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None and getattr(vision_cfg, "hidden_size", None):
            return int(vision_cfg.hidden_size)
        raise RuntimeError(f"cannot determine feature width for {self.name}")

    def _patch_side(self) -> int:
        """Spatial grid width = image size / patch size, read from the config.

        Never inferred from the token count -- see ``_patch_grid``.
        """
        cfg = getattr(self.backbone, "config", None)
        candidates = [cfg, getattr(cfg, "vision_config", None)]
        for source in candidates:
            if source is None:
                continue
            image = getattr(source, "image_size", None) or getattr(source, "crop_size", None)
            patch = getattr(source, "patch_size", None)
            if isinstance(image, (list, tuple)):
                image = image[-1]
            if isinstance(image, dict):
                image = image.get("height") or image.get("shortest_edge")
            if isinstance(patch, (list, tuple)):
                patch = patch[-1]
            if image and patch:
                return int(image) // int(patch)
        raise RuntimeError(
            f"cannot determine the patch grid size for {self.name}; "
            "set it explicitly rather than letting _patch_grid guess"
        )

    # ------------------------------------------------------------- features
    @torch.no_grad()
    def backbone_features(self, pixel_inputs: dict[str, Tensor]) -> Tensor:
        """Frozen patch tokens -> (B, D, G, G). No gradients ever flow here."""
        model = getattr(self.backbone, "vision_model", self.backbone)
        output = model(**pixel_inputs)
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None:
            tokens = output[0]
        cells = self.patch_side * self.patch_side
        drop = 1 if tokens.shape[1] % cells == 1 else 0   # drop a CLS token if present
        return _patch_grid(tokens, drop_prefix=drop, side=self.patch_side)

    def preprocess(self, frames) -> dict[str, Tensor]:
        """PIL images (or a list of frame lists for vjepa2) -> model inputs."""
        if self.config.backbone == "vjepa2":
            batch = frames if isinstance(frames[0], (list, tuple)) else [frames]
            return self.processor(batch, return_tensors="pt")
        return self.processor(images=frames, return_tensors="pt")

    # ---------------------------------------------------------------- forward
    def forward(self, pixel_inputs: dict[str, Tensor], scalars: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone_features(pixel_inputs).to(torch.float32)
        resized = F.interpolate(
            features, size=(self.config.window_size, self.config.window_size),
            mode="bilinear", align_corners=False,
        )
        z = self.out_norm(self.adapter(resized))
        return z, self.pool(z)

    def pool(self, z: Tensor) -> Tensor:
        return self.pool_proj(torch.cat([z.mean(dim=(2, 3)), z.amax(dim=(2, 3))], dim=1))

    def trainable_parameters(self):
        yield from self.adapter.parameters()
        yield from self.out_norm.parameters()
        yield from self.pool_proj.parameters()

    def describe(self) -> dict[str, str]:
        n_frozen = sum(p.numel() for p in self.backbone.parameters())
        n_train = sum(p.numel() for p in self.trainable_parameters())
        return {
            "backbone": self.name,
            "revision": self.revision,
            "frozen_params": f"{n_frozen / 1e6:.1f}M",
            "adapter_params": f"{n_train / 1e6:.2f}M",
        }


class HybridEncoder(nn.Module):
    """Tile features plus frozen pretrained visual features.

    The honest way to "benefit from transfer learning" on a tile world: keep the
    tile access that actually works, and add pretrained visual features
    alongside it, so the comparison against the tile-only encoder isolates what
    the backbone contributes. Requires both observation modes for the same
    rollout, which costs a Playwright render per step.
    """

    def __init__(self, tile_encoder, vision_encoder: FrozenVisionEncoder, channels: int = 64):
        super().__init__()
        self.tile_encoder = tile_encoder
        self.vision_encoder = vision_encoder
        self.pooled_dim = tile_encoder.pooled_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(channels * 2, self.pooled_dim), nn.LayerNorm(self.pooled_dim)
        )

    def forward(self, window: Tensor, scalars: Tensor, pixel_inputs: dict[str, Tensor]):
        z_tile, _ = self.tile_encoder(window, scalars)
        z_vision, _ = self.vision_encoder(pixel_inputs, scalars)
        z = self.fuse(torch.cat([z_tile, z_vision], dim=1))
        return z, self.pool(z)

    def pool(self, z: Tensor) -> Tensor:
        return self.pool_proj(torch.cat([z.mean(dim=(2, 3)), z.amax(dim=(2, 3))], dim=1))
