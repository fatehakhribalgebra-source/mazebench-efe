"""Frame episodes -> encoder latents -> training windows.

Two stages, deliberately separate.

**Encode once.** Running a ViT over the same frames every epoch wastes most of
the GPU time available. ``encode_episodes`` turns PNG frames into latents once
and writes them to disk; training then reads vectors and the encoder is never
loaded again. At 256 patches x 384 dims x float16 a frame costs 192 KB, so
100k frames is ~19 GB -- large but sequential, and it makes epochs cheap.

**Window on demand.** ``LatentEpisodes`` serves the windows a given level needs.
Level *k* has stride *s*: the context is ``W`` latents spaced *s* apart, each
rollout step consumes the *s* primitive actions spanning it, and the targets are
the latents the rollout should land on. One episode therefore has to be at least
``(W - 1) * s + T * s + 1`` frames long before the top level can use it at all --
with the defaults that is 81, so collect episodes of **at least 128 steps** or
level 3 silently gets no data.

On-disk format, one ``.npz`` per episode:

* ``latents``  ``(T, N, D)``  float16
* ``actions``  ``(T - 1,)``   int8, values 0..3 indexing ``model.hjepa.ACTIONS``
* ``level_id`` str, optional, for held-out-room splits
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(images: np.ndarray, size: int = 224) -> Tensor:
    """``(B, H, W, 3)`` uint8 -> normalised ``(B, 3, size, size)`` float tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(images)).float().div_(255.0)
    tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[-1] != size or tensor.shape[-2] != size:
        tensor = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def encode_episodes(
    encoder,
    frames: np.ndarray,
    actions: np.ndarray,
    out_path: str | Path,
    batch_size: int = 32,
    device: str = "cuda",
    image_size: int = 224,
) -> dict:
    """Encode one episode's frames and write ``out_path``."""
    if len(frames) != len(actions) + 1:
        raise ValueError(
            f"need one more frame than actions, got {len(frames)} frames "
            f"and {len(actions)} actions"
        )
    chunks = []
    for start in range(0, len(frames), batch_size):
        batch = preprocess(frames[start : start + batch_size], image_size).to(device)
        chunks.append(encoder(batch).half().cpu())
    latents = torch.cat(chunks).numpy()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, latents=latents, actions=np.asarray(actions, dtype=np.int8)
    )
    return {"frames": len(frames), "shape": tuple(latents.shape), "path": str(out_path)}


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------


@dataclass
class WindowSpec:
    stride: int
    context: int
    rollout: int

    def span(self) -> int:
        """Frames an episode must have for one window at this spec."""
        return (self.context - 1) * self.stride + self.rollout * self.stride + 1


class LatentEpisodes(Dataset):
    """Windows for one level of the hierarchy."""

    def __init__(self, root: str | Path, spec: WindowSpec, episodes: list[str] | None = None):
        self.root = Path(root)
        self.spec = spec
        paths = sorted(self.root.glob("*.npz")) if episodes is None else [
            self.root / name for name in episodes
        ]
        if not paths:
            raise FileNotFoundError(f"no .npz episodes under {self.root}")

        self.paths = paths
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.index: list[tuple[int, int]] = []
        skipped = 0
        for ep, path in enumerate(paths):
            with np.load(path) as data:
                length = len(data["latents"])
            first = (spec.context - 1) * spec.stride
            last = length - spec.rollout * spec.stride - 1
            if last < first:
                skipped += 1
                continue
            self.index += [(ep, t) for t in range(first, last + 1)]
        self.skipped_episodes = skipped

        if not self.index:
            raise ValueError(
                f"every episode is shorter than {spec.span()} frames, which is what "
                f"stride {spec.stride} needs. Collect longer episodes."
            )

    def __len__(self) -> int:
        return len(self.index)

    def _episode(self, ep: int) -> tuple[np.ndarray, np.ndarray]:
        if ep not in self._cache:
            with np.load(self.paths[ep]) as data:
                self._cache[ep] = (data["latents"], data["actions"])
            if len(self._cache) > 32:                       # bounded, FIFO-ish
                self._cache.pop(next(iter(self._cache)))
        return self._cache[ep]

    def __getitem__(self, i: int) -> dict:
        ep, t = self.index[i]
        latents, actions = self._episode(ep)
        stride, context, rollout = self.spec.stride, self.spec.context, self.spec.rollout

        ctx_idx = [t - (context - 1 - k) * stride for k in range(context)]
        tgt_idx = [t + (k + 1) * stride for k in range(rollout)]
        act = np.stack(
            [actions[t + k * stride : t + (k + 1) * stride] for k in range(rollout)]
        )
        return {
            "context": torch.from_numpy(latents[ctx_idx].astype(np.float32)),
            "actions": torch.from_numpy(act.astype(np.int64)),
            "targets": torch.from_numpy(latents[tgt_idx].astype(np.float32)),
        }


def split_episodes(root: str | Path, val_frac: float = 0.1, seed: int = 0):
    """Split by **episode**, never by window.

    Windows from one episode overlap heavily, so splitting them at random puts
    near-duplicate frames on both sides and the validation loss becomes a
    memorisation score. This project has already been burnt once by a metric
    that looked fine and meant nothing.
    """
    names = sorted(p.name for p in Path(root).glob("*.npz"))
    if not names:
        raise FileNotFoundError(f"no .npz episodes under {root}")
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_frac))
    return names[n_val:], names[:n_val]
