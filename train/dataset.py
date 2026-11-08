"""Datasets over collected episodes.

Two views of the same buffer:

``TransitionDataset`` one-step transitions for level 1.
``ChunkDataset``      (t, t + chunk_len) pairs for level 2.

Both only emit windows whose whole span is ``valid``, so no sample ever crosses
a control action.

Rare-event oversampling is not optional here. Gems and room changes are a
percent or two of transitions; trained on the raw distribution the gem head
learns to predict "no gem" forever and the EFE pragmatic term goes flat. The
sampler reweights those transitions up and the loss weights ``w_gem`` higher
still. The honest consequence is that the head's predicted probabilities are
not calibrated to the true base rate -- it is used for *ranking* candidate
plans, where that does not matter, and the README says so.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from train.collect import Episode


class TransitionDataset(Dataset):
    def __init__(self, episodes: list[Episode]):
        self.episodes = episodes
        self.index: list[tuple[int, int]] = []
        for e, ep in enumerate(episodes):
            for t in range(len(ep)):
                if ep.valid[t]:
                    self.index.append((e, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, t = self.index[i]
        ep = self.episodes[e]
        return {
            "window": torch.from_numpy(ep.windows[t].astype(np.int64)),
            "scalars": torch.from_numpy(ep.scalars[t]),
            "action": torch.tensor(int(ep.actions[t]), dtype=torch.long),
            "next_window": torch.from_numpy(ep.windows[t + 1].astype(np.int64)),
            "next_scalars": torch.from_numpy(ep.scalars[t + 1]),
            "gem_delta": torch.tensor(int(ep.gem_delta[t]), dtype=torch.long),
            "room_change": torch.tensor(float(ep.room_change[t])),
            "player_vis": torch.tensor(float(ep.player_vis[t])),
            "death": torch.tensor(float(ep.death[t])),
            "moved": torch.tensor(float(ep.moved[t])),
        }

    # -- sampling -----------------------------------------------------------
    def event_weights(self, gem_boost: float = 40.0, room_boost: float = 10.0,
                      push_boost: float = 1.0) -> np.ndarray:
        """Upweight the transitions that actually carry a learning signal."""
        weights = np.ones(len(self.index), dtype=np.float64)
        for i, (e, t) in enumerate(self.index):
            ep = self.episodes[e]
            if ep.gem_delta[t] > 0:
                weights[i] *= gem_boost
            if ep.room_change[t]:
                weights[i] *= room_boost
            # a changed board with no gem/room event is usually a box push
            elif not np.array_equal(ep.windows[t], ep.windows[t + 1]):
                weights[i] *= push_boost
        return weights

    def sampler(self, num_samples: int | None = None, **kwargs) -> WeightedRandomSampler:
        weights = self.event_weights(**kwargs)
        return WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples or len(self),
            replacement=True,
        )

    def label_stats(self) -> dict[str, float]:
        gem = np.array([self.episodes[e].gem_delta[t] for e, t in self.index])
        room = np.array([self.episodes[e].room_change[t] for e, t in self.index])
        moved = np.array([self.episodes[e].moved[t] for e, t in self.index])
        death = np.array([self.episodes[e].death[t] for e, t in self.index])
        return {
            "transitions": len(self.index),
            "gem_rate": float((gem > 0).mean()),
            "room_change_rate": float(room.mean()),
            "moved_rate": float(moved.mean()),
            "death_rate": float(death.mean()),
        }


class ChunkDataset(Dataset):
    """Multi-timescale samples, one future per abstract level.

    Level ``k`` spans ``strides[k]`` primitive actions (4, 16, 64 for a
    2/3/4-level model), so each sample carries a future window per level plus
    the primitive actions in between.

    The longest stride sets the index: a sample only exists where the *whole*
    span of every level is valid, so no sample crosses a control action. That
    makes deeper hierarchies strictly data-hungrier -- a 4-level model needs 64
    consecutive clean transitions per sample, and an episode of 120 actions
    yields far fewer of those than it does 4-step ones. Worth remembering when
    reading a 2-vs-3-vs-4 comparison: the deeper arms see fewer samples from
    the same buffer, and ``level_counts`` reports exactly how many.
    """

    def __init__(
        self,
        episodes: list[Episode],
        chunk_len: int = 4,
        n_levels: int = 2,
        index_levels: int | None = None,
    ):
        """``index_levels`` fixes *which* samples exist, independently of depth.

        This matters for the 2-vs-3-vs-4 comparison. A sample only exists where
        the longest level's whole span is valid, so indexing at the model's own
        depth gives a 2-level model 6578 samples and a 4-level model 2689 from
        the same buffer. The deeper arm would then look worse for being starved
        rather than for being deep -- in a smoke run its *level 2* came out
        worse than the shallow model's, which is the confound showing up.

        Setting ``index_levels`` to the deepest arm you intend to compare makes
        every arm train on an identical sample set, so the only difference left
        is depth.
        """
        self.episodes = episodes
        self.chunk_len = chunk_len
        self.n_levels = n_levels
        self.strides = [chunk_len ** (k + 1) for k in range(max(0, n_levels - 1))]
        index_depth = max(index_levels or n_levels, n_levels)
        index_strides = [chunk_len ** (k + 1) for k in range(max(0, index_depth - 1))]
        self.index_levels = index_depth
        self.longest = max(index_strides) if index_strides else chunk_len
        self.index: list[tuple[int, int]] = []
        for e, ep in enumerate(episodes):
            for t in range(len(ep) - self.longest):
                if ep.valid[t : t + self.longest].all():
                    self.index.append((e, t))

    def __len__(self) -> int:
        return len(self.index)

    def level_counts(self) -> dict[str, int]:
        """Samples available per level, if each were indexed on its own."""
        out: dict[str, int] = {}
        for k, stride in enumerate(self.strides):
            total = 0
            for ep in self.episodes:
                for t in range(len(ep) - stride):
                    if ep.valid[t : t + stride].all():
                        total += 1
            out[f"level{k + 2}_stride{stride}"] = total
        out["joint"] = len(self.index)
        return out

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, t = self.index[i]
        ep = self.episodes[e]
        sample = {
            "window": torch.from_numpy(ep.windows[t].astype(np.int64)),
            "scalars": torch.from_numpy(ep.scalars[t]),
        }
        for k, stride in enumerate(self.strides):
            sample[f"chunk_actions_{k}"] = torch.from_numpy(
                ep.actions[t : t + stride].astype(np.int64)
            )
            sample[f"future_window_{k}"] = torch.from_numpy(
                ep.windows[t + stride].astype(np.int64)
            )
            sample[f"future_scalars_{k}"] = torch.from_numpy(ep.scalars[t + stride])
        return sample


def bootstrap_mask(n_members: int, batch_size: int, device, p: float = 0.8) -> torch.Tensor:
    """Per-member Bernoulli mask over the batch.

    Each ensemble member sees a different ~80% of every batch. Without this the
    members converge to the same function and the disagreement signal -- the
    whole epistemic term -- goes to zero.
    """
    return (torch.rand(n_members, batch_size, device=device) < p).float()
