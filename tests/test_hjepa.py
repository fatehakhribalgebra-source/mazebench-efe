"""End-to-end check of the H-JEPA pipeline on a toy world.

Not a unit test. It builds a small deterministic grid world, encodes each
position as a fixed latent, trains the predictor on random walks, then asks
``GoalCEM`` to reach a goal latent. If the planner cannot navigate a world the
model can predict perfectly, the bug is in the planner, and we want to find
that here rather than on the GPU machine after a day of encoding frames.

    python tests/test_hjepa.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.goal_cem import GoalCEM, GoalCEMConfig           # noqa: E402
from model.hjepa import HJEPA, HJEPAConfig, goal_cost        # noqa: E402
from train.frames import LatentEpisodes, WindowSpec, split_episodes  # noqa: E402

GRID = 8
N_PATCHES = 16
PATCH_DIM = 32
DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # up, down, left, right


def toy_config(levels: int = 2, rollout: int = 3) -> HJEPAConfig:
    return HJEPAConfig(
        n_patches=N_PATCHES, patch_dim=PATCH_DIM, depth=2, heads=2,
        n_levels=levels, strides=(1, 4, 16), context=2, rollout_steps=rollout,
    )


def make_codebook(seed: int = 0) -> np.ndarray:
    """One fixed latent per grid cell. Smooth in space so the task is learnable."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(2, N_PATCHES, PATCH_DIM)).astype(np.float32)
    book = np.zeros((GRID, GRID, N_PATCHES, PATCH_DIM), dtype=np.float32)
    for x in range(GRID):
        for y in range(GRID):
            book[x, y] = (x / (GRID - 1)) * basis[0] + (y / (GRID - 1)) * basis[1]
    return book


def step(x: int, y: int, action: int) -> tuple[int, int]:
    dx, dy = DELTAS[action]
    return min(max(x + dx, 0), GRID - 1), min(max(y + dy, 0), GRID - 1)


def make_episodes(root: Path, book: np.ndarray, n: int = 12, length: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    for ep in range(n):
        x, y = int(rng.integers(GRID)), int(rng.integers(GRID))
        actions = rng.integers(0, 4, size=length - 1).astype(np.int8)
        latents = np.zeros((length, N_PATCHES, PATCH_DIM), dtype=np.float16)
        latents[0] = book[x, y]
        for t, a in enumerate(actions):
            x, y = step(x, y, int(a))
            latents[t + 1] = book[x, y]
        np.savez_compressed(root / f"ep{ep:03d}.npz", latents=latents, actions=actions)


def train_level(model, root: Path, level: int, stride: int, steps: int = 400) -> dict:
    spec = WindowSpec(stride=stride, context=model.config.context,
                      rollout=model.config.rollout_steps)
    train_eps, val_eps = split_episodes(root, val_frac=0.25, seed=0)
    train_set = LatentEpisodes(root, spec, train_eps)
    val_set = LatentEpisodes(root, spec, val_eps)
    loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.levels[level].parameters(), lr=1e-3)

    seen = 0
    while seen < steps:
        for batch in loader:
            out = model.loss(batch["context"], batch["actions"], batch["targets"], level=level)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            seen += 1
            if seen >= steps:
                break

    model.eval()
    losses, identities = [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(val_set, batch_size=32):
            out = model.loss(batch["context"], batch["actions"], batch["targets"], level=level)
            last = batch["context"][:, -1:].expand_as(batch["targets"])
            losses.append(out["loss"].item())
            identities.append(torch.nn.functional.mse_loss(last, batch["targets"]).item())
    model.train()
    return {"val": float(np.mean(losses)), "identity": float(np.mean(identities))}


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    failures = []
    book = make_codebook()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_episodes(root, book)

        config = toy_config(levels=2)
        model = HJEPA(config)

        print("shapes")
        ctx = torch.randn(4, config.context, N_PATCHES, PATCH_DIM)
        acts = torch.randint(0, 4, (4, 1))
        got = model.predict(ctx, acts, level=0).shape
        want = (4, N_PATCHES, PATCH_DIM)
        print(f"  predict -> {tuple(got)} (want {want})")
        if tuple(got) != want:
            failures.append("predict returned the wrong shape")

        with torch.no_grad():
            identity_out = model.predict(ctx, acts, level=0)
        drift = (identity_out - ctx[:, -1]).abs().max().item()
        print(f"  identity init: max drift {drift:.2e} (want 0)")
        if drift > 1e-6:
            failures.append("predictor is not initialised to the identity")

        print("\ntraining level 1 (stride 1)")
        stats = train_level(model, root, level=0, stride=1)
        ratio = stats["val"] / max(stats["identity"], 1e-9)
        print(f"  val {stats['val']:.4f}  identity {stats['identity']:.4f}  ratio {ratio:.3f}")
        if ratio >= 0.5:
            failures.append(f"level 1 barely beats identity (ratio {ratio:.3f})")

        print("\ntraining level 2 (stride 4)")
        stats2 = train_level(model, root, level=1, stride=4)
        ratio2 = stats2["val"] / max(stats2["identity"], 1e-9)
        print(f"  val {stats2['val']:.4f}  identity {stats2['identity']:.4f}  ratio {ratio2:.3f}")
        if ratio2 >= 1.0:
            failures.append(f"level 2 does not beat identity (ratio {ratio2:.3f})")

        print("\ngoal reaching with CEM")
        model.eval()
        planner = GoalCEM(
            model,
            GoalCEMConfig(horizon=8, n_samples=128, n_iters=2, coarse_keep=32,
                          mid_keep=16, seed=0),
        )
        reached = 0
        trials = 8
        for trial in range(trials):
            rng = np.random.default_rng(100 + trial)
            sx, sy = int(rng.integers(GRID)), int(rng.integers(GRID))
            gx, gy = sx, sy
            for _ in range(4):                       # goal a few steps away
                gx, gy = step(gx, gy, int(rng.integers(4)))
            if (gx, gy) == (sx, sy):
                trials -= 1
                continue

            goal = torch.from_numpy(book[gx, gy])
            x, y = sx, sy
            planner.reset()
            for _ in range(12):
                ctx = torch.from_numpy(
                    np.stack([book[x, y]] * config.context)
                ).float()
                action = planner.plan(ctx, goal)["action"]
                x, y = step(x, y, action)
                if (x, y) == (gx, gy):
                    reached += 1
                    break
        rate = reached / max(trials, 1)
        print(f"  reached {reached}/{trials} = {rate:.0%}")
        if rate < 0.6:
            failures.append(f"CEM reached only {rate:.0%} of toy goals")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
