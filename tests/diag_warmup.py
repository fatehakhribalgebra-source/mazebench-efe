"""Is the renderer's nondeterminism a one-off warm-up, or ongoing?

``diag_latent_jitter.py`` found jitter at 16.8% of one action's latent change,
unevenly: one room was byte-identical across a full replay while another jittered
over a 340x356 region. That pattern fits **asset warm-up** rather than random
nondeterminism -- ``applyRenderStateSnapshot`` awaits
``preloadImagesForLevelState``, but a room's textures still have to arrive once,
and the first pass over a room would then render partly-unloaded geometry while
every later pass renders from a warm cache.

If that is the cause, pass 2 and pass 3 agree even though pass 1 differs from
both, and the fix is one throwaway render per room. If instead every pass
disagrees with every other, the renderer is genuinely nondeterministic and no
warm-up saves it.

Also reports the blocked-move rate, because the previous run showed a per-action
latent gap of exactly 0.0 in two rooms: a move into a wall changes nothing, the
engine never says so explicitly, and those samples make any jitter/signal ratio
unbounded. That is a real property of the dataset and belongs in the report
rather than in a divide-by-zero.

    python tests/diag_warmup.py [--rooms 4] [--steps 12] [--passes 4]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from env.bridge import BridgeConfig, BridgeError, MazeBridge
from env.renderer import FrameRenderer, RendererConfig
from model.hjepa import HJEPAConfig, FrozenEncoder, goal_cost
from train.frames import preprocess

MOVES = ("up", "down", "left", "right")
ROOMS = ("level_CxC", "level_HxI", "level_MxM", "level_FxK",
         "level_AxA", "level_PxP")


def to_array(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rooms", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FrozenEncoder(HJEPAConfig()).to(device).eval()

    renderer = FrameRenderer(RendererConfig(
        level_id=ROOMS[0], width=512, height=512, view=1, yaw=0))
    renderer.init()

    @torch.no_grad()
    def encode(frames: list[np.ndarray]) -> torch.Tensor:
        return encoder(preprocess(np.stack(frames), 224).to(device)).float()

    blocked_total = 0
    action_total = 0
    warm_jitter: list[float] = []
    cold_jitter: list[float] = []
    warm_signal: list[float] = []

    try:
        for room in ROOMS[: args.rooms]:
            try:
                bridge = MazeBridge(BridgeConfig(level_id=room, yaw=0,
                                                 observation_mode="text"))
            except BridgeError as error:
                print(f"{room}: bridge refused ({error})")
                continue
            snaps = []
            try:
                obs = bridge.observe()
                snaps.append(obs["_render_state"])
                for i in range(args.steps):
                    obs = bridge.move(MOVES[i % 4])
                    snaps.append(obs["_render_state"])
                    if obs.get("player_dead"):
                        break
            finally:
                bridge.close()

            passes = []
            for _ in range(args.passes):
                passes.append([to_array(renderer.render(s)) for s in snaps])

            print(f"\n{room}  ({len(snaps)} states, {args.passes} passes)")
            zs = [encode(p) for p in passes]

            # Byte equality of each pass against pass 1 and against pass 2.
            for p in range(1, args.passes):
                same_first = sum(1 for i in range(len(snaps))
                                 if np.array_equal(passes[0][i], passes[p][i]))
                same_prev = sum(1 for i in range(len(snaps))
                                if np.array_equal(passes[p - 1][i], passes[p][i]))
                lat_first = torch.stack([goal_cost(zs[0][i], zs[p][i])
                                         for i in range(len(snaps))]).mean()
                lat_prev = torch.stack([goal_cost(zs[p - 1][i], zs[p][i])
                                        for i in range(len(snaps))]).mean()
                print(f"  pass{p + 1} vs pass1: {same_first:3d}/{len(snaps)} identical  "
                      f"latent {lat_first:.6f}   |   "
                      f"vs pass{p}: {same_prev:3d}/{len(snaps)} identical  "
                      f"latent {lat_prev:.6f}")

            # Cold = pass1 vs pass2. Warm = pass2 onwards against each other.
            cold_jitter += torch.stack(
                [goal_cost(zs[0][i], zs[1][i]) for i in range(len(snaps))]
            ).tolist()
            for p in range(2, args.passes):
                warm_jitter += torch.stack(
                    [goal_cost(zs[p - 1][i], zs[p][i]) for i in range(len(snaps))]
                ).tolist()

            # Signal measured on a warm pass, and the blocked-move rate.
            warm = zs[-1]
            gaps = torch.stack([goal_cost(warm[i], warm[i + 1])
                                for i in range(len(snaps) - 1)])
            warm_signal += gaps.tolist()
            blocked = int((gaps < 1e-9).sum())
            blocked_total += blocked
            action_total += len(gaps)
            print(f"  blocked moves (zero latent change): "
                  f"{blocked}/{len(gaps)}")
    finally:
        renderer.close()

    cold = np.array(cold_jitter)
    hot = np.array(warm_jitter)
    sig = np.array(warm_signal)
    moving = sig[sig > 1e-9]

    print("\n" + "=" * 64)
    print("OVERALL")
    print(f"  cold jitter (pass1 vs pass2) mean : {cold.mean():.6f}  "
          f"max {cold.max():.6f}")
    if len(hot):
        print(f"  WARM jitter (pass2 onwards)  mean : {hot.mean():.6f}  "
              f"max {hot.max():.6f}")
    print(f"  per-action gap, moves that moved  : {moving.mean():.6f}  "
          f"min {moving.min():.6f}")
    print(f"  blocked moves                     : "
          f"{blocked_total}/{action_total} "
          f"({100.0 * blocked_total / max(action_total, 1):.1f}%)")
    print(f"\n  cold jitter / per-action gap : "
          f"{cold.mean() / moving.mean():.2%}")
    if len(hot):
        print(f"  WARM jitter / per-action gap : "
              f"{hot.mean() / moving.mean():.2%}")
        if hot.max() == 0.0:
            print("\n  => WARM RENDERING IS BIT-EXACT. One throwaway render per")
            print("     room makes replay fully deterministic.")
        else:
            print("\n  => warm rendering still jitters; warm-up is not the whole")
            print("     story.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
