"""Does the renderer's jitter survive into the latent space the model uses?

This is the number that decides PHASE0's replay gate, and the pixel statistics
were only a proxy for it. Training and planning both happen in DINOv2's patch
latents under ``goal_cost``, so the question is not "do the bytes match" but:

    jitter   = goal_cost(z(frame rendered again), z(frame rendered first))
    signal   = goal_cost(z(next state),           z(this state))

If ``jitter / signal`` is small, replaying the action log reproduces the dataset
for every purpose the model can detect, and the gate's intent is met even though
byte equality is not. If it is not small, the frames carry history-dependent
noise the predictor cannot explain, and collection should not start.

Measured over several rooms rather than one state pair, because the earlier pixel
numbers came from a single room and a ``max_abs`` of 124 means something moves a
lot on a few pixels. Also localises the differing pixels, to name the mechanism.

    python tests/diag_latent_jitter.py [--rooms 4] [--steps 16]
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
#: Spread across the world rather than clustered, so the result is not one room.
ROOMS = ("level_HxI", "level_CxC", "level_MxM", "level_FxK",
         "level_AxA", "level_PxP")


def to_array(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def pixel_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = d > 0
    rows = np.any(changed, axis=(1, 2))
    cols = np.any(changed, axis=(0, 2))
    box = None
    if rows.any():
        r = np.where(rows)[0]
        c = np.where(cols)[0]
        box = (int(r[0]), int(r[-1]), int(c[0]), int(c[-1]))
    return {
        "changed_subpixels": int(changed.sum()),
        "pct": 100.0 * changed.sum() / d.size,
        "mean_abs": float(d.mean()),
        "max_abs": int(d.max()),
        "bbox_rows_cols": box,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rooms", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("loading frozen DINOv2 ViT-S/14 ...", flush=True)
    encoder = FrozenEncoder(HJEPAConfig()).to(device).eval()

    renderer = FrameRenderer(RendererConfig(
        level_id=ROOMS[0], width=args.size, height=args.size, view=1, yaw=0))
    renderer.init()

    @torch.no_grad()
    def encode(frames: list[np.ndarray]) -> torch.Tensor:
        batch = preprocess(np.stack(frames), 224).to(device)
        return encoder(batch).float()

    all_jitter: list[float] = []
    all_signal: list[float] = []
    worst_pixels: list[dict] = []

    try:
        for room in ROOMS[: args.rooms]:
            yaw = hash(room) % 4
            try:
                bridge = MazeBridge(BridgeConfig(level_id=room, yaw=yaw,
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

            # Pass 1 then pass 2 over the same states, in the same order. This
            # is exactly the replay the gate asks about, including the session
            # history that makes the renderer path-dependent.
            pass1 = [to_array(renderer.render(s)) for s in snaps]
            pass2 = [to_array(renderer.render(s)) for s in snaps]

            z1 = encode(pass1)
            z2 = encode(pass2)

            # jitter: same state, rendered twice along a realistic path
            jitter = torch.stack([goal_cost(z1[i], z2[i]) for i in range(len(snaps))])
            # signal: the latent change one action actually produces
            signal = torch.stack([goal_cost(z1[i], z1[i + 1])
                                  for i in range(len(snaps) - 1)])

            identical = sum(1 for i in range(len(snaps))
                            if np.array_equal(pass1[i], pass2[i]))
            px = [pixel_stats(pass1[i], pass2[i]) for i in range(len(snaps))]
            worst = max(px, key=lambda d: d["max_abs"])
            worst_pixels.append({"room": room, **worst})

            all_jitter += jitter.tolist()
            all_signal += signal.tolist()

            print(f"\n{room} (yaw={yaw}, {len(snaps)} states)")
            print(f"  byte-identical frames : {identical}/{len(snaps)}")
            print(f"  latent jitter  mean {jitter.mean():.6f}  "
                  f"max {jitter.max():.6f}")
            print(f"  latent signal  mean {signal.mean():.6f}  "
                  f"min {signal.min():.6f}")
            print(f"  jitter/signal  mean ratio "
                  f"{(jitter.mean() / signal.mean()):.4%}")
            print(f"  worst pixel frame: max_abs={worst['max_abs']} "
                  f"changed={worst['changed_subpixels']} "
                  f"bbox(r0,r1,c0,c1)={worst['bbox_rows_cols']}")
    finally:
        renderer.close()

    if not all_signal:
        print("\nno data collected")
        return 1

    jitter = np.array(all_jitter)
    signal = np.array(all_signal)
    print("\n" + "=" * 62)
    print("OVERALL")
    print(f"  states measured        : {len(jitter)}")
    print(f"  latent jitter   mean   : {jitter.mean():.6f}")
    print(f"  latent jitter   max    : {jitter.max():.6f}")
    print(f"  per-action gap  mean   : {signal.mean():.6f}")
    print(f"  per-action gap  min    : {signal.min():.6f}")
    print(f"  jitter / mean gap      : {jitter.mean() / signal.mean():.4%}")
    print(f"  WORST CASE  max jitter / min gap : "
          f"{jitter.max() / signal.min():.4%}")
    print("=" * 62)
    print("\nverdict guide: the predictor cannot explain the jitter, so it is an")
    print("irreducible noise floor. Under ~2% of the per-action gap it is")
    print("negligible beside the dynamics being learned; approaching 100% it")
    print("would mean replaying the log does not reproduce the dataset.")

    ratio = jitter.mean() / signal.mean()
    print(f"\n=> jitter is {ratio:.2%} of one action's latent change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
