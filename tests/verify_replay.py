"""PHASE 0's gate: replay a collected episode and check it reproduces it.

Replays the stored action list through a fresh bridge and a fresh renderer
session, then compares the result against the PNGs on disk. Reports both:

* **byte equality** -- the strict reading of the gate. Expect this to fail.
  Chromium's capture carries sub-pixel jitter that the 512->224 downsample
  removes before the encoder sees it.
* **latent agreement** -- the reading that matters, because the model consumes
  DINOv2 latents and nothing else. Measured under ``goal_cost``, the planner's
  own cost, against the latent change one action produces, so the ratio says how
  big the replay error is relative to the signal being learned.

    python tests/verify_replay.py runs/pilot/frames [--episodes 3]

Exit code 0 when the latent replay error is a small fraction of the per-action
gap; 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from env.bridge import BridgeConfig, MazeBridge
from env.renderer import FrameRenderer, RendererConfig
from model.hjepa import HJEPAConfig, FrozenEncoder, goal_cost
from train.frames import preprocess

MOVES = ("up", "down", "left", "right")

#: Replay error above this fraction of the per-action latent gap is a failure:
#: the dataset would then not be reproducible at a scale the model can detect.
TOLERANCE = 0.10


def to_array(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="collection root, e.g. runs/vision/frames")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root)
    episode_dirs = sorted(p for p in root.glob("ep_*")
                          if (p / "episode.json").is_file())
    if not episode_dirs:
        print(f"no episodes under {root}")
        return 1
    episode_dirs = episode_dirs[: args.episodes]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FrozenEncoder(HJEPAConfig()).to(device).eval()

    @torch.no_grad()
    def encode(frames: list[np.ndarray]) -> torch.Tensor:
        out = []
        for start in range(0, len(frames), 32):
            batch = preprocess(np.stack(frames[start:start + 32]), 224).to(device)
            out.append(encoder(batch).float())
        return torch.cat(out)

    all_err: list[float] = []
    all_gap: list[float] = []
    all_bytes_same = all_bytes_total = 0
    state_exact = True

    for ep_dir in episode_dirs:
        meta = json.loads((ep_dir / "episode.json").read_text(encoding="utf8"))
        actions = meta["actions"]
        level_id = meta["start_level"]
        yaw = int(meta["yaw"])
        view = meta["view"]

        stored_paths = sorted((ep_dir / "frames").glob("*.png"))
        if len(stored_paths) != len(actions) + 1:
            print(f"{ep_dir.name}: {len(stored_paths)} frames but "
                  f"{len(actions)} actions -- skipping")
            continue

        # Fresh bridge, fresh renderer session: the phase restarts at render 0,
        # which is what the collector did.
        renderer = FrameRenderer(RendererConfig(
            level_id=level_id, width=args.size, height=args.size,
            view=view, yaw=yaw))
        bridge = MazeBridge(BridgeConfig(level_id=level_id, yaw=yaw,
                                         observation_mode="text"))
        replay: list[bytes] = []
        try:
            renderer.init()
            obs = bridge.observe()
            replay.append(renderer.render(obs["_render_state"]))
            for action in actions:
                obs = bridge.move(MOVES[action])
                replay.append(renderer.render(obs["_render_state"]))
        finally:
            bridge.close()
            renderer.close()

        stored = [p.read_bytes() for p in stored_paths]
        bytes_same = sum(1 for a, b in zip(stored, replay) if a == b)
        all_bytes_same += bytes_same
        all_bytes_total += len(stored)

        z_stored = encode([to_array(b) for b in stored])
        z_replay = encode([to_array(b) for b in replay])
        err = torch.stack([goal_cost(z_stored[i], z_replay[i])
                           for i in range(len(stored))])
        gaps = torch.stack([goal_cost(z_stored[i], z_stored[i + 1])
                            for i in range(len(stored) - 1)])
        moving = gaps[gaps > 1e-9]
        all_err += err.tolist()
        all_gap += moving.tolist()

        print(f"  {ep_dir.name}  {len(stored):4d} frames  "
              f"bytes {bytes_same:4d}/{len(stored):<4d}  "
              f"latent err mean {err.mean():.8f} max {err.max():.8f}  "
              f"gap {moving.mean():.6f}  "
              f"ratio {err.mean() / moving.mean():.3%}",
              flush=True)

    if not all_gap:
        print("nothing comparable")
        return 1

    err = np.array(all_err)
    gap = np.array(all_gap)
    ratio = err.mean() / gap.mean()

    print("\n" + "=" * 66)
    print("REPLAY VERIFICATION")
    print(f"  frames compared            : {len(err)}")
    print(f"  byte-identical             : {all_bytes_same}/{all_bytes_total}")
    print(f"  latent replay error  mean  : {err.mean():.8f}")
    print(f"  latent replay error  max   : {err.max():.8f}")
    print(f"  per-action latent gap mean : {gap.mean():.6f}")
    print(f"  replay error / signal      : {ratio:.3%}")
    print(f"  tolerance                  : {TOLERANCE:.0%}")
    print("=" * 66)
    if ratio <= TOLERANCE:
        print("GATE PASSES. The action log reproduces the dataset as the model")
        print("perceives it. Frame bytes are not reproducible; latents are, to")
        print(f"within {ratio:.2%} of one action's latent change. Report that")
        print("figure as the dataset's noise floor beside every loss.")
        return 0
    print("GATE FAILS. Replay error is too large a fraction of the signal.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
