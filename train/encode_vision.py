"""Run the frozen encoder over collected frames once, and write latents.

``train/frames.py`` explains why this is a separate stage: re-running a ViT over
the same frames every epoch is the easiest way to waste a GPU. This walks a
collection root, encodes each episode, and writes one ``.npz`` per episode named
after it, so the episode-level train/val split lines back up with the frame
directories for evaluation.

    python -m train.encode_vision --frames runs/vision/frames --out runs/vision/latents

Only the frames and the actions cross into the latents. ``privileged.json`` is
not read here at all -- that is the point of it being a separate file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from model.hjepa import HJEPAConfig, FrozenEncoder
from train.frames import encode_episodes


def load_frames(ep_dir: Path) -> np.ndarray:
    paths = sorted((ep_dir / "frames").glob("*.png"))
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="runs/vision/frames")
    p.add_argument("--out", default="runs/vision/latents")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--min-frames", type=int, default=81)
    args = p.parse_args()

    root = Path(args.frames)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(d for d in root.glob("ep_*")
                          if (d / "episode.json").is_file())
    if not episode_dirs:
        print(f"no episodes under {root}")
        return 1

    print(f"device: {args.device}")
    encoder = FrozenEncoder(HJEPAConfig()).to(args.device).eval()

    total_frames = 0
    encoded = 0
    skipped_short = 0
    t_start = time.perf_counter()

    for ep_dir in episode_dirs:
        out_path = out_root / f"{ep_dir.name}.npz"
        meta = json.loads((ep_dir / "episode.json").read_text(encoding="utf8"))
        actions = np.asarray(meta["actions"], dtype=np.int8)

        if out_path.is_file() and args.skip_existing:
            with np.load(out_path) as data:
                total_frames += len(data["latents"])
            encoded += 1
            continue

        if meta["n_frames"] < args.min_frames:
            skipped_short += 1
            continue

        frames = load_frames(ep_dir)
        if len(frames) != len(actions) + 1:
            print(f"  {ep_dir.name}: {len(frames)} frames vs {len(actions)} "
                  f"actions -- skipping")
            continue

        info = encode_episodes(
            encoder, frames, actions, out_path,
            batch_size=args.batch_size, device=args.device,
            image_size=args.image_size,
        )
        # The episode's rooms ride along so held-out-room splits are possible
        # later. Agent-visible, not privileged. Stored as a JSON string rather
        # than an object array: an object dtype would force allow_pickle=True
        # on every np.load downstream, including in LatentEpisodes.
        with np.load(out_path) as data:
            payload = {k: data[k] for k in data.files}
        np.savez_compressed(
            out_path, **payload,
            rooms_json=np.asarray(json.dumps(meta.get("rooms", []))),
            yaw=np.asarray(int(meta.get("yaw", 0))),
        )

        total_frames += info["frames"]
        encoded += 1
        rate = total_frames / max(time.perf_counter() - t_start, 1e-9)
        print(f"  {ep_dir.name}  {info['frames']:4d} frames  "
              f"{info['shape']}  total={total_frames}  "
              f"{rate:.1f} frames/s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nencoded {encoded} episodes, {total_frames} frames in "
          f"{elapsed / 60:.1f} min ({total_frames / max(elapsed, 1e-9):.1f} frames/s)")
    if skipped_short:
        print(f"skipped {skipped_short} episodes shorter than {args.min_frames} frames")
    print(f"latents in {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
