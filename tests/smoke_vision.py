"""Smoke test: can we get a real 3D frame out of the real engine at all?

Everything downstream depends on this, so it is worth its own script. It starts
the bridge, steps it, pushes each ``_render_state`` to the renderer, saves the
PNGs, and reports seconds per frame -- the number that decides whether 500k
frames is feasible.

    python tests/smoke_vision.py [--steps 12] [--out runs/smoke]

Run it before writing any collector. If it fails, nothing else can work.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.bridge import BridgeConfig, MazeBridge           # noqa: E402
from env.renderer import FrameRenderer, RendererConfig    # noqa: E402

MOVES = ("up", "down", "left", "right")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--level", default="level_HxI")
    parser.add_argument("--view", default="1")
    parser.add_argument("--yaw", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out", default="runs/smoke")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== bridge ===", flush=True)
    bridge = MazeBridge(BridgeConfig(level_id=args.level, yaw=args.yaw,
                                     observation_mode="text"))
    obs = bridge.observe()
    print(f"bridge up. action_count={obs.get('action_count')} "
          f"room={obs.get('current_room')} yaw={obs.get('yaw')}")
    snapshot = obs.get("_render_state")
    if not snapshot:
        print("FAIL: observation carried no _render_state")
        return 1
    print(f"_render_state: level_id={snapshot.get('level_id')} "
          f"actors={len(snapshot.get('actors') or [])} "
          f"yaw={snapshot.get('yaw')} pitch={snapshot.get('pitch')} "
          f"terrain_overrides={len(snapshot.get('terrain_overrides') or [])}")
    print("board:")
    print(obs.get("level", "")[:400])

    print("\n=== renderer ===", flush=True)
    view = args.view if args.view == "world" else int(args.view)
    renderer = FrameRenderer(RendererConfig(
        level_id=args.level, width=args.size, height=args.size,
        view=view, yaw=args.yaw,
    ))
    print(f"node    : {renderer.node}")
    print(f"browser : {renderer.browser}")

    t0 = time.perf_counter()
    first = renderer.init()
    print(f"init + first frame: {time.perf_counter() - t0:.2f}s, "
          f"{len(first) / 1024:.1f} KB")
    (out / "frame_init.png").write_bytes(first)

    # Confirm it is a real PNG of the size we asked for.
    import struct
    if first[:8] != b"\x89PNG\r\n\x1a\n":
        print("FAIL: not a PNG")
        return 1
    width, height = struct.unpack(">II", first[16:24])
    print(f"PNG header: {width}x{height}")
    if (width, height) != (args.size, args.size):
        print(f"WARNING: expected {args.size}x{args.size}")

    print("\n=== stepping ===", flush=True)
    digests = []
    times = []
    rng_actions = [MOVES[i % 4] for i in range(args.steps)]
    for step, direction in enumerate(rng_actions):
        result = bridge.move(direction)
        snapshot = result.get("_render_state")
        t0 = time.perf_counter()
        png = renderer.render(snapshot)
        dt = time.perf_counter() - t0
        times.append(dt)
        digest = hashlib.sha256(png).hexdigest()[:16]
        digests.append(digest)
        (out / f"frame_{step:03d}.png").write_bytes(png)
        print(f"  {step:3d} move {direction:<5} {dt:5.2f}s  "
              f"{len(png) / 1024:6.1f} KB  {digest}  "
              f"room={result.get('current_room')} "
              f"dead={result.get('player_dead')}")

    print("\n=== replay determinism ===", flush=True)
    # Same level, same actions, a fresh bridge and a fresh renderer session.
    bridge2 = MazeBridge(BridgeConfig(level_id=args.level, yaw=args.yaw,
                                      observation_mode="text"))
    bridge2.observe()
    replay = []
    for direction in rng_actions:
        result = bridge2.move(direction)
        png = renderer.render(result.get("_render_state"))
        replay.append(hashlib.sha256(png).hexdigest()[:16])
    matches = sum(1 for a, b in zip(digests, replay) if a == b)
    print(f"identical frames: {matches}/{len(digests)}")
    if matches != len(digests):
        print("first mismatches:")
        for i, (a, b) in enumerate(zip(digests, replay)):
            if a != b:
                print(f"  step {i}: {a} != {b}")

    n = len(times)
    print(f"\nseconds/frame: mean {sum(times) / n:.3f}  "
          f"min {min(times):.3f}  max {max(times):.3f}")
    print(f"=> 50k frames ~= {sum(times) / n * 50_000 / 3600:.1f} h "
          f"single-process")
    print(f"frames written to {out.resolve()}")

    bridge.close()
    bridge2.close()
    renderer.close()
    return 0 if matches == len(digests) else 2


if __name__ == "__main__":
    raise SystemExit(main())
