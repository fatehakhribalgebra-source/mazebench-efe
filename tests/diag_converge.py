"""How many renders does a state need before the frame stops changing?

The GPU hypothesis is dead: ``launchBrowser`` already launches with
``--disable-gpu --use-angle=swiftshader``, so rasterisation is software and
deterministic already. What is left in the evidence is this shape:

* the same snapshot rendered 5x in a row -> one distinct frame
* A, then B, then A once -> a different A
* A, then B, then A twice (``diag_settle``) -> still a different A

That is progressive accumulation: the renderer keeps buffers that carry history
from whatever was drawn before, and a state needs several redraws before it
reaches its own fixed point. If so, there is a *k* at which the frame for a state
stops depending on what preceded it, and the collector simply renders *k* times
and keeps the last. That makes replay bit-exact at a known cost of k renders per
frame.

The test: approach the same target state from two different predecessors, with k
renders each, and find the smallest k where both give the identical frame.

    python tests/diag_converge.py [--max-k 10]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.bridge import BridgeConfig, MazeBridge           # noqa: E402
from env.renderer import FrameRenderer, RendererConfig     # noqa: E402

MOVES = ("up", "down", "left", "right")
ROOMS = ("level_HxI", "level_FxK", "level_CxC")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--rooms", type=int, default=3)
    args = parser.parse_args()

    renderer = FrameRenderer(RendererConfig(
        level_id=ROOMS[0], width=512, height=512, view=1, yaw=0))
    renderer.init()

    def approach(predecessors: list[dict], target: dict, k: int) -> bytes:
        """Render the predecessors, then the target k times. Return the last."""
        for snapshot in predecessors:
            renderer.render(snapshot)
        out = b""
        for _ in range(k):
            out = renderer.render(target)
        return out

    verdict_k = None
    try:
        for room in ROOMS[: args.rooms]:
            bridge = MazeBridge(BridgeConfig(level_id=room, yaw=0,
                                             observation_mode="text"))
            snaps = []
            try:
                obs = bridge.observe()
                snaps.append(obs["_render_state"])
                for i in range(6):
                    obs = bridge.move(MOVES[i % 4])
                    snaps.append(obs["_render_state"])
            finally:
                bridge.close()

            target = snaps[-1]
            # Two genuinely different histories leading into the same state.
            path_a = [snaps[0]]
            path_b = [snaps[2], snaps[1], snaps[3]]

            print(f"\n{room}")
            print(f"  {'k':>3}  {'from path A':>14}  {'from path B':>14}   match")
            first_match = None
            for k in range(1, args.max_k + 1):
                fa = approach(path_a, target, k)
                fb = approach(path_b, target, k)
                ok = fa == fb
                if ok and first_match is None:
                    first_match = k
                print(f"  {k:>3}  {digest(fa):>14}  {digest(fb):>14}   "
                      f"{'YES' if ok else 'no'}")
            print(f"  => converges at k={first_match}" if first_match
                  else f"  => no convergence up to k={args.max_k}")
            if first_match is not None:
                verdict_k = max(verdict_k or 0, first_match)
            else:
                verdict_k = None
                break
    finally:
        renderer.close()

    print("\n" + "=" * 58)
    if verdict_k:
        print(f"CONVERGES. Render each state {verdict_k}x and keep the last;")
        print(f"replay is then bit-exact. Cost: {verdict_k}x render time")
        print(f"(~{0.131 * verdict_k:.2f} s/frame, "
              f"{0.131 * verdict_k * 50_000 / 3600:.1f} h for 50k frames).")
    else:
        print("DOES NOT CONVERGE. The frame for a state keeps depending on")
        print("what preceded it however many times it is redrawn, so a")
        print("deterministic frame cannot be obtained this way.")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
