"""Why did replay not reproduce the frames? Separate the two possible causes.

Replay can fail for two very different reasons and they have opposite
consequences:

* **State divergence** -- the bridge took a different path the second time.
  Fatal: the actions no longer describe the transitions.
* **Renderer jitter** -- identical state, different pixels. Not fatal by itself;
  the frames are still honest observations of that state. But it has to be
  measured, because a byte-exact replay gate cannot be met and the real question
  becomes *how large* the jitter is relative to the change one action makes.

Three experiments, in increasing order of what they rule out:

1. Same bridge path twice -> compare ``_render_state`` JSON. Isolates state.
2. Same snapshot rendered twice in a row -> compare PNG bytes. Isolates the
   renderer with state held fixed.
3. Same snapshot, fresh session -> compare PNG bytes. Isolates session history.

    python tests/diag_determinism.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.bridge import BridgeConfig, MazeBridge           # noqa: E402
from env.renderer import FrameRenderer, RendererConfig     # noqa: E402

MOVES = ["up", "down", "left", "right"] * 3
LEVEL = "level_HxI"


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(data) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def run_bridge() -> list[dict]:
    bridge = MazeBridge(BridgeConfig(level_id=LEVEL, yaw=0, observation_mode="text"))
    try:
        bridge.observe()
        out = []
        for direction in MOVES:
            result = bridge.move(direction)
            out.append(result.get("_render_state"))
        return out
    finally:
        bridge.close()


def main() -> int:
    print("=== 1. is the BRIDGE deterministic? ===", flush=True)
    first = run_bridge()
    second = run_bridge()
    same = [canon(a) == canon(b) for a, b in zip(first, second)]
    print(f"identical _render_state: {sum(same)}/{len(same)}")
    if not all(same):
        idx = same.index(False)
        print(f"  first divergence at step {idx}")
        print(f"  pass 1: {canon(first[idx])[:300]}")
        print(f"  pass 2: {canon(second[idx])[:300]}")
        print("  => STATE DIVERGENCE. This is the fatal kind.")
    else:
        print("  => the bridge replays exactly. Divergence is in the renderer.")

    renderer = FrameRenderer(RendererConfig(
        level_id=LEVEL, width=512, height=512, view=1, yaw=0))
    try:
        renderer.init()
        snapshot = first[-1]

        print("\n=== 2. same snapshot, rendered twice in a row ===", flush=True)
        a = renderer.render(snapshot)
        b = renderer.render(snapshot)
        print(f"  {digest(a)} vs {digest(b)}  "
              f"{'IDENTICAL' if a == b else 'DIFFERENT'}  "
              f"({len(a)} vs {len(b)} bytes)")

        print("\n=== 3. same snapshot, three more times ===", flush=True)
        repeats = [renderer.render(snapshot) for _ in range(3)]
        print(f"  digests: {[digest(r) for r in repeats]}")
        print(f"  all equal to first: {all(r == a for r in repeats)}")
        print(f"  distinct digests over 5 renders: "
              f"{len({digest(x) for x in [a, b, *repeats]})}")

        print("\n=== 4. interleave two different snapshots, then come back ===",
              flush=True)
        other = first[0]
        back = renderer.render(other)
        again = renderer.render(snapshot)
        print(f"  snapshot A first render : {digest(a)}")
        print(f"  snapshot B              : {digest(back)}")
        print(f"  snapshot A again        : {digest(again)}  "
              f"{'IDENTICAL' if again == a else 'DIFFERENT'}")

        Path("runs/diag").mkdir(parents=True, exist_ok=True)
        Path("runs/diag/a1.png").write_bytes(a)
        Path("runs/diag/a2.png").write_bytes(b)
        Path("runs/diag/a3.png").write_bytes(again)
        print("\n  wrote runs/diag/a1.png a2.png a3.png "
              "(same state, three renders)")
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
