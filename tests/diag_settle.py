"""Is the renderer's path-dependence a settle race, and how big is it?

``diag_determinism.py`` established:

* the bridge replays exactly, so state is not the problem
* the same snapshot rendered twice in a row is byte-identical
* but A, B, A gives a different A than the first time

The shape of that -- identical on an immediate repeat, different after an
intervening state -- is a **settle race**: ``waitUntilSettled(page, 20)`` returns
before Chromium has finished drawing, so the captured frame depends on how much
work was still pending, which depends on the previous state.

If that is the cause, then rendering the snapshot a second time and keeping the
second frame should be path-independent, because the first render absorbs the
settling. This measures both the hypothesis and the magnitude of the error, with
a stdlib PNG decoder so it does not need numpy.

    python tests/diag_settle.py
"""

from __future__ import annotations

import hashlib
import struct
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.bridge import BridgeConfig, MazeBridge           # noqa: E402
from env.renderer import FrameRenderer, RendererConfig     # noqa: E402

LEVEL = "level_HxI"


# ---------------------------------------------------------------------------
# minimal PNG decode: enough for Chromium's own output (8-bit, non-interlaced)
# ---------------------------------------------------------------------------

def decode_png(data: bytes) -> tuple[int, int, int, bytearray]:
    """-> (width, height, channels, raw pixel bytes). Stdlib only."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos = 8
    width = height = depth = color = interlace = 0
    idat = bytearray()
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    if depth != 8 or interlace != 0:
        raise ValueError(f"unsupported PNG: depth={depth} interlace={interlace}")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    raw = zlib.decompress(bytes(idat))

    stride = width * channels
    out = bytearray(height * stride)
    prev = bytearray(stride)
    src = 0
    for y in range(height):
        filt = raw[src]
        src += 1
        line = bytearray(raw[src:src + stride])
        src += stride
        if filt == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filt == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filt == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filt == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif filt != 0:
            raise ValueError(f"bad filter {filt}")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, channels, out


def compare(png_a: bytes, png_b: bytes) -> dict:
    wa, ha, ca, a = decode_png(png_a)
    wb, hb, cb, b = decode_png(png_b)
    if (wa, ha, ca) != (wb, hb, cb):
        return {"shape_mismatch": True}
    diffs = 0
    total = 0
    worst = 0
    # Compare RGB only; alpha is constant in these captures.
    for i in range(0, len(a), ca):
        for ch in range(min(3, ca)):
            d = abs(a[i + ch] - b[i + ch])
            if d:
                diffs += 1
                total += d
                worst = max(worst, d)
    pixels = wa * ha
    return {
        "pixels": pixels,
        "changed_subpixels": diffs,
        "pct_subpixels": 100.0 * diffs / (pixels * 3),
        "mean_abs_over_all": total / (pixels * 3),
        "max_abs": worst,
    }


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    bridge = MazeBridge(BridgeConfig(level_id=LEVEL, yaw=0, observation_mode="text"))
    bridge.observe()
    snaps = []
    for direction in ["up", "right", "down", "left", "up", "right"]:
        snaps.append(bridge.move(direction).get("_render_state"))
    bridge.close()

    renderer = FrameRenderer(RendererConfig(
        level_id=LEVEL, width=512, height=512, view=1, yaw=0))
    try:
        renderer.init()
        A, B = snaps[0], snaps[3]

        print("=== single render, path A -> B -> A ===", flush=True)
        a1 = renderer.render(A)
        renderer.render(B)
        a2 = renderer.render(A)
        print(f"  A first {digest(a1)} | A again {digest(a2)} | "
              f"{'IDENTICAL' if a1 == a2 else 'DIFFERENT'}")
        if a1 != a2:
            print(f"  pixel diff: {compare(a1, a2)}")

        print("\n=== double render (keep the second), path A -> B -> A ===",
              flush=True)

        def settled(snapshot):
            renderer.render(snapshot)
            return renderer.render(snapshot)

        d1 = settled(A)
        settled(B)
        d2 = settled(A)
        print(f"  A first {digest(d1)} | A again {digest(d2)} | "
              f"{'IDENTICAL' if d1 == d2 else 'DIFFERENT'}")
        if d1 != d2:
            print(f"  pixel diff: {compare(d1, d2)}")

        print("\n=== how big is one ACTION, for scale? ===", flush=True)
        one_step = compare(settled(snaps[0]), settled(snaps[1]))
        print(f"  consecutive states differ by: {one_step}")

        print("\n=== full 6-step replay, double-rendered ===", flush=True)
        t0 = time.perf_counter()
        pass1 = [digest(settled(s)) for s in snaps]
        pass2 = [digest(settled(s)) for s in snaps]
        dt = time.perf_counter() - t0
        match = sum(1 for x, y in zip(pass1, pass2) if x == y)
        print(f"  identical: {match}/{len(snaps)}")
        print(f"  seconds/frame (double-rendered): {dt / (2 * len(snaps)):.3f}")
        if match != len(snaps):
            for i, (x, y) in enumerate(zip(pass1, pass2)):
                if x != y:
                    print(f"    step {i}: {x} != {y}")
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
