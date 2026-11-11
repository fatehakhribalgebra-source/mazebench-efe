# Phase 0 — get vision frames out of the real engine

Brief for the machine with the GPU. Written 10 Nov 2026. Read it all before
starting; the first line changes what you were about to do.

---

## 0. Stop the run you were queued to do

**Do not run the `--window 31` baseline.** The project has switched from ASCII
observations to the engine's **vision** mode. There is no window in vision
mode, and the gem-head window hypothesis it was built to test no longer
exists. The 179k ASCII buffer in `runs/bigbuffer/` is also dead for this
purpose — it is mock dynamics and glyph observations, and a vision model
cannot learn from it.

Nothing above phase 0 can start until phase 0 passes its gate.

## 1. What this phase has to produce

A dataset of `(frame, action, next_frame)` from the **real** MazeBench engine.

**Gate:** 50,000 512x512 PNG frames with their actions on disk, and a replay
that reproduces the same frames from the same action sequence. If replay is
not deterministic, stop and report — everything downstream assumes it.

Nothing is trained in this phase.

## 2. Why this is worth real effort

Vision frames can only come from their renderer, which means running their
engine, which means **the dynamics become theirs too**. Every result this
project has produced so far carried "this is not a MazeBench score" because
the step function was our reimplementation in `env/mock.py`. That caveat dies
here. You are not just collecting pixels, you are buying the real physics.

## 3. The two processes, and how they compose

The engine ships two line-oriented JSON servers that are built to work
together. Both live in the MazeBenchEngine checkout.

| Process | Script | Role |
|---|---|---|
| **Bridge** | `scripts/maze-bridge.js` | Authoritative game state, scoring, the observation contract. **We already drive this** — `env/bridge.py` works. |
| **Renderer** | `scripts/maze-render-frame.js --serve` | Headless Chromium, returns a PNG per request. New. |

The join between them: every bridge observation carries a `_render_state`
field — a compact authoritative render checkpoint. The renderer accepts it as
`snapshot`. So you do **not** replay actions in the browser and hope the two
stay in step. You step the bridge, take its `_render_state`, push it to the
renderer, get the frame back.

**Use the snapshot route, not the action route.** The renderer's own
`{"command":"action"}` works, but the play page and an agent session disagree
about edge cases — the play page auto-undoes falls and agent sessions keep
deaths terminal (it is commented in `maze-render-frame.js`). Driving both with
actions will silently desynchronise the frame from the state. The snapshot
cannot.

## 4. The renderer protocol

`node scripts/maze-render-frame.js --serve`, then one JSON object per line on
stdin. From its own `--help`:

```
{"command":"init","gameId":"maze","levelId":"level_HxI","width":512,"height":512,"yaw":0}
{"command":"render","snapshot": <the bridge's _render_state>}
{"command":"frame"}
{"command":"close"}
```

Each response is one JSON line:

```
{"ok":true,"frame":"data:image/png;base64,..."}
{"ok":false,"error":"..."}
```

Payload fields that matter: `width`/`height` (default 512), `levelId`, `yaw`,
`view`, `cameraTiltDegrees` (default 58), `cameraStepDegrees` (default 18),
`cameraZoom`, `edges` (default true), `browser`.

**`view` is not what we assumed.** It is the number of rings of neighbouring
rooms drawn around the player: `1` (the default) is the classic 3x3 room
neighbourhood, up to `26`, or `"world"` for everything. The agent does **not**
see a single room. Leave it at the default — that is the benchmark's own
observation — and record the value you used.

There is also `--listen`, the same protocol over a 127.0.0.1 socket with a
`--port-file`, which survives short-lived callers. For one long collection run
`--serve` is simpler.

## 5. Setup

The renderer needs `playwright-core` and a Chromium-family browser.

- `playwright-core` from npm in the engine checkout, or point
  `MAZEBENCH_PLAYWRIGHT_CORE` at an existing copy.
- **On Windows, pass `browser` explicitly.** `DEFAULT_BROWSER_PATHS` in the
  script lists only macOS and Linux paths, so discovery will fail silently and
  you will debug the wrong thing. Give it a full path to Chrome, Edge, Brave
  or Chromium.
- `ffmpeg` is only needed for replay video. Not needed here.

`pip install mazebench` also works and brings its own Node runtime, if driving
the checkout directly fights you.

## 6. Firewall — read this before writing the collector

`_render_state` contains actor positions. It is **evaluator-only**. It is
legitimate to send it to the renderer, because the renderer is the thing that
draws the picture the agent is allowed to see. It is **not** legitimate for it
to reach the model, the objective, or the planner, directly or as a derived
feature.

Same rule as always, and the one that invalidates results if broken:
`board_state_hash`, player coordinates, gem ids, `push_count`, `moved` and
`room_changed` are evaluator-only. The engine's own README confirms the list
and adds that a blocked movement is never reported explicitly.

Keep the frames and the privileged fields in separate arrays in whatever you
write out, so a later mistake is hard rather than easy.

## 7. Collection policy

Reuse what already works. `--policy coverage` with its blocked set and epsilon
noise, and `--start-in-gem-rooms`, is what produced the only non-degenerate
ASCII buffer; the reasoning is in `REPORT.md` section 9 and still applies.

Two additions for vision:

- **Emit camera rotations deliberately.** Pure pathfinding emits none, and in
  3D they are the actions that reveal occluded geometry. A buffer without them
  cannot teach the model what rotation does.
- **Vary `yaw` across episodes** so the model does not learn one camera angle.

## 8. What to write out

Per episode: the frames (PNG bytes or a video), the action taken at each step,
and the privileged fields kept separately. Store frames as files or a shard
format, not one giant `.npz` — these are images, and 50k of them at 512x512
will not fit the pattern the old buffer used.

## 9. Report back

- frames collected, episodes, rooms covered, and the `view`/`yaw` values used
- whether replay is deterministic
- wall-clock seconds per frame (this decides whether 500k frames is feasible)
- anything the renderer did that this document did not predict

Phase 1 trains a frozen DINOv2 ViT-S/14 with a 6-layer action-conditioned
predictor, following `facebookresearch/jepa-wms`. It cannot start until the
frames exist.
