# Phase 0 results — vision frames from the real engine

Measured 11 Nov 2026 on Windows 11, RTX 4070 SUPER (12 GB, driver 560.94),
Python 3.12.0, torch 2.5.1+cu121, `MazeBenchEngine` at `07aa03a5`.

**Status: the frame pipeline works and the replay gate passes. Collection was
stopped early at the user's request — 50k frames were not reached.**

---

## 1. What was answered

| PHASE0 question | Answer |
| --- | --- |
| Can the renderer produce frames on Windows? | **Yes.** Chrome passed explicitly; its own default paths really are macOS/Linux only. |
| Seconds per frame | **0.131 s** render, **0.204 s** wall including the per-episode browser launch |
| Is 500k frames feasible? | At 0.204 s/frame, **~28 h single-process**. 50k is ~2.8 h of pure collection, ~4.7 h at the observed episode yield. |
| Is replay deterministic? | **State: yes, bit-exact. Pixels: no. Latents: to within 1.72%.** See §3. |
| `view` / `yaw` used | `view=1` (the benchmark default, 3x3 room neighbourhood); `yaw` varied across episodes over {0,1,2,3}, fixed within each |

## 2. Frames collected

Collection was interrupted, so these are partial numbers, not the gate.

| | |
| --- | --- |
| Episodes attempted | 67 |
| Episodes kept (>= 81 frames) | 43 |
| Frames encoded to latents | **5,464** |
| Rooms covered | 8+ distinct, sampled from the 256 shipped `level_{A..P}x{A..P}` |
| Capture | 512x512 PNG, ~64 KB each |
| Into the model | 224x224 RGB, ImageNet-normalised, 256 patches x 384 dims |

Resume with:

```
python -m train.collect_vision --episodes 800 --steps 128 \
    --target-frames 50000 --out runs/vision/frames --seed 1 --resume
```

### Episode lengths are bimodal, and that matters

57% of episodes run the full 129 frames; **p10 is 2 frames**. 43% end in death
(18 `player_dead`, 5 bridge refusals after death). The player spawns near a
hazard and the coverage policy walks into it.

Window yield per level, measured on 54 episodes and extrapolated to 50k frames:

| Stride | Frames needed | Episodes usable | Windows (at 4.5k) | Projected at 50k |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 44/54 | 4,286 | ~47k |
| 4 | 21 | 39/54 | 3,688 | ~41k |
| 16 | 81 | 33/54 | 1,567 | ~17k |

So level 3 is thinner but **not starved**, which was PHASE0's worry.

**Known bias, not yet corrected:** because deaths truncate, rooms with hazards
near the spawn are under-represented. The policy was deliberately left alone
mid-run — PHASE0 says reuse `--policy coverage`, and changing it partway would
make the dataset inhomogeneous.

## 3. The replay gate — read this part

PHASE0 says to stop and report if replay is not deterministic. It is not, byte
exactly, and the reason matters more than the fact.

**The bridge is bit-exact.** Same level, same actions, fresh process → 12/12
identical `_render_state`. Actions explain state transitions exactly. The fatal
kind of divergence does not happen.

**The renderer advances a per-session phase.** `tests/diag_converge.py` showed
the frames for one fixed state form a single chain: approaching that state with
*k* renders gives chain element *k*, regardless of the path taken. Something
redraws one tick per `render()` call.

Three hypotheses were tested and rejected:

- *GPU rasterisation* — `launchBrowser` **already** passes `--disable-gpu
  --use-angle=swiftshader`. Software rasterisation was never not in force.
- *Asset warm-up* — `tests/diag_warmup.py`: warm jitter 4.63% vs cold 6.98%.
  Warming helps slightly and fixes nothing.
- *A settle race* — double-rendering to absorb it does not help.

**Measured fidelity** (`tests/verify_replay.py`, on real collected episodes):

| | |
| --- | --- |
| Byte-identical frames | 12/349 |
| Latent replay error, mean | 0.00212 |
| Per-action latent gap, mean | 0.12335 |
| **Replay error / signal** | **1.72%** |

A fixed state re-rendered 8 times in one session drifts by **exactly 0**. So the
renderer is stable given state *and* phase; only the phase moves.

**Consequence for the collector:** one renderer session **per episode**, so each
episode's phase starts at 0 and replays independently. Sharing one session is
what made the first measurement read 16.8% — the second pass ran 17 renders deep
against a phase-offset first pass. That figure was a measurement artefact.

**What this means in practice:** the action log reproduces the dataset as the
model perceives it. Anything keyed off exact frame bytes (hash-based frame
dedup, byte-exact regression tests on frames) will not work.

## 4. Other findings the brief did not predict

**14.6% of moves are blocked and produce exactly zero latent change.** The
engine never reports a blocked move, so the buffer legitimately contains
action-labelled no-ops. Two consequences: any jitter-to-signal ratio is
undefined on one move in seven, and the identity baseline is partly computed
over frames where identity is the *correct* answer.

**`transformers` and the cu121 pin are incompatible.** Current `transformers`
(5.22.2) requires torch >= 2.6; the cu121 index tops out at torch 2.5.1.
Pinned `transformers==4.49.0`. `requirements.txt` still says `>=4.46`, which
resolves to something that cannot import `AutoModel`.

**Batch 64 is 33x slower than batch 32**, not faster. Peak allocation at 64 is
13.98 GB on a 12 GB card, so CUDA spills to host memory over WDDM and throughput
collapses. `train_hjepa.py`'s default of 32 is already correct.

| | batch 32 | batch 64 |
| --- | ---: | ---: |
| level 1 (stride 1) | 7.46 ms/window | 249.26 ms/window |
| level 2 (stride 4) | 11.42 ms/window | 131.23 ms/window |
| level 3 (stride 16) | 7.46 ms/window | 118.44 ms/window |

**bf16 autocast gives 2.47x**, uniformly across levels, peak 7.22 GB. Added as
an opt-in `--amp` flag. The identity baseline is computed *outside* autocast so
the number the gate is judged on stays fp32 and comparable.

**The `beats_identity` flag is too lenient.** It is `ratio < 1.0`, and on the
pilot it printed `PASS` at **0.994** — a model that had learned nothing. Judge
"clearly beats identity" against a real threshold, not that flag. This is the
same shape of error as the 2.9x incident in `REPORT.md`.

**No camera rotations are collected.** `GOAL.md` fixes yaw within an episode,
and the model has exactly four action tokens. A rotated frame pair would be
labelled with a move that did not cause it, which teaches the predictor noise
rather than rotation. `MoveOnlyCoverage` now raises if a non-move escapes the
policy. (This contradicts PHASE0 §7; `GOAL.md` wins, and the user confirmed.)

## 5. Environment traps on this machine

1. No `git` and no `node` installed. Portable Node v26.13.0 and MinGit 2.56.1
   under `C:\mazebench\tools\`, put on PATH by `C:\mazebench\env.ps1`.
2. **A Fortinet firewall MITMs TLS for `registry.npmjs.org` only**, and its CA
   is not in the Windows trust store, so `npm install` fails with
   `UNABLE_TO_VERIFY_LEAF_SIGNATURE` and `--use-system-ca` cannot help.
   Fix: `npm install --registry=https://registry.yarnpkg.com` — an official
   mirror serving identical tarballs, so TLS verification *and* the lockfile's
   integrity hashes both stay intact. PyPI, pytorch, GitHub, HuggingFace are
   all clean. Do not reach for `strict-ssl false`.
3. Work in a short path. `MAX_PATH` killed `git clone` in a deep folder before
   `node_modules` or 50k frames were even in play.

## 6. Not done

- 50k-frame gate (stopped at 5,464 encoded frames)
- training, so **no identity-baseline ratios exist yet**
- goal-reaching success rate, 3 levels vs 1 level — the headline number
- the early check of whether level 1 beats identity at all

Nothing here is a claim about MazeBench performance. No model has been trained
on real frames yet.

## 7. New code

| File | What |
| --- | --- |
| `env/renderer.py` | Persistent `maze-render-frame.js --serve` client, snapshot route, explicit browser path |
| `train/collect_vision.py` | Collector; fresh renderer session per episode; privileged fields written separately |
| `train/encode_vision.py` | Encode-once driver → one `.npz` per episode |
| `agent/eval_goal.py` | Goal-reaching harness. Planner sees latents only; the evaluator judges arrival from engine actor positions, never from the planner's own cost |
| `tests/verify_replay.py` | The gate: replays an episode and reports byte and latent fidelity |
| `tests/smoke_vision.py`, `tests/diag_*.py` | The six diagnostics that produced §3 |
