# Handoff

> **Read `GOAL.md` first.** It states what we are actually doing right now —
> reproducing the published JEPA goal-reaching result on MazeBench — and it
> overrides both this file and `REPORT.md` wherever they disagree. Then
> `PHASE0.md` for collecting frames.
>
> `REPORT.md` is the project's history, traps and benchmark facts, and is still
> worth reading for those. Its queued runs are **cancelled**: do not run the
> `--window 31` baseline. This file is the machine setup (section 1); sections
> 3, 4 and 5 below are superseded.

Everything needed to continue this project on a different machine. Written
9 Nov 2026, because the assistant's own memory of this work is stored per
machine and does not travel — this file is the continuity, so keep it current.

## 1. Set the machine up

Windows / PowerShell, deliberately without activating the venv — activation
needs a relaxed execution policy and is the most common place this gets stuck.
Call the venv's interpreter by path instead and nothing needs enabling:

```powershell
git clone https://github.com/fatehakhribalgebra-source/mazebench-efe.git
cd mazebench-efe
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

From here on, **`.venv\Scripts\python.exe` is "python"** in every command
below. On Linux or macOS it is `.venv/bin/python` and everything else is the
same.

`requirements.txt` asks only for `torch>=2.1` and `numpy>=1.24`. On Windows
plain pip gives a **CPU-only** torch, which runs but trains slowly. For the GPU
build, install it from PyTorch's own index first:

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Confirm which one you got — it must print `True` to train on the GPU:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Known-good on the original machine: `torch 2.5.1+cu121`, `numpy 2.3.2`,
CUDA 12.1. Any recent torch works; the model is plain convolutions, a GRU and
linear heads.

Real level geometry comes from a second repository, which is **not** vendored
here. Clone it and point the loader at it:

Clone it inside this directory — `.gitignore` already excludes it:

```powershell
git clone https://github.com/mazebench/MazeBenchEngine.git
git -C MazeBenchEngine checkout 07aa03a5c0ee8b4e0b025a52793055e89cb68bfe
$env:MAZEBENCH_REPO = "$PWD\MazeBenchEngine"
```

`07aa03a` (23 Aug 2026) is the commit every number in this repo was produced
against. **`$env:` only lasts for that one terminal window** — set it again in
any new one, or `--repo-root` the path explicitly.

Check the install:

```powershell
.venv\Scripts\python.exe tests\test_all.py
```

47 tests, expect `47/47 passed`. **Do not use pytest** — `requirements.txt`
does not install it, and the suite has its own runner that needs nothing extra.
The tests drive the mock engine, so they pass with or without the clone above;
the real-levels ones return early when it is missing, which means a green suite
does *not* prove `MAZEBENCH_REPO` is set correctly. The `torch.cuda` line above
is the check that matters for speed, and section 4's first run is the check
that matters for the levels.

## 2. What this is

An agent that plays [MazeBench](https://mazebench.com/) by minimising
**expected free energy** over imagined rollouts of a hierarchical latent world
model, instead of maximising a reward.

```
G(pi) = sum_t gamma^t [ risk_t + ambiguity_t - information_gain_t ]
```

Information gain is the only subtracted term, so it is the only reason the
agent walks into an unmapped room. It combines five-member ensemble
disagreement (Plan2Explore) with count-based novelty over cells, local
patterns and pushable-block layouts. The block-layout term is hashed with a
Zobrist key, which is what pays the agent for pushing a box onto a gem.

Planning is CEM: 256 action sequences of 8 steps, 4 refinement iterations,
scored entirely on imagined latents. The architecture write-up with figures is
`report.html` in this repo.

## 3. State of play

Trained, on mock dynamics over real geometry:

| | value | where |
|---|---|---|
| changed-tile accuracy (held out) | 0.874 | `runs/demo/report.json`, 30 epochs |
| level-2 predictor loss | 0.145 | `runs/big/report.json`, 179k buffer |
| level-2 VICReg variance (lower better) | 0.077 | same |
| unique states/episode, full vs random | 54.2 ± 3.7 vs 19.0 ± 1.6 | `runs/arms/results.json` |

The collected buffer — `runs/bigbuffer/buffer.npz`, 700 episodes, 179,139
transitions, 552 gem events, 2,123 room changes — is committed. It cost an
overnight run; train from it with `--buffer` rather than collecting again.

## 4. Next three runs, in order

**1 — the baseline.** This is also the test of the gem-head theory in section 5,
which is why `--window 31` is on it.

```powershell
.venv\Scripts\python.exe -u -m train.train_wm --engine mock --real-levels --buffer runs/bigbuffer/buffer.npz --window 31 --levels 2 --members 5 --epochs 30 --steps-per-epoch 250 --out runs/baseline
```

Watch the per-epoch line: `val changed` should climb past 0.8, and
`val gem_rec` is the number the whole run exists to move off 0.167.

**2 — the ablation ladder.**

```powershell
.venv\Scripts\python.exe -u play.py --arms random,memoryless,no_epistemic,full --episodes 10 --checkpoint runs/baseline/final.pt --real-levels --window 31 --out runs/ladder
```

**3 — depth.** Run three times, changing `--levels` and `--out` each time;
`--index-levels 4` stays fixed or the comparison is meaningless (section 6).

```powershell
.venv\Scripts\python.exe -u -m train.train_wm --engine mock --real-levels --buffer runs/bigbuffer/buffer.npz --window 31 --levels 2 --index-levels 4 --out runs/d2
```

## 5. Open problems

- **The gem head overfits; it is not short of data.** On the 179k buffer train
  recall reaches 1.000 while held-out sits at 0.167 — worse than the 7k pilot's
  0.21, despite 5× the gem events and 40× oversampling. Collecting more will
  not fix it. Leading suspect: that run used `--window 15`, which cannot frame
  a 16-tile room, so the gem is often outside the view at the step it has to be
  predicted. `--window 31` is the first test, and it is why the baseline above
  carries that flag.
- **The room head has no recall metric** — only a loss reading ~0.0003 at a
  1.2% base rate, which is uninformative. This is the same trap `gem_acc` fell
  into. Add recall and precision before trusting any room-change claim.
- **The real engine has never been executed from an assistant session.** Claude
  Code's auto-mode classifier refuses to run the cloned repo ("Code from
  External"). Geometry, gem and block placement and room adjacency are parsed
  from the real level files, but the dynamics stepping them are a
  reimplementation in `env/mock.py`. **No number in this repo is a MazeBench
  score.** Unblocking it needs a Bash permission rule for
  `node .../maze-bridge.js`.
- The frozen V-JEPA 2 / SigLIP 2 transfer path is wired up at pinned revisions
  but no real frame has passed through it.

## 6. Traps already paid for — do not rediscover

- **Elevation.** A `.+#+#` cell is ground at height 2, not a wall. Collapsing
  stacks to walls sealed the levels (104 of 256 cells reachable, no episode
  ever left its start room). Heights are parsed and a move is allowed when
  surfaces differ by ≤1, giving 244 reachable.
- **Collection policy.** Random and heuristic walks score 0 gems and 0 room
  changes on real geometry. Use `--policy coverage`, which needs its blocked
  set (without it the move rate is 0.24, worse than random) and its epsilon
  noise (pure pathfinding emits **no camera rotations**, half the action set).
- **Score `changed_tile_acc`, never `tile_acc`** — "nothing moves" scores ~99%
  on a grid world.
- **Depth comparisons need `--index-levels 4`.** Indexed at its own stride a
  4-level model draws 2,689 chunk samples where a 2-level one draws 6,578; the
  deep model loses at *level 2* for being starved, not for being worse.
- **The observation firewall is the core constraint.** `board_state_hash`,
  player coordinates, gem ids, `push_count`, `moved` and `room_changed` are
  evaluator-only. Reading any of them invalidates an exploration result.
  `test_firewall_never_leaks` guards it.
- Only `P`/`p` and `G`/`g` are fixed glyphs; `hide_names` permutes everything
  else, so the hazard safety mask only holds under canonical glyphs.
- A local `import` inside one branch of `main()` in `train/train_wm.py` once
  shadowed the module-level import on every path. `--collect-only` passed while
  full training died with `UnboundLocalError`. Fixed; the shape of the bug is
  worth remembering.

## 7. Benchmark facts, measured from the engine repo

- World is 16×16 **rooms**, each level 16×16 **tiles**, `camera_view` 16×16 —
  the ASCII view frames a whole room, so an egocentric window must be
  2·16−1 = **31** to cover one from any corner.
- `world_map.json` places all 256 levels with no duplicates, so room adjacency
  in `env/real_levels.py` is the benchmark's own.
- **The world holds 90 gems but `game_won_gem_count` is 100.** The win
  condition is unreachable in the shipped world; report gems out of 90. This
  fits the published best scores of roughly 10–13%.
- The default start room `level_HxI` contains **0 gems**; its only exit is
  row 0, columns 6–9. 86 of 256 rooms hold a gem.
