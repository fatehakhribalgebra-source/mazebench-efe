# Handoff

Everything needed to continue this project on a different machine. Written
9 Nov 2026, because the assistant's own memory of this work is stored per
machine and does not travel — this file is the continuity, so keep it current.

## 1. Set the machine up

```bash
git clone <this repo> mazebench_efe && cd mazebench_efe
python -m venv .venv && . .venv/Scripts/activate    # Linux/mac: .venv/bin/activate
pip install -r requirements.txt
```

Known-good versions on the original machine: `torch 2.5.1+cu121`,
`numpy 2.3.2`, CUDA 12.1. Any recent torch should work; the model is plain
convolutions, a GRU and linear heads.

Real level geometry comes from a second repository, which is **not** vendored
here. Clone it and point the loader at it:

```bash
git clone https://github.com/mazebench/MazeBenchEngine.git
git -C MazeBenchEngine checkout 07aa03a5c0ee8b4e0b025a52793055e89cb68bfe
export MAZEBENCH_REPO=$PWD/MazeBenchEngine     # Windows: set MAZEBENCH_REPO=...
```

`07aa03a` (23 Aug 2026) is the commit every number in this repo was produced
against. `env/real_levels.py` also looks in `~/Documents/mazebench_src/repo`
if the variable is unset.

Check the install:

```bash
python -m pytest tests/ -q
```

47 tests. They use the mock engine only, so they pass without the clone; the
real-levels tests skip themselves when `MAZEBENCH_REPO` is missing.

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

```bash
python -u -m train.train_wm --engine mock --real-levels \
  --buffer runs/bigbuffer/buffer.npz --window 31 --levels 2 \
  --members 5 --epochs 30 --steps-per-epoch 250 --out runs/baseline
```

```bash
python -u play.py --arms random,memoryless,no_epistemic,full --episodes 10 \
  --checkpoint runs/baseline/final.pt --real-levels --window 31 --out runs/ladder
```

```bash
python -u -m train.train_wm --engine mock --real-levels \
  --buffer runs/bigbuffer/buffer.npz --window 31 --levels 2 \
  --index-levels 4 --out runs/d2       # then --levels 3, --levels 4
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
