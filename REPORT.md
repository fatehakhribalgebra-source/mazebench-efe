# MazeBench EFE Agent — full report

**Written 10 Nov 2026 on machine A. Audience: a fresh Claude session on
machine B, with no memory of this project.**

Read this whole file before running anything. It is the briefing, the
measured state, and the next three runs. `HANDOFF.md` section 1 is the
machine setup in detail; section 10 below is the condensed version.

Every number in this report was read out of the committed `runs/*/report.json`
and `runs/arms/results.json` on 10 Nov, not recalled. Where a claim is not yet
supported by a measurement, it says so.

---

## 1. The research question

MazeBench is a 256-room grid world. An agent sees an ASCII camera view, emits
one action at a time, and is scored on gems collected. Published best scores
are around 10–13%, which is 10–13 gems.

The question this project asks is **not** "can we score well on MazeBench".
It is:

> Does an agent that plans by **minimising expected free energy** over a
> **hierarchical latent world model** explore better than one driven by
> reward or by novelty alone — and does the hierarchy earn its keep?

The scientific content is in the comparisons (with/without each EFE term,
2 vs 3 vs 4 levels), not in the absolute gem count. This matters because of
section 3: absolute scores here are not MazeBench scores and must never be
reported as such.

Three things the agent is supposed to be able to do, which a reward-maximiser
cannot do without being told to:

1. Walk into an unmapped room because it is unmapped.
2. Imagine pushing a box onto a gem before ever having done it.
3. Carry a model learned in one part of the world into another.

---

## 2. The algorithm

### 2.1 The objective

The planner minimises

```
G(pi) = sum_{t=1..H} gamma^t [ risk_t + ambiguity_t - information_gain_t ]
```

with `H = 8`, `gamma = 0.95`. Information gain is the **only subtracted**
term, so it is the only reason the agent ever leaves known ground.

Actual weights, from `agent/efe.py` (`EFEConfig`):

| term | weight | what it is |
|---|---|---|
| **risk** | | divergence from preferred outcomes |
| `w_gem` | 10.0 | expected gems collected this step |
| `w_gem_proximity` | 1.0 | dense shaping toward a *visible* gem, capped at 8 tiles |
| `w_room_change` | 2.0 | entering any new room |
| `w_box_novelty` | 0.5 | a block configuration not seen before |
| `w_death` | 20.0 | strongly dispreferred |
| `w_hazard_proximity` | 0.3 | soft standoff from hazards |
| **ambiguity** | | |
| `w_ambiguity` | 0.5 | predicted observation entropy — penalises *unreadable* states |
| **information gain** | | all subtracted |
| `w_disagreement` | 2.0 | variance across the 5-member ensemble (Plan2Explore) |
| `w_cell_novelty` | 3.0 | count-based, per cell |
| `w_pattern_novelty` | 2.0 | count-based, per local pattern |
| `w_abstract_novelty` | 1.0 | count-based, in abstract latent space |

`EFEConfig.scaled(exploit)` trades the two sides off: at `exploit=1.0` the
gem terms are at full weight and `w_disagreement`, `w_cell_novelty` and
`w_pattern_novelty` are **halved**. See section 7, run 2 — this setting was a
mistake in the ablation that already ran.

Note the shape of ambiguity: it is *added*, so the agent avoids states whose
observation it cannot predict, while information gain *rewards* states whose
latent it cannot predict. Confusion is punished, ignorance is paid for. That
split is the whole reason this is not just a novelty bonus.

### 2.2 The world model (`model/world_model.py`)

Prediction happens entirely in representation space — there is no pixel or
tile decoder above level 1. This is the JEPA choice, and it is why
representation collapse is a live failure mode rather than a theoretical one.

**`GridEncoder`** — `nn.Embedding(VOCAB_SIZE, 32)` over glyphs, a
`Linear(SCALAR_DIM, 16)` for scalars, concatenated to 48 channels, then
`Conv2d(48 -> 64, 3)`, 3 residual blocks, `GroupNorm(8, 64)`. `pool()`
concatenates spatial mean **and** max, so a single gem in a corner survives
pooling, then `Linear(128 -> 256) + LayerNorm`.

**`Level1Predictor`** — action embedding `(N_ACTIONS=8, 32)`, 2 residual
blocks, and crucially returns `z + head(blocks(h))` with `nn.init.zeros_` on
the head's weight *and* bias. **The predictor starts as the identity.** In a
grid world almost nothing moves per step, so identity is a strong prior and
this removes the early-training phase where the model learns "stay put".

**`OutcomeHeads`** — a shared trunk `256 -> 128` GELU, then five heads:
`gem_delta`, `room_change`, `player_visible`, `death`, `moved`. The `moved`
label is **self-derived from the agent's own percept** (an identical window
after a move command means blocked), not taken from the evaluator's `moved`
flag. That is a firewall requirement, not a convenience.

**`AbstractLevel`** — a `lift` MLP, a `chunk_rnn` GRU(32 -> 64) that
summarises `chunk_len=4` primitive steps, and `step` =
`Linear(abstract_dim + chunk_dim -> 256)`. No decoder, by design.

**Anti-collapse** — VICReg variance and covariance terms against an EMA
target encoder (`ema_momentum=0.99`). The variance loss is
`F.relu(1.0 - std).mean()`, so **lower is better** and `0.0` means every
dimension has unit standard deviation. Read every `*_var` number in this
report with that sign convention in mind; it is the single easiest metric in
this project to misread.

Config defaults: `window_size=15`, `channels=64`, `pooled_dim=256`,
`abstract_dim=128`, `n_members=5`, `chunk_len=4`, `n_levels=2`. Loss weights:
`w_tile=1.0`, `w_latent=1.0`, `w_gem=2.0`, `w_room=0.5`, `w_aux=0.2`,
`w_vicreg_var=1.0`, `w_vicreg_cov=0.04`.

### 2.3 Temporal abstraction

Fixed 2/3/4 levels, not arbitrary N. Strides are 1 / 4 / 16 / 64 primitive
steps. With `terminal_chunks=4` the planner's terminal novelty term reaches
16 / 80 / 336 primitive steps ahead at depths 2 / 3 / 4 — that reach is the
entire argument for depth, and section 6.4 says why it has not paid off yet.

### 2.4 Planning (`agent/planner.py`)

CEM: `n_samples=256` action sequences of `horizon=8`, `n_iters=4` refinement
rounds, `n_elite=32`, `elite_smoothing=0.1` Laplace smoothing when refitting
the categorical logits, and `prior_weight=0.3` carrying the previous
timestep's shifted plan forward so the agent does not re-plan from scratch and
dither. Every candidate is scored **entirely on imagined latents** — the
environment is stepped once per decision, never for search.

Action space is 8 plannable actions: 4 moves and 4 camera rotations. The
rotations matter and are easy to lose — see section 9.

### 2.5 Episodic memory (`agent/memory.py`)

Count-based novelty over cells, local patterns, and pushable-block layouts.
The block layout is hashed with a **Zobrist** key, which is what pays the
agent for pushing a box somewhere it has never been — including onto a gem.
This is the mechanism behind capability 2 in section 1.

### 2.6 The observation firewall (`env/obs.py`)

`board_state_hash`, player coordinates, gem ids, `push_count`, `moved` and
`room_changed` are **evaluator-only**. The agent never reads them; they exist
so the harness can score an episode. Reading any of them invalidates an
exploration result, because the whole claim is that the agent found things out
for itself. `test_firewall_never_leaks` guards this.

**If you are a Claude session working on this: do not route an
evaluator-only field into the agent's observation, its reward, or its
planner, even to debug. If you think you need one, say so and stop.**

---

## 3. What is real and what is not

This is the most important section for interpreting any number below.

**Real**, parsed straight out of the engine repo's level files
(`env/real_levels.py`):

- room geometry, all 256 rooms
- gem placement (90 gems across 86 gem-bearing rooms)
- box placement, hazard placement, elevation
- room adjacency, from `world_map.json`

**Not real** — a reimplementation in `env/mock.py`:

- the step function. Movement, pushing, death, room transitions.

The real engine is JavaScript. It has **never been executed from an assistant
session**: Claude Code's auto-mode classifier refuses to run the cloned repo
("Code from External"). Unblocking it needs a Bash permission rule for
`node .../maze-bridge.js`.

The real game also has ice, slopes, lifts, buttons, punchers and clones.
**None of those are modelled.** `env/mock.py` implements walking, pushing,
elevation and hazards and nothing else.

> **No number in this repository is a MazeBench score.** Everything is mock
> dynamics over real geometry. Do not write "scores X on MazeBench" anywhere.

**Decided by the user on 10 Nov 2026: continue on the mock for now; the real
engine is for a later day.** Do not re-raise the permission rule unprompted.
The plan when it happens is to pretrain on mock, fine-tune the level-1
predictor on real transitions, and measure the gap by replaying identical
action sequences through both engines.

What should transfer when that day comes: the encoder (the observation format
is genuinely real), the room-change head (adjacency is real), episodic memory,
the CEM planner, the EFE weights. What will not: the level-1 predictor, the
death head, the moved head — everything that encodes dynamics.

---

## 4. Benchmark facts, measured from the engine repo

Pinned to commit `07aa03a5c0ee8b4e0b025a52793055e89cb68bfe` (23 Aug 2026).
Every number in this repo was produced against that commit.

- The world is 16x16 **rooms**; each room is 16x16 **tiles**; `camera_view`
  is 16x16 — so the ASCII view frames a whole room. An egocentric window must
  therefore be `2*16 - 1 = 31` to cover a room from any corner. **This is the
  origin of the central open problem in section 6.1.**
- `world_map.json` places all 256 levels with no duplicates, so adjacency is
  the benchmark's own.
- **The world holds 90 gems but `game_won_gem_count` is 100.** The win
  condition is unreachable in the shipped world. Report gems out of 90. This
  is consistent with published bests of ~10–13%.
- The default start room `level_HxI` holds **0 gems**, and its only exit is
  row 0, columns 6–9. 86 of 256 rooms hold a gem. **This fact wrecked the
  ablation that already ran** — section 6.3.
- Only `P`/`p` (player) and `G`/`g` (gem) are fixed glyphs. `hide_names`
  permutes everything else, so the hazard safety mask only holds under
  canonical glyphs.

---

## 5. State of play — the measured numbers

Two training runs exist. Both used `--window 15`.

| | pilot (`runs/demo`) | big (`runs/big`) |
|---|---|---|
| transitions | 7,029 | 179,139 |
| collection policy | heuristic | coverage, `--start-in-gem-rooms` |
| epochs x steps | 30 x 150 | 12 x 250 |
| gem rate in buffer | 1.55% | 0.31% |
| room-change rate | 1.12% | 1.19% |
| death rate | 2.01% | 0.03% |
| moved rate | 80.9% | 82.9% |
| **val changed-tile acc** | **0.869** (best 0.874 @ ep 28) | **0.815** (best @ ep 12 = last) |
| train changed-tile acc | 0.956 | 0.800 |
| **val gem recall** | **0.208** | **0.167** |
| val gem precision | 0.229 | 0.167 |
| train gem recall | 1.000 | 1.000 |
| train gem precision | 1.000 | 0.999 |
| L2 predictor loss | 0.227 | **0.145** |
| L2 VICReg variance (lower better) | 0.307 | **0.077** |
| val VICReg variance | 0.151 | 0.260 |

Two cells there look wrong and are not:

- **train changed-tile acc (0.800) is *below* val (0.815) in the big run.**
  Training batches are importance-sampled with `event_weights(gem_boost=40.0,
  room_boost=10.0)`, so the train stream is deliberately loaded with the rare,
  hard transitions while val is not. Do not read this as "no overfitting".
- **best epoch = last epoch (12) in the big run.** It was still improving when
  it stopped. That run is undertrained, which is a confound on every
  comparison against the pilot's 30 epochs.

The collected buffer — `runs/bigbuffer/buffer.npz`, 700 episodes, 179,139
transitions, 552 gem events, 2,123 room changes — **is committed to the repo**.
It cost an overnight run. Always train with `--buffer runs/bigbuffer/buffer.npz`
rather than collecting again.

Checkpoints are **not** committed (`*.pt` is gitignored): 18 files at ~7 MB
each, all window-15, all superseded by the run-1 baseline below. They are
regenerable from the buffer.

---

## 6. What the numbers mean — the open problems

### 6.1 The gem head overfits. It is not short of data. (the main problem)

Train gem recall **1.000**, held-out **0.167**. On 5x the gem events of the
pilot, with 40x oversampling already active, held-out recall went *down*
(0.208 -> 0.167). Collecting more data will not fix this.

Check the obvious alternative explanation and discard it: the train positive
rate is 0.103 and the val positive rate is 0.0033, a 30x prior shift. But a
classifier trained at a higher positive rate would *over*-predict positives on
val, inflating recall and destroying precision. Here recall and precision are
**both** 0.167. That is not prior shift. That is memorisation.

**Leading hypothesis: `--window 15` cannot frame a 16-tile room (section 4).**
At window 15 the gem is frequently outside the agent's view at the exact step
the head has to predict it will be collected, so the only way to fit the
training set is to memorise which specific windows preceded a gem. `--window
31` is the first test, and it is the whole reason run 1 carries that flag.

If window 31 does not move held-out recall clear of 0.21, the hypothesis is
wrong and the next suspects are, in order: the label definition (is
`gem_delta` at step t predictable from the percept at step t at all?),
regularisation on the head, and the loss weight `w_gem=2.0` interacting with
40x sampling to produce an effective 80x emphasis.

### 6.2 The room head has no recall metric

`runs/big` reports `val room = 0.00037`. The base rate is 1.2%. A head that
always predicts "no room change" would also score a tiny loss. **There is
currently no evidence at all that the room-change head works**, and the loss
figure is uninformative in exactly the way `gem_acc = 0.9993` is
uninformative.

This is the same trap as `tile_acc` (section 9). Add recall and precision for
the room head before anyone claims it works. This is a small change in
`train/train_wm.py` alongside where `gem_recall`/`gem_precision` are already
computed, and it is worth doing **before** run 1 so run 1 reports it.

### 6.3 The ablation that already ran is degenerate

`runs/arms/results.json`, 5 episodes x 120 steps, pilot checkpoint,
`--window 15`, from `level_HxI`:

| arm | unique states | novelty rate | moves | rotations | rooms visited | gems |
|---|---|---|---|---|---|---|
| random | 19.0 ± 1.5 | 0.158 | ~58 | ~62 | **1.0** | **0** |
| memoryless | 4.0 ± 0.8 | 0.033 | 3–26 | 94–117 | **1.0** | **0** |
| full | **54.2 ± 3.7** | **0.452** | 72–108 | 12–48 | **1.0** | **0** |

The headline "full explores 2.9x better than random" is real but narrow. Look
at the last two columns: **every arm visited exactly one room, scored zero
gems, pushed zero blocks and died zero times.** The only axis with any
separation is within-room state coverage.

Three reasons, all fixable:

1. **It started in `level_HxI`, which has 0 gems** (section 4). Zero gems was
   guaranteed before the run began. `play.py` has `--level` but **no
   `--start-in-gem-rooms`** — unlike `train_wm.py`, which has it. 86 room ids
   are available from `RealLevelSource.rooms_with_gems()`, e.g.
   `level_AxA`, `level_AxF`, `level_AxG`, `level_AxI`, `level_AxK`.
2. **120 steps is too short to leave a room**, whose only exit is a 4-tile gap
   in one wall.
3. **`--exploit 1.0` halved the epistemic weights** (section 2.1) in a run
   whose entire purpose was measuring exploration.

One genuinely informative result does survive: **memoryless collapses to 3–26
moves and ~117 camera rotations.** Strip episodic novelty out and the agent has
no gradient to follow, so it stands still and spins the camera. That is the
clearest evidence in the repo that the count-based term is doing real work.

Also note `no_epistemic` was listed but **never ran** — there are three arms
in that file, not four.

### 6.4 Depth has not paid off, and has not been tested fairly either

Indexed at stride 4 so all depths see the same number of chunk samples
(`runs/fair2`, `fair3`, `fair4`; pilot buffer, window 15, 5 hierarchy epochs):

| levels | l2_pred | l2_var | l3_pred | l4_pred |
|---|---|---|---|---|
| 2 | **0.168** | 0.474 | — | — |
| 3 | 0.179 | 0.465 | 0.171 | — |
| 4 | 0.186 | 0.466 | 0.190 | 0.160 |

Deeper is monotonically **worse** at level 2. But do not conclude anything yet:
`l2_var ≈ 0.47` across all three means the representations are weak
everywhere (the big run reached 0.077 on the same metric). These are 5-epoch
runs on the 7k buffer.

One real warning sign: in `fair4`, the top level has the **lowest** prediction
loss (`l4_pred = 0.160`) and the **highest** variance loss
(`l4_var = 0.590`). Predicting well while losing variance is textbook partial
collapse — the top level may be predicting well because its representation is
drifting toward constant. **Watch `l3_var` and `l4_var` in run 3.** If they sit
above ~0.4 while `l*_pred` falls, the deep levels are not learning
abstractions, they are flatlining.

### 6.5 The vision transfer path is untested

The frozen V-JEPA 2 / SigLIP 2 path in `model/vision.py` is wired up at pinned
revisions, but **no real frame has ever passed through it**.

---

## 7. The next three runs, in order

The order is not bureaucratic. Run 2 takes run 1's checkpoint as an input
file, so it physically cannot start first. Run 3 could start early but its
results are read against run 1's baseline at the same window, so starting it
first means redoing it. And run 1 is a **diagnosis** as much as a baseline: if
gem recall is still ~0.167, then the `full` arm in run 2 is planning over a
model that cannot see gems coming, and a null result between `full` and
`no_epistemic` would be uninterpretable — you could not tell whether the
epistemic term fails to help or merely reads a broken head.

In every command below, `python` means `.venv\Scripts\python.exe` (Windows)
or `.venv/bin/python` (Linux/macOS).

### Run 1 — the baseline, and the test of section 6.1

```powershell
.venv\Scripts\python.exe -u -m train.train_wm --engine mock --real-levels --buffer runs/bigbuffer/buffer.npz --window 31 --levels 2 --members 5 --epochs 30 --steps-per-epoch 250 --out runs/baseline
```

30 epochs, not 12 — section 5 showed the big run stopped while still
improving.

**What to watch, per epoch:** `val gem_rec` is the number this run exists to
move. `val changed` should climb past 0.8. Also watch `val vicreg_var`; the
big run's 0.260 against a train 0.085 is a gap worth keeping an eye on.

**How to read the result:**

- `val gem_rec` clearly above 0.21 — the window hypothesis was right. Proceed
  to runs 2 and 3 as written.
- `val gem_rec` still ~0.17 — the hypothesis is wrong. **Stop and report
  before spending GPU time on runs 2 and 3.** Go to the alternative suspects
  in section 6.1. Running the ladder on a gem-blind model wastes the hours and
  produces a number nobody can interpret.

Consider adding room-head recall/precision (section 6.2) **before** launching
this, so the run reports it. It is a small, contained change.

### Run 2 — the ablation ladder

The command in `HANDOFF.md` section 4 is **superseded**; as written it
reproduces the degenerate result in section 6.3. Fix at least the start room
and the step budget:

```powershell
.venv\Scripts\python.exe -u play.py --arms random,memoryless,no_epistemic,full --episodes 10 --max-actions 400 --level level_AxA --exploit 0.3 --checkpoint runs/baseline/final.pt --real-levels --window 31 --out runs/ladder
```

Changes from the old command and why: `--max-actions 400` so an episode can
actually leave a room; `--level level_AxA` so there is a gem to find at all;
`--exploit 0.3` so the epistemic terms are not halved in the run that
measures exploration; `--window 31` to match the checkpoint; and
`no_epistemic` is now in the arm list.

**Better option, if you are willing to make a small change:** add
`--start-in-gem-rooms` to `play.py`, mirroring `train_wm.py`, so the ladder
samples across the 86 gem rooms instead of hard-coding one. A single room is
one map's worth of luck, and with 10 episodes you cannot tell a good explorer
from a convenient floor plan. **Ask the user before adding it** — it changes
the evaluation protocol, not just a flag.

Report `rooms_visited` and `gems` alongside `unique_board_states`. If
`rooms_visited` is still 1.0 across all arms, the run is degenerate again and
the step budget is still too small — say so rather than reporting the state
coverage number alone.

### Run 3 — depth

Three runs, changing `--levels` and `--out` each time. `--index-levels 4` is
fixed across all three or the comparison is meaningless (section 9).

```powershell
.venv\Scripts\python.exe -u -m train.train_wm --engine mock --real-levels --buffer runs/bigbuffer/buffer.npz --window 31 --levels 2 --index-levels 4 --out runs/d2
```

Then the same with `--levels 3 --out runs/d3` and `--levels 4 --out runs/d4`.

**Read `l3_var` and `l4_var`, not just `l*_pred`** — section 6.4 explains why
a falling prediction loss at the top level can be a collapse signature rather
than a success.

---

## 8. What a good result would look like

So that nobody has to guess what counts as progress:

- **Run 1 succeeds** if held-out gem recall clears ~0.4 while
  `val changed_tile_acc` stays above 0.85 and `val vicreg_var` stays below
  ~0.2. Recall above 0.21 at all is already informative.
- **Run 2 succeeds** if `full` beats `no_epistemic` on rooms visited (not just
  within-room state coverage), with non-overlapping standard errors over 10
  episodes, and at least one arm collects a gem.
- **Run 3 succeeds** if any depth > 2 beats depth 2 on `l2_pred` at matched
  sample counts, with `l*_var` at the deep levels staying below ~0.2.

A clean negative is a real result here and should be reported as one. "Depth
does not help on this task at this scale" is publishable; "depth helps" with
`l4_var = 0.59` is not.

---

## 9. Traps already paid for — do not rediscover these

- **Elevation.** A `.+#+#` cell is ground at height 2, not a wall. Collapsing
  stacks to walls sealed the levels: 104 of 256 cells reachable, and no episode
  ever left its start room. Heights are parsed and a move is allowed when
  surfaces differ by <= 1, giving 244 reachable.
- **Collection policy.** Random and heuristic walks score 0 gems and 0 room
  changes on real geometry. Use `--policy coverage`, which needs **both** its
  blocked set (without it the move rate is 0.24, worse than random) **and** its
  epsilon noise — pure pathfinding emits **no camera rotations**, i.e. half the
  action set never appears in the buffer.
- **Score `changed_tile_acc`, never `tile_acc`.** "Nothing moves" scores ~99%
  on a grid world. A previous project made exactly this mistake. The same
  reasoning is why `gem_acc = 0.9993` is worthless and why section 6.2 matters.
- **VICReg variance: lower is better.** `0.0` is healthy, `0.5` is near
  collapse. It is `relu(1 - std)`, not a variance.
- **Depth comparisons need `--index-levels 4`.** Indexed at its own stride, a
  4-level model draws 2,689 chunk samples where a 2-level one draws 6,578 —
  the deep model then loses *at level 2* for being starved, not for being
  worse. Note the flag equalises **downward**: all depths get 2,689, so the
  shallow models lose data. That is the price of a fair comparison.
- **The observation firewall is the core constraint**, not a nicety. Section
  2.6.
- **Glyphs.** Only `P`/`p` and `G`/`g` are fixed; `hide_names` permutes the
  rest, so the hazard safety mask only holds under canonical glyphs.
- **A local `import` inside one branch of `main()`** in `train/train_wm.py`
  once shadowed the module-level import on *every* code path. `--collect-only`
  passed while full training died with `UnboundLocalError` after the data was
  already collected. Fixed, but the shape of the bug is worth remembering:
  Python function-scope import shadowing is invisible until the branch that
  does not contain it runs.
- **Do not use pytest.** `requirements.txt` does not install it. The suite has
  its own runner.

---

## 10. Setup, condensed

Full version with explanations in `HANDOFF.md` section 1. Deliberately avoids
activating the venv, because activation needs a relaxed PowerShell execution
policy and is the most common place this gets stuck.

```powershell
git clone https://github.com/fatehakhribalgebra-source/mazebench-efe.git
cd mazebench-efe
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
git clone https://github.com/mazebench/MazeBenchEngine.git
git -C MazeBenchEngine checkout 07aa03a5c0ee8b4e0b025a52793055e89cb68bfe
$env:MAZEBENCH_REPO = "$PWD\MazeBenchEngine"
.venv\Scripts\python.exe tests\test_all.py
```

Expect `47/47 passed`.

Three things that bite:

1. Plain `pip install torch` gives a **CPU-only** build on Windows, silently.
   Hence the explicit `--index-url`. The `cuda.is_available()` line must print
   `True`.
2. `$env:MAZEBENCH_REPO` lasts for **one terminal window**. Set it again in any
   new one, or pass `--repo-root` explicitly.
3. The test suite drives the mock engine, so **a green suite does not prove
   `MAZEBENCH_REPO` is set correctly** — the real-levels tests return early
   when the clone is missing. Run 1 is the first thing that actually exercises
   the real geometry.

Known-good: torch 2.5.1+cu121, numpy 2.3.2, CUDA 12.1.

---

## 11. Rules of engagement for a Claude session on this project

1. **Never route an evaluator-only field** into the agent, its objective or
   its planner. Section 2.6. If you think you need one, stop and say so.
2. **Never call any number here a MazeBench score.** Section 3.
3. **Never report an accuracy on a rare event without recall and precision.**
   Section 9.
4. **Do not raise the real-engine permission rule unprompted.** The user
   decided on 10 Nov to continue on the mock. They know the dynamics are
   approximate.
5. **Train from `--buffer runs/bigbuffer/buffer.npz`.** Collection cost an
   overnight run.
6. **If run 1 does not move gem recall, stop and report** rather than
   proceeding to runs 2 and 3. Section 7.
7. Checkpoints are gitignored and large; do not try to commit them. The
   buffer is the exception that is already committed.
8. `report.html` in this repo is the architecture write-up with figures, if a
   visual explanation is wanted.

---

## 12. Repo map

| path | what |
|---|---|
| `env/obs.py` | observation construction and the firewall |
| `env/real_levels.py` | parses real level files; `RealLevelSource`, `rooms_with_gems()` |
| `env/mock.py` | the reimplemented step function — the sim-to-real gap lives here |
| `env/bridge.py` | the real-engine bridge (never executed from a session) |
| `env/actions.py` | 8 plannable actions: 4 moves, 4 rotations |
| `model/world_model.py` | encoder, level-1 predictor, outcome heads, abstract levels |
| `model/vision.py` | frozen V-JEPA 2 / SigLIP 2 path (untested) |
| `agent/efe.py` | the EFE objective and its weights |
| `agent/planner.py` | CEM |
| `agent/memory.py` | count-based novelty, Zobrist block hashing |
| `train/train_wm.py` | training entry point |
| `train/dataset.py` | `event_weights` oversampling |
| `train/collect.py` | collection policies |
| `play.py` | evaluation and the ablation arms |
| `viz/render.py` | replay HTML |
| `tests/test_all.py` | 47 tests, own runner, no pytest |
| `runs/bigbuffer/buffer.npz` | the committed 179k-transition buffer |
| `report.html` | architecture write-up with figures |
| `HANDOFF.md` | machine setup |

This document supersedes `HANDOFF.md` sections 3, 4 and 5 where they differ —
in particular the run-2 command, which is corrected in section 7 above.
