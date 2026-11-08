# MazeBench agent: hierarchical latent world model + Expected Free Energy planning

An agent that plays [MazeBench](https://github.com/mazebench/MazeBenchEngine) by
imagining action sequences inside a learned latent world model and choosing the
one that minimises **expected free energy** — so exploration and gem-collection
come out of a single objective instead of a hand-tuned reward mix.

```
observation (ASCII tiles)
   │
   ├─► GridEncoder ─────────────► z  (B, 64, 15, 15)   latent, one vector per tile
   │                              └► h  (B, 256)        pooled
   │
   ├─► EpisodicMemory            cell counts · pattern counts · room graph · abstract coverage
   │
   └─► CEM planner ── for each of 256 candidate action sequences:
           level 1:  z, a → z'   ×5 ensemble   (disagreement = epistemic value)
                     z' → tile logits          (imagination you can read)
                     z' → gem / room / death / moved heads
           level 2:  h2, chunk(4 actions) → h2'   coarse horizon extension
           score:    G = Σ γᵗ [ risk + ambiguity − epistemic ]  → minimise
                                                  │
                                      execute first action, replan
```

## Expected free energy, concretely

For a candidate action sequence π,

```
G(π) = Σ_t γᵗ [ risk_t  +  ambiguity_t  −  epistemic_t ]
```

| Term | Meaning | How it is estimated here |
|---|---|---|
| **risk** | `−ln P(o_t \| C)`: how far expected observations fall from preferred ones | negative utility: expected gems collected, dense proximity to a visible gem glyph, room-change probability, minus death probability and hazard proximity |
| **ambiguity** | `E_q[H[P(o\|s)]]`: expected observation entropy, **penalised** | mean per-tile predictive entropy of the tile head |
| **epistemic** | expected information gain, **rewarded** | *parameters*: disagreement across the 5-member predictor ensemble. *state*: count-based novelty over dead-reckoned cells and egocentric patterns from episodic memory |
| **terminal** | long-horizon novelty past the primitive horizon | distance of the level-2 latent to the nearest visited abstract state |

Two details matter more than they look.

**Gems alone cannot steer an 8-step horizon.** Gem events are ~2% of
transitions. A pure `Δgems` preference is flat almost everywhere, so the
pragmatic term also carries a *dense* proximity signal read off the imagined
board: how close the imagined player is to a gem glyph it can see. That is what
turns "prefer gems" into something a short-horizon planner can act on.

**Ambiguity is what stops curiosity degenerating.** An agent rewarded purely
for information gain seeks noise. Penalising expected observation entropy means
that between two equally informative states it takes the one whose consequences
it can actually predict.

`--exploit 0` zeroes the gem weights and runs the reward-free exploration
protocol — the condition the research question is actually about — without
changing anything else.

## The observation firewall

This is the part to check first, because everything else depends on it.

`maze-bridge.js` returns the full authoritative snapshot. MazeBench deliberately
withholds some of it from the model: `board_state_hash`, `player` coordinates,
`collected_gems` ids, `push_count`, `novel_push_count`, the scorecard, and the
`moved` / `room_changed` flags. Those are **evaluator** fields. An exploration
result obtained while reading them is worthless, because the agent was handed
the state it was supposed to infer.

So `MazeBenchEnv.step()` splits every snapshot in two:

* `obs` — only [`AGENT_VISIBLE_FIELDS`](env/obs.py). All the agent ever sees.
* `info` — the privileged remainder, for logging and scoring only.

`test_firewall_never_leaks` asserts the split. Consequences the agent lives with:

| Wanted | Not allowed | What it does instead |
|---|---|---|
| state novelty | `board_state_hash` | hashes its own ASCII percept and dead-reckons cells |
| "did I move?" | `moved` flag | compares consecutive egocentric windows — identical means blocked |
| position | `player.x/y` | integrates its own intended moves, resets the frame on room change |
| box progress | `novel_push_count` | tracks pushable glyphs in its own window |

Only `P`/`p` and `G`/`g` are fixed glyphs in the engine — `hide_names` permutes
everything else per rollout. So gem-seeking and self-location survive glyph
permutation, while terrain and box identity have to be learned. The one place
that assumption is hard-coded is the safety mask, which must be disabled under
`hide_names` (`AgentConfig.safety_mask=False`).

## Why this shape of world model

The latent is a **grid**, `(C, S, S)`, at full window resolution — one feature
vector per visible tile. MazeBench is tiles, so tile-level latents are the right
inductive bias. Three things follow:

1. The one-step predictor is **residual**. Almost nothing changes in a grid
   world when you take one step, so predicting a delta is far easier than
   predicting the next grid. Its output head is zero-initialised, so an
   untrained model imagines "nothing happens" rather than injecting noise.
2. A 1×1 conv **tile head** decodes `z` back to glyph logits. Imagination is
   therefore *readable*: the replay viewer renders every imagined step as ASCII,
   so "imagine the box pushed onto the gem" is something you can print and
   check. It also makes level-1 collapse impossible — a collapsed latent cannot
   reconstruct the board.
3. **Level 2** runs on pooled latents over chunks of 4 actions, JEPA-style with
   an EMA target and VICReg, no decoder. At that timescale the exact board is
   not predictable and should not be. It buys the planner 16 extra steps of
   lookahead for 4 calls instead of 16.

### This is not a pure JEPA, and the name matters

A pure JEPA predicts only in representation space and never reconstructs.
Level 1 here is a latent predictive model **with** a reconstruction head. That
is a deliberate trade: reconstruction is what gives the EFE planner groundable,
inspectable outcomes on a tile world, and it removes collapse as a risk rather
than merely discouraging it. Level 2 is JEPA-style. Calling the whole thing
"H-JEPA" would overclaim, so the class is `HJEPA` for continuity with the
earlier work but the docstring says exactly what each level is.

### Transfer learning

The tile encoder is the default, and it is the right choice for a tile world: it
trains in minutes and a frozen image model cannot beat direct glyph access. So
the transfer path in [`model/vision.py`](model/vision.py) exists to answer a
*different* question — can the agent still work from pixels, where MazeBench's
`vision` observation mode gives a perspective PNG and the same win condition.

`FrozenVisionEncoder` keeps `GridEncoder`'s exact output contract, `(z, h)` with
`z` of shape `(B, C, S, S)`: patch tokens → native spatial grid → bilinear
resample to `S×S` → 1×1 projection to `C`. So predictor, planner and EFE are all
unchanged. Both backbones stay **frozen** and only the adapter trains (~0.1M
params against 300M frozen) — at a few thousand transitions, fine-tuning a
ViT-L would just memorise the buffer, and the earlier run of this project found
frozen-backbone heads to be the only arm that trained stably at this scale.

Revisions are pinned to the two checkpoints your earlier package validated, so
results stay comparable:

| Backbone | Checkpoint | Revision |
|---|---|---|
| V-JEPA 2 | `facebook/vjepa2-vitl-fpc64-256` | `b3c1679…` |
| SigLIP 2 | `google/siglip2-base-patch16-224` | `75de2d5…` |

`HybridEncoder` fuses tile and visual features, which is the honest way to
measure what the backbone *adds* rather than whether it suffices alone.

One real cost: in vision mode there is no ASCII board to supervise the tile
head, so it is switched off (`w_tile=0`) and the latent is trained by the JEPA
term and the outcome heads alone. Imagination stops being renderable as ASCII —
the interpretability in the replay viewer is specific to the tile path.

Needs `transformers` + `pillow`, and Playwright for the vision observation mode.
Not in the default pipeline.

## Action space, and what is deliberately not in it

**Plannable (8)** — 4 moves, 4 camera rotations. The planner searches only these.

Camera rotation is in on purpose: it cannot change world state, so it has zero
pragmatic value, but it *does* reveal occluded tiles. That makes it the cleanest
available test of whether the epistemic term is doing real work — an agent that
never rotates is ignoring free information; one that only rotates has collapsed
into novelty-seeking.

**Control (supervisor, never searched)** — `undo`, `reset_level`, `goto_level`,
`quit`. `undo`/`reset` depend on history a Markov latent cannot carry, and
letting the planner treat `reset` as an ordinary move invites a reset loop that
farms "novel" states. This is also what the earlier run of this project
concluded on its own: reset is an episode-control decision for a stuck
detector, not a movement.

The supervisor handles three things: death (undo, else reset — gem score is
monotonic so recovery never costs progress), room exhaustion (when the room
stops yielding new patterns the local EFE gradient is flat, so jump to the best
frontier room), and the action budget.

## Window sizing

Counted from `games/maze/levels/*.txt`: **all 256 shipped levels are exactly
16×16 tiles.** (Those files are space-separated cell tokens with `+#` stacking
for elevation, so raw character width is misleading — count the fields.)

That fixes the trade-off on the egocentric window:

| `--window` | Covers a 16×16 room | Relative planner cost |
|---|---|---|
| 15 (default) | only when the player is near the middle | 1× |
| 21 | most positions | ~2× |
| **31** | **always, from any corner** (2·16−1) | ~4.3× |

15 is the default because it keeps local iteration fast. Use `--window 31` for
real runs on a real GPU — it is the only size that guarantees the whole room is
in view no matter where the agent stands. The mock's rooms are 16×16 to match,
so coverage behaves the same in both.

## Where things are

| Path | What it holds |
|---|---|
| [`env/glyphs.py`](env/glyphs.py) | the tile vocabulary, transcribed from the engine's own `maze-observation-contract.js` |
| [`env/obs.py`](env/obs.py) | ASCII → egocentric window + public scalars; `AGENT_VISIBLE_FIELDS` |
| [`env/bridge.py`](env/bridge.py) | persistent `maze-bridge.js` subprocess, Node discovery |
| [`env/mazebench_env.py`](env/mazebench_env.py) | **the observation firewall**, `EvalLog`, death recovery |
| [`env/mock.py`](env/mock.py) | Python stand-in for the engine. Not the benchmark |
| [`env/actions.py`](env/actions.py) | 8 plannable primitives vs supervisor-only control actions |
| [`model/world_model.py`](model/world_model.py) | encoder, predictor ensemble, tile head, outcome heads, level 2, losses |
| [`model/vision.py`](model/vision.py) | frozen V-JEPA 2 / SigLIP 2 transfer path |
| [`agent/memory.py`](agent/memory.py) | cell/pattern/box counts, room graph, dead reckoning, frontier scores |
| [`agent/efe.py`](agent/efe.py) | the expected-free-energy terms |
| [`agent/planner.py`](agent/planner.py) | CEM in latent space + hierarchical terminal value |
| [`agent/agent.py`](agent/agent.py) | the loop, the supervisor, the safety mask, the random baseline |
| [`train/`](train/) | collection, datasets with rare-event oversampling, the 3-phase trainer |
| [`viz/render.py`](viz/render.py) | self-contained HTML replay viewer |
| [`tests/test_all.py`](tests/test_all.py) | 31 tests, no Node and no GPU needed |

## Install and run

Needs Python 3.10+, PyTorch, NumPy. For the real engine, Node 18+.

```bash
python tests/test_all.py
```

```bash
python -m train.train_wm --engine mock --out runs/dev --episodes 28 --epochs 30
```

```bash
python play.py --checkpoint runs/dev/final.pt --engine mock --episodes 3
```

That writes `runs/play/replay.html` — open it to step through the rollout and
see the imagined rollout and EFE breakdown beside the real board.

### Against the real engine

```bash
git clone https://github.com/mazebench/MazeBenchEngine.git
export MAZEBENCH_REPO=/path/to/MazeBenchEngine
export MAZEBENCH_NODE=/path/to/node        # optional if node is on PATH
python -m train.train_wm --engine real --out runs/real --episodes 32
python play.py --checkpoint runs/real/final.pt --engine real --episodes 5
```

`env/bridge.py` speaks `scripts/maze-bridge.js` directly — one persistent
subprocess per rollout, one JSON object per line. No browser, no Prime account,
no `npm ci` (the ASCII path's only npm dependency, `three`, is for the 3D
renderer). It also exposes `restore_checkpoint`, so you can snapshot the
authoritative engine and rewind it to compare imagined against real outcomes.

## How deep should the hierarchy be?

A fixed small stack, as H-JEPA is drawn — not an arbitrary depth. `--levels`
takes **2, 3 or 4**, counting the primitive level, so "does another level
help?" is a three-way experiment rather than a free parameter.

| `--levels` | abstract levels | strides (primitive actions) | params | terminal lookahead |
|---|---|---|---|---|
| 2 | level 2 | 4 | 1.68M | 16 steps |
| 3 | + level 3 | 4, 16 | 1.82M | 80 steps |
| 4 | + level 4 | 4, 16, 64 | 1.95M | 336 steps |

That is the case for depth: with `terminal_chunks=4`, lookahead past the
planning horizon grows **geometrically** while parameters and per-decision work
grow linearly. Each level conditions on the primitive actions inside its own
span, so no level has to interpret the one below's chunk encoding, and each
scores novelty against its own coverage store — a coarse level cannot shout
down a fine one just by reaching further.

### Run the comparison fairly, or don't run it

```bash
for L in 2 3 4; do
  python -m train.train_wm --real-levels --levels $L --index-levels 4 --out runs/lv$L
done
```

**`--index-levels 4` is not optional here.** A chunk sample only exists where
the longest level's whole span is valid, so indexed at its own depth a 4-level
model gets 2,689 samples from a buffer that gives a 2-level model 6,578. In a
smoke run that showed up exactly as you'd expect — the deep model's *level 2*
came out worse than the shallow model's, which is data starvation wearing the
costume of a depth result. Pinning the index to the deepest arm gives every arm
an identical sample set.

With that fixed, a short run gave:

| | level 2 | level 3 | level 4 |
|---|---|---|---|
| `--levels 2` | 0.168 | — | — |
| `--levels 3` | 0.179 | 0.171 | — |
| `--levels 4` | 0.186 | 0.190 | 0.160 |

Two warnings about reading that table. **Losses at different levels are not
comparable** — each predicts in its own space, and coarser spaces are smoother,
which is why level 4 posts the lowest number while predicting the vaguest
thing. Only same-level-across-depth columns mean anything. And these are
5-epoch smoke runs: the only verdict that counts is downstream agent behaviour
on the ablation ladder below, not prediction loss.

## Collecting data that contains anything to learn from

Real geometry broke the obvious collection policies, in two ways that both
produced a buffer with **zero gem events and zero room changes**. Both are
worth knowing before trusting any buffer.

### 1. Collapsing elevation sealed the levels

MazeBench is 3D. A `.+#+#` cell is ground at height 2, not a wall. The first
parser called every `#` stack a wall, and the levels closed up: in `level_HxI`
only 104 of 256 cells were reachable from the spawn, the single door at row 0
was behind a wall, and **no episode ever left the starting room**.

Parsing elevation as a height map and allowing a step when surfaces differ by
at most 1 took reachability from **104 to 244 cells**, with the borders
reachable. Walls now emerge from the terrain instead of being asserted. A cell
too high or low to step onto renders as an obstacle, which is roughly what a
player can infer, and is what keeps a path on the board a path in actions.

### 2. Random walks never find the door

Even with the levels open, `level_HxI`'s only walkable border cells are row 0,
columns 6–9 — one narrow door. A random walk from the spawn at (12,4) stays in
rows 9–14 and never reaches it. Over 6 episodes of 256 actions: 0 gems,
0 room changes. That is realistic, and it is why random scores ~0 on this
benchmark, but it makes random collection worthless.

`--policy coverage` runs BFS over the rendered board. **This is not
privileged**: `camera_view` is 16×16, the same size as a room, so the agent
already sees the whole board, and movement is screen-relative so a path on the
board is a path in actions. It targets a visible gem, then an unvisited cell,
then a door.

Two fixes it needed, both found by measurement:

* **A blocked set.** A ledge only renders as an obstacle relative to where the
  player stands, so BFS is optimistic about cells two steps away. Without
  remembering refusals, the move rate collapsed to **0.24**, worse than
  random's 0.58. With it: **0.83**.
* **Epsilon noise over all 8 actions.** Pathfinding only ever emits the four
  moves, so the buffer contained **no camera rotations at all** — half the
  plannable action set, which the world model would never learn and the
  planner would still propose.

### 3. Gems are genuinely sparse, so start where they are

90 gems across 256 rooms, and an episode visits 2–4 rooms. `--start-in-gem-rooms`
spreads collection starts over the 86 rooms that hold at least one gem.
Collection only — evaluation still starts where the benchmark says.

Measured effect of the three together, per 256-action episode on real geometry:

| policy | gem rate | room changes | move rate | rotations |
|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.58 | yes |
| heuristic | 0.000 | 0.000 | 0.78 | yes |
| coverage, default start | 0.000 | 0.004 | 0.24 | **none** |
| **coverage + fixes + gem rooms** | **0.0024** | 0.010 | **0.83** | yes |

A 0.24% gem rate is still thin. That is the benchmark, not the collector —
90 gems in 256 rooms is intrinsically sparse, which is exactly why gem-seeking
needs a world model and not a reward signal. The answer is volume.

### The collected buffer

```bash
python -m train.train_wm --real-levels --start-in-gem-rooms --collect-only \
    --policy coverage --episodes 700 --max-actions 256 --out runs/bigbuffer
```

700 episodes, **179,139 transitions**, 3.8 MB:

| | count | rate | vs the 7k buffer |
|---|---:|---:|---:|
| gem events | 552 | 0.308% | **5.1×** |
| room changes | 2,123 | 1.185% | **26.9×** |
| successful moves | 148,592 | 82.9% | — |
| deaths | 62 | 0.035% | — |
| chunk samples (index depth 4) | 132,776 | — | — |

Train from it with `--buffer runs/bigbuffer/buffer.npz`, which skips collection
entirely.

## The ablation ladder — the thing that answers the question

"Beats uniform random" is not a result. Random only shows the planner does
something; it says nothing about *which part* is doing the work. The research
question is whether a world model **with persistent episodic memory** beats a
**memoryless** one at reward-free exploration, so the memoryless arm is the
comparison, and random is just the floor.

| arm | preferences | ambiguity | disagreement | episodic memory | frontier jump |
|---|:-:|:-:|:-:|:-:|:-:|
| `random` | — | — | — | — | — |
| `pragmatic_only` | ✓ | — | — | — | — |
| `no_epistemic` | ✓ | ✓ | — | — | — |
| `memoryless` | ✓ | ✓ | ✓ | — | — |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ |

```bash
python play.py --checkpoint runs/dev/final.pt --real-levels \
    --arms random,pragmatic_only,no_epistemic,memoryless,full --episodes 10
```

Every arm runs the same seeds, action budget and observation mode, and the
table reports mean ± standard error. `--exploit 0` zeroes the gem preference
across all arms for the reward-free protocol.

Note what `memoryless` keeps: **ensemble disagreement stays on**. Model
uncertainty is not episodic memory, and removing both at once would conflate
curiosity with memory — the isolation is the whole point. `pragmatic_only`
removes EFE entirely and is plain reward maximisation through the world model,
which is the comparison that says whether expected free energy earned its
complexity.

## Real levels without running the engine

The engine is JavaScript and needs a Node subprocess. The *levels* are plain
text in a documented token format, so [`env/real_levels.py`](env/real_levels.py)
parses them directly — real room geometry, real gem and box placement, nothing
executed:

```bash
python -m train.train_wm --engine mock --real-levels --out runs/real_geom
```

`world_map.json` places every level on the world grid, so room adjacency here
is MazeBench's own — walking east from `level_HxI` arrives where it really
would. From `world_parsing.json`:

```
world_size:  [16, 16]    a 16x16 grid of rooms = 256 rooms
level_size:  [16, 16]    each room is 16x16 tiles
camera_view: [16, 16]    the ASCII view frames a whole room
```

What the placed world actually contains:

| | |
|---|---|
| rooms | 256, all placed, no duplicates, none missing |
| dimensions | **16×16, every single one** |
| boxes | 2,155 (~8.4 per room) |
| **gems in the entire world** | **90** |
| explicit player spawn | 236 of 256 |
| gems in the default start room `level_HxI` | **0** |

### The win condition is not reachable in the packaged world

`config.json` sets `game_won_gem_count: 100`. The whole placed world contains
**90** gems. Counting `old/` and `other/` as well only reaches 103, and those
are not placed on the world map.

So no agent can "win" this world — the benchmark is effectively a gem-count
race, which fits the reported best scores of ~10–13% being ~10–13 gems out of
90. Unless gems respawn, or clones collect them separately, or the hosted world
differs from the packaged one, treat "100 gems" as unreachable and report
*gems out of 90*. Worth confirming against the live engine before it goes in a
proposal.

That the start room holds **zero** gems is the other operative fact: the agent
cannot score at all without leaving it, which is exactly the regime the
epistemic half of EFE exists for.

**This is a 2D projection of a 3D world.** MazeBench has elevation, ice
physics, slopes, lifts, buttons and clones; here a cell collapses to
floor/wall/hole/gem/box by reading the top of its stack. So geometry and object
placement are real, **dynamics are still the mock's**. One parser bug worth
recording: void cells were first mapped to lethal holes, which killed a random
walk within a few steps — `level_parsing.json` has no hole token at all, so a
floorless cell is simply out of bounds, and a 16×16 level is mostly that.

## The mock is not the benchmark

`env/mock.py` is a small multi-room Sokoban-with-gems world that speaks the same
command surface and emits the same snapshot fields. It exists so the model,
planner and tests run with no Node and no browser, and so phase-0 iteration is
fast. **Numbers from it say nothing about MazeBench performance.** It reproduces
the glyph contract, limited visibility, yaw/pitch rendering, persistent
cross-room state, box pushing, holes and room transitions. It does not reproduce
ice, slopes, clones, punchers, lifts, buttons, or the real world's 200+
hand-built rooms.

## Honest status

All 31 tests pass. On the mock, 28 episodes × 256 actions = 7,029 transitions,
30 epochs:

| | epoch 1 | epoch 10 | epoch 30 |
|---|---|---|---|
| `changed_tile_acc` (held out) | 0.283 | 0.791 | **0.869** |
| `tile_acc` (held out) | 0.867 | 0.954 | 0.970 |
| gem recall (held out) | 0.062 | 0.229 | 0.208 |

Level 2 converges to a prediction loss of 0.227 with a VICReg variance term of
0.31 — well clear of the ~1.0 a collapsed representation would show.

Read the first row, not the second. `tile_acc` is 0.87 at epoch *one*, because
a model that predicts "nothing moves" is right about almost every tile on a
grid world. `changed_tile_acc` — accuracy restricted to tiles that actually
changed — is the number that moved from 0.28 to 0.87. This project made exactly
that mistake before: the earlier run's own notes record whole-board error
improving 26% while the changed-pixel error got *worse*.

Known problems, all real:

* **The gem head is the weak point.** Held-out recall tops out near 0.21 at a
  1.6% base rate. With a 6-episode buffer it was 0.00 against a training recall
  of 1.00; 10× the data moved it to 0.21, so it is a data-volume problem and
  not a broken head. But the oversampler makes gems ~50% of training batches
  against the 1.6% true rate, so its probabilities are **not calibrated** —
  they rank candidate plans, where that matters less, and should not be read as
  "probability a gem is collected".
* **A learned death head is not enough.** Relying on it at a 2% base rate, the
  agent died *more often than uniform random* (8.5 vs 6.5). That is why the
  hard safety mask exists. The mask reads hazard glyphs straight from the
  agent's own board, so it only works under canonical glyphs — under
  `hide_names` the hazard has to be learned and this regression can return.
* **Room exploration is under-tested.** The frontier-jump supervisor and the
  room-change preference fire rarely in short mock episodes. The mock's rooms
  are also procedurally generated and far easier to leave than MazeBench's.
* **Never run against the real engine.** Every number here is from the mock,
  which is not the benchmark. Nothing in this README is a MazeBench result.

The research question, metrics and protocol are inherited from
`MAZEBENCH-RESEARCH-PIVOT.md`: identical starts, observation mode, action budget
and seeds; at minimum random / memoryless / memory-enabled arms; multiple runs
with uncertainty reported. One rollout is a demo, not evidence.
