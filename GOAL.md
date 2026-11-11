# What we are doing right now

**Read this first. It overrides `REPORT.md` and `HANDOFF.md` wherever they
disagree.** Updated 11 Nov 2026.

---

## The goal, in one sentence

**Reproduce the published JEPA world-model result on MazeBench: learn a world
model from 3D frames, then reach a goal screenshot by planning with CEM.**

That is the whole job. Not exploration, not gems, not expected free energy.
Those were the plan and they are still the plan *eventually* — but none of them
are worth building on a world model that cannot predict its own next frame.

## The bar

From *What drives success in physical planning with JEPA world models*
([arXiv 2512.24497](https://arxiv.org/abs/2512.24497)), Table 2. Goal-reaching
success rate, CEM planning with an L2 latent cost:

| Environment | DINO-WM | JEPA-WM |
| --- | ---: | ---: |
| **PointMaze** | 81.6% | 83.9% |
| **Wall** | 64.1% | 78.8% |
| Push-T | 66.0% | 70.2% |

MazeBench is navigation, so PointMaze and Wall are the analogues.
**Target: ~80% success on single-room goal screenshots.** Below ~60% the model
is not good enough to build anything on top of.

## What that means concretely

- Frozen **DINOv2 ViT-S/14 @224** encoder, never fine-tuned. It is 1.7B images
  of prior that comes free.
- A learned **action-conditioned predictor** on top. This is the only thing
  that learns MazeBench.
- **Three levels**, strides 1 / 4 / 16, all predicting in the same patch-latent
  space.
- **Four actions: up, down, left, right.** Nothing else.
- Evaluation: hand the agent a goal screenshot, let it plan, count arrivals.

## Explicitly out of scope until the above works

Do not build, extend or argue about any of these yet. They each depend on a
world model that works, and we do not have evidence of one.

- expected free energy, curiosity, ambiguity, information gain
- gems, rooms, scoring, the win condition
- the learned temporal distance `d(s,g)`
- subgoal proposal and hierarchical EFE planning
- ASCII or JSON observation modes
- camera rotation, undo, reset, level jumps

The observation firewall still applies throughout: privileged fields
(`board_state_hash`, player coordinates, gem ids, `push_count`, `moved`,
`room_changed`, and the renderer's `_render_state`) never reach the model.

## Order of work

| Step | What | Gate |
| --- | --- | --- |
| **0** | Collect frames. See `PHASE0.md`. | 50k frames + actions on disk, replay reproduces them |
| **1** | `train/frames.py` encodes frames to latents once | latents on disk, one `.npz` per episode |
| **2** | `python -m train.train_hjepa --latents … --levels 3` | **every level clearly beats the identity baseline** (ratio well under 1.0) |
| **3** | Goal-reaching success rate | **~80%**, measured against the table above |

Step 2's gate is the one people skip. The trainer prints the identity baseline
— the loss from predicting that nothing changes — beside every validation
number. On a grid world where one move alters a small part of the frame,
identity is a strong predictor. A model that does not clearly beat it has
learned no dynamics, however small its absolute loss looks.

Run the flat baseline too (`--levels 1`). That is the published DINO-WM
configuration, and the comparison anyone reading this will ask for. If three
levels do not beat one, say so — that is a real result.

## Before anything else

```
python tests/test_hjepa.py
```

Builds a toy grid world, trains on it, and plans to goals. Under a minute on
CPU. If it passes, the pipeline is sound and any later failure is data or
scale, not code. It already caught one real planner bug this way.

## What is superseded

`REPORT.md` is still the right description of the *project's* history, traps
and benchmark facts, and its "traps already paid for" section is all still
true. But these parts of it are now wrong:

- the three queued runs (`--window 31` baseline, ablation ladder, depth sweep)
- anything about ASCII observations, `GridEncoder`, or the 179k buffer in
  `runs/bigbuffer/` — that buffer cannot train a vision model
- "no pretrained model is worth downloading", which was answering the wrong
  question; `facebookresearch/jepa-wms` is exactly the right thing

**Do not run the `--window 31` baseline.** There is no window in vision mode.
