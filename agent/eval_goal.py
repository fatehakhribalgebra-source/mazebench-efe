"""Goal-reaching success rate: the number `GOAL.md` exists to produce.

The protocol the papers use, and the one this reproduces: hand the agent a goal
**screenshot**, let it plan with the learned world model, and count how often it
actually arrives. DINO-WM reports 81.6% on PointMaze and 64.1% on Wall; jepa-wms
reports 83.9% and 78.8%. Target here is ~80%, and below ~60% the model is not
worth building on.

    python -m agent.eval_goal --checkpoint runs/hjepa3/final.pt \
        --frames runs/vision/frames --latents runs/vision/latents --n-goals 100

**Two separations that make this measurement honest.**

1. *The planner sees pixels only.* It receives the current frame's latents and
   the goal frame's latents. Nothing else -- no coordinates, no room id, no
   distance-to-goal, no privileged field.

2. *The evaluator judges arrival from privileged state.* Success is "the player
   is standing where the goal screenshot shows it", decided from the engine's own
   actor positions. It deliberately is **not** the planner's latent cost: that
   would be scoring the model against its own objective, which can be driven to
   zero by a model that has learned nothing about the world. Everything read
   from ``privileged.json`` lives in variables prefixed ``eval_`` and never
   touches the model.

Goals are sampled **within a single room**, per `GOAL.md`, from held-out
episodes, at a known action distance -- so every goal is reachable by
construction, and a failure is the planner's, not the task's.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from agent.goal_cem import GoalCEM, GoalCEMConfig
from env.bridge import BridgeConfig, BridgeError, MazeBridge
from env.renderer import FrameRenderer, RendererConfig
from model.hjepa import HJEPA, HJEPAConfig, FrozenEncoder, goal_cost
from train.frames import preprocess, split_episodes

MOVES = ("up", "down", "left", "right")


def to_array(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


class Task:
    """One goal-reaching problem drawn from a held-out episode."""

    def __init__(self, ep_dir: Path, start_step: int, goal_step: int,
                 eval_goal_pos: tuple, eval_goal_room: str):
        self.ep_dir = ep_dir
        self.start_step = start_step
        self.goal_step = goal_step
        # Evaluator-only. Used to judge arrival; never shown to the planner.
        self.eval_goal_pos = eval_goal_pos
        self.eval_goal_room = eval_goal_room

    @property
    def distance(self) -> int:
        return self.goal_step - self.start_step


def sample_tasks(frames_root: Path, episodes: list[str], horizon: int,
                 per_episode: int, rng: random.Random) -> list[Task]:
    """Single-room goals at a fixed action distance, from held-out episodes."""
    tasks: list[Task] = []
    for name in episodes:
        ep_dir = frames_root / name.replace(".npz", "")
        meta_path = ep_dir / "episode.json"
        priv_path = ep_dir / "privileged.json"
        if not (meta_path.is_file() and priv_path.is_file()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        eval_steps = json.loads(priv_path.read_text(encoding="utf8"))["steps"]
        n = meta["n_frames"]

        candidates = []
        for t in range(0, n - horizon):
            a, b = eval_steps[t], eval_steps[t + horizon]
            # Same room (GOAL.md: single-room goals), and the player actually
            # went somewhere, so the goal is not the start.
            if a.get("level_id") != b.get("level_id"):
                continue
            pa = (a.get("player_x"), a.get("player_y"))
            pb = (b.get("player_x"), b.get("player_y"))
            if None in pa or None in pb or pa == pb:
                continue
            candidates.append((t, pb, str(b.get("level_id") or "")))
        rng.shuffle(candidates)
        for t, pb, room in candidates[:per_episode]:
            tasks.append(Task(ep_dir, t, t + horizon, pb, room))
    return tasks


def run_task(task: Task, model: HJEPA, encoder: FrozenEncoder,
             cem_config: GoalCEMConfig, device: str, budget: int,
             size: int) -> dict:
    """Replay to the start state, then plan to the goal screenshot."""
    meta = json.loads((task.ep_dir / "episode.json").read_text(encoding="utf8"))
    actions = meta["actions"]
    level_id = meta["start_level"]
    yaw = int(meta["yaw"])
    view = meta["view"]

    frame_paths = sorted((task.ep_dir / "frames").glob("*.png"))
    goal_png = frame_paths[task.goal_step].read_bytes()

    @torch.no_grad()
    def encode(frame: np.ndarray) -> torch.Tensor:
        batch = preprocess(frame[None], 224).to(device)
        return encoder(batch).float()[0]

    goal_latent = encode(to_array(goal_png))

    renderer = FrameRenderer(RendererConfig(
        level_id=level_id, width=size, height=size, view=view, yaw=yaw))
    bridge = MazeBridge(BridgeConfig(level_id=level_id, yaw=yaw,
                                     observation_mode="text"))
    planner = GoalCEM(model, cem_config, device=device)
    planner.reset()

    reached_at = None
    closest_cost = float("inf")
    eval_trace = []
    try:
        renderer.init()
        obs = bridge.observe()
        # Replay to the start state. The bridge is bit-exact, so this lands
        # exactly where the collector was.
        for action in actions[: task.start_step]:
            obs = bridge.move(MOVES[action])

        png = renderer.render(obs["_render_state"])
        latent = encode(to_array(png))
        # The model wants W context frames and we have one: repeat it, which is
        # the standard cold-start and is what a human gets from one screenshot.
        context = latent.unsqueeze(0).expand(model.config.context, -1, -1).clone()

        for step in range(budget):
            eval_pos = _eval_position(obs)
            eval_room = str((obs.get("_render_state") or {}).get("level_id") or "")
            eval_trace.append({"pos": eval_pos, "room": eval_room})
            if eval_pos == task.eval_goal_pos and eval_room == task.eval_goal_room:
                reached_at = step
                break

            decision = planner.plan(context, goal_latent)
            closest_cost = min(closest_cost, float(decision["cost"]))
            try:
                obs = bridge.move(MOVES[decision["action"]])
            except BridgeError:
                break
            png = renderer.render(obs["_render_state"])
            latent = encode(to_array(png))
            context = torch.cat([context[1:], latent.unsqueeze(0)], dim=0)
        else:
            eval_pos = _eval_position(obs)
            eval_room = str((obs.get("_render_state") or {}).get("level_id") or "")
            if eval_pos == task.eval_goal_pos and eval_room == task.eval_goal_room:
                reached_at = budget

        final_pos = _eval_position(obs)
        final_cost = float(goal_cost(encode(to_array(png)), goal_latent))
    finally:
        bridge.close()
        renderer.close()

    gx, gy = task.eval_goal_pos
    fx, fy = final_pos
    manhattan = (abs((fx or 0) - (gx or 0)) + abs((fy or 0) - (gy or 0))
                 if None not in (fx, fy, gx, gy) else None)

    return {
        "episode": task.ep_dir.name,
        "start_step": task.start_step,
        "goal_step": task.goal_step,
        "distance": task.distance,
        "reached": reached_at is not None,
        "reached_at": reached_at,
        "final_manhattan": manhattan,
        "final_latent_cost": final_cost,
        "best_plan_cost": closest_cost,
    }


def _eval_position(obs: dict) -> tuple:
    """Evaluator-only: the engine's own player position."""
    player = obs.get("player")
    if isinstance(player, dict):
        return (player.get("x"), player.get("y"))
    return (None, None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frames", default="runs/vision/frames")
    p.add_argument("--latents", default="runs/vision/latents")
    p.add_argument("--out", default=None)
    p.add_argument("--n-goals", type=int, default=100)
    p.add_argument("--per-episode", type=int, default=3)
    p.add_argument("--horizon", type=int, default=16,
                   help="action distance from start to goal")
    p.add_argument("--budget", type=int, default=32,
                   help="steps the agent may take; 2x horizon by default")
    p.add_argument("--plan-horizon", type=int, default=32,
                   help="CEM planning horizon in primitive steps")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device,
                            weights_only=False)
    config = HJEPAConfig(**checkpoint["config"])
    model = HJEPA(config).to(args.device).eval()
    model.load_state_dict(checkpoint["state"])
    print(f"model: {config.n_levels} level(s), strides {model.strides}")

    encoder = FrozenEncoder(HJEPAConfig()).to(args.device).eval()

    # The same held-out episodes the trainer kept back, by the same rule.
    _, val_eps = split_episodes(args.latents, args.val_frac, args.seed)
    print(f"held-out episodes: {len(val_eps)}")

    rng = random.Random(args.seed)
    tasks = sample_tasks(Path(args.frames), val_eps, args.horizon,
                         args.per_episode, rng)
    rng.shuffle(tasks)
    tasks = tasks[: args.n_goals]
    if not tasks:
        print("no goals could be sampled -- are the held-out episodes present "
              "under --frames?")
        return 1
    print(f"goals: {len(tasks)} at distance {args.horizon}, "
          f"budget {args.budget} steps\n")

    cem_config = GoalCEMConfig(horizon=args.plan_horizon,
                               n_samples=args.samples, n_iters=args.iters,
                               seed=args.seed)

    results = []
    for i, task in enumerate(tasks, 1):
        result = run_task(task, model, encoder, cem_config, args.device,
                          args.budget, args.size)
        results.append(result)
        rate = sum(r["reached"] for r in results) / len(results)
        print(f"  {i:4d}/{len(tasks)}  {result['episode']} "
              f"t={result['start_step']:3d}->{result['goal_step']:3d}  "
              f"{'REACHED' if result['reached'] else 'missed ':<8} "
              f"d={result['final_manhattan']}  "
              f"running {rate:.1%}", flush=True)

    reached = sum(r["reached"] for r in results)
    near = sum(1 for r in results
               if r["final_manhattan"] is not None and r["final_manhattan"] <= 1)
    summary = {
        "checkpoint": args.checkpoint,
        "n_levels": config.n_levels,
        "strides": list(model.strides),
        "goals": len(results),
        "horizon": args.horizon,
        "budget": args.budget,
        "plan_horizon": args.plan_horizon,
        "samples": args.samples,
        "success_rate": reached / len(results),
        "within_1_cell_rate": near / len(results),
        "mean_final_manhattan": float(np.mean(
            [r["final_manhattan"] for r in results
             if r["final_manhattan"] is not None] or [float("nan")])),
        "cem": asdict(cem_config),
        "results": results,
    }

    print("\n" + "=" * 60)
    print(f"GOAL-REACHING SUCCESS RATE : {summary['success_rate']:.1%} "
          f"({reached}/{len(results)})")
    print(f"within 1 cell              : {summary['within_1_cell_rate']:.1%}")
    print(f"mean final manhattan       : {summary['mean_final_manhattan']:.2f}")
    print(f"target (GOAL.md)           : ~80%   floor: ~60%")
    print("=" * 60)

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval_goal.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
