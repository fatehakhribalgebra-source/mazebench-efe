"""Run the ablation ladder, score every arm on shared seeds, write a replay.

    python play.py --checkpoint runs/dev/final.pt --real-levels --episodes 5

The point is the *comparison*, not a score. The project's research question is
whether an action-conditioned world model **with persistent episodic memory**
beats a **memoryless** one at reward-free exploration, so "beats uniform
random" does not answer it -- random only shows the planner does something.

Arms (see ``agent.agent.ARMS``):

    random          uniform over the 8 primitives
    pragmatic_only  world model, preferences only: no EFE curiosity at all
    no_epistemic    + ambiguity, still no information gain
    memoryless      + disagreement, but no persistent record of anything seen
    full            + episodic memory and the frontier-jump supervisor

Every arm runs on the *same* seeds, the same action budget and the same
observation mode, so differences are the arm and not the draw. With
``--exploit 0`` the gem preference is zeroed across all of them, which is the
reward-free protocol the question is actually about.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from agent.agent import ARMS, AgentConfig, EFEAgent, RandomAgent
from agent.efe import EFEConfig
from agent.planner import PlannerConfig
from env.mazebench_env import MazeBenchEnv
from model.world_model import HJEPA, WorldModelConfig
from viz.render import write_replay

REPORT_KEYS = (
    "gems", "rooms_visited", "unique_board_states", "state_novelty_rate",
    "novel_block_positions", "deaths", "blocked_move_rate", "total",
)


def make_env(args, seed: int) -> MazeBenchEnv:
    if args.engine == "mock":
        from env.mock import MockMazeBridge
        return MazeBenchEnv(
            bridge=MockMazeBridge(
                seed=seed, real_levels=args.real_levels, repo_root=args.repo_root
            ),
            max_actions=args.max_actions,
        )
    from env.bridge import BridgeConfig, MazeBridge
    return MazeBenchEnv(
        bridge=MazeBridge(BridgeConfig(
            level_id=args.level, view=args.view,
            repo_root=args.repo_root, node_bin=args.node_bin,
        )),
        max_actions=args.max_actions,
    )


def load_model(path: str | None, device, window: int, members: int) -> tuple[HJEPA, bool]:
    if path and Path(path).exists():
        payload = torch.load(path, map_location=device, weights_only=False)
        config = WorldModelConfig(**payload["config"]) if "config" in payload else WorldModelConfig()
        model = HJEPA(config)
        model.load_state_dict(payload["model"])
        return model.to(device), True
    print("!! no checkpoint found -- running an UNTRAINED model.")
    print("!! its predictions are meaningless; this only exercises the plumbing.")
    return HJEPA(WorldModelConfig(window_size=window, n_members=members)).to(device), False


def aggregate(summaries: list[dict]) -> dict[str, dict[str, float]]:
    """mean and standard error per metric. One rollout is a demo, not evidence."""
    out: dict[str, dict[str, float]] = {}
    for key in REPORT_KEYS:
        vals = [s[key] for s in summaries if isinstance(s.get(key), (int, float))]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        sem = (statistics.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
        out[key] = {"mean": mean, "sem": sem, "n": len(vals)}
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--engine", choices=["mock", "real"], default="mock")
    p.add_argument("--real-levels", action="store_true",
                   help="mock dynamics on the 256 real 16x16 level layouts")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--node-bin", default=None)
    p.add_argument("--level", default="level_HxI")
    p.add_argument("--view", default="top-diagonal")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-actions", type=int, default=256)
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exploit", type=float, default=1.0,
                   help="1 = benchmark objective, 0 = reward-free exploration protocol")
    p.add_argument("--arms", default="random,memoryless,full",
                   help=f"comma-separated; any of random,{','.join(ARMS)}")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--out", default="runs/play")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, trained = load_model(args.checkpoint, device, args.window, args.members)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm != "random" and arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from random,{','.join(ARMS)}")

    results: dict[str, list[dict]] = {arm: [] for arm in arms}
    replay_source = None

    for episode in range(args.episodes):
        seed = args.seed + episode
        for arm in arms:
            env = make_env(args, seed=seed)       # same seed for every arm
            try:
                if arm == "random":
                    RandomAgent(seed=seed, window_size=args.window).run_episode(
                        env, max_steps=args.max_actions
                    )
                    history = None
                else:
                    agent = EFEAgent(model, AgentConfig(
                        window_size=args.window, exploit=args.exploit, arm=arm,
                        device=str(device),
                        efe=EFEConfig(horizon=args.horizon),
                        planner=PlannerConfig(
                            horizon=args.horizon, n_samples=args.samples, n_iters=args.iters
                        ),
                    ))
                    history = agent.run_episode(env, max_steps=args.max_actions)
                summary = env.log.summary()
            finally:
                env.close()
            results[arm].append(summary)
            if arm == "full" and replay_source is None and history:
                replay_source = (history, summary)
            print(f"[{arm:<14} ep{episode}] gems {summary['gems']:3d}  "
                  f"rooms {summary['rooms_visited']:2d}  states {summary['unique_board_states']:4d}  "
                  f"novelty {summary['state_novelty_rate']:.3f}  deaths {summary['deaths']:2d}")

    stats = {arm: aggregate(rows) for arm, rows in results.items()}

    width = max(len(a) for a in arms) + 2
    print("\n" + "=" * (26 + width * len(arms)))
    header = f"{'metric':<26}" + "".join(f"{a:>{width + 8}}" for a in arms)
    print(header)
    print("-" * len(header))
    for key in REPORT_KEYS:
        if not any(key in stats[a] for a in arms):
            continue
        row = f"{key:<26}"
        for arm in arms:
            cell = stats[arm].get(key)
            row += f"{cell['mean']:>{width + 2}.2f}±{cell['sem']:<5.2f}" if cell else " " * (width + 8)
        print(row)
    print("=" * len(header))
    print(f"{args.episodes} episodes/arm on shared seeds, {args.max_actions} actions, "
          f"engine={args.engine}{' +real-levels' if args.real_levels else ''}, exploit={args.exploit}")
    if not trained:
        print("UNTRAINED MODEL -- these numbers measure nothing but that the code runs.")
    if args.episodes < 5:
        print("NOTE: fewer than 5 episodes per arm. Treat the error bars as decorative.")

    (out / "results.json").write_text(json.dumps({
        "args": vars(args), "trained": trained, "per_episode": results, "stats": stats,
    }, indent=2, default=str), encoding="utf8")

    if replay_source:
        history, summary = replay_source
        baseline = {k: v["mean"] for k, v in stats.get("random", {}).items()}
        path = write_replay(
            history, out / "replay.html",
            title=f"MazeBench EFE agent{'' if trained else ' (UNTRAINED)'}",
            summary=summary, baseline=baseline,
        )
        print(f"\nreplay: {path}")
    print(f"results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
