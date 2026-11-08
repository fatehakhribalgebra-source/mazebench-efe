"""Train the hierarchical world model.

Three phases, run in order by ``main``:

0. **collect** -- fill a replay buffer with the heuristic explorer (and uniform
   random, for the baseline's data distribution).
1. **level 1** -- encoder, predictor ensemble, tile head and outcome heads on
   one-step transitions.
2. **level 2** -- chunk dynamics on pooled latents, with level 1 frozen. Frozen
   on purpose: level 2 should learn abstract dynamics over a *fixed*
   representation, otherwise the two levels chase each other and neither
   converges.

Then optionally iterate: ``--dyna-rounds`` re-collects with the trained EFE
agent and retrains, so later data comes from states a curious agent actually
reaches rather than from a random walk.

Validation is on held-out episodes, and the metric that matters is
``changed_tile_acc`` -- accuracy restricted to tiles that actually changed.
Overall tile accuracy is near 99% for a model that predicts "nothing moves",
so it is close to meaningless on a grid world. This project has made that
mistake before; the earlier run's own notes record whole-board error improving
while the changed-pixel error got worse.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from agent.agent import AgentConfig, EFEAgent
from env.mazebench_env import MazeBenchEnv
from model.world_model import HJEPA, WorldModelConfig
from train.collect import Episode, collect_episode, load_episodes, save_episodes
from train.dataset import ChunkDataset, TransitionDataset, bootstrap_mask


# ---------------------------------------------------------------- env factory
def make_env(args, seed: int) -> MazeBenchEnv:
    """Real engine when available, mock otherwise."""
    if args.engine == "mock":
        from env.mock import MockMazeBridge
        level_id = args.level
        if getattr(args, "start_in_gem_rooms", False) and getattr(args, "real_levels", False):
            # Only 86 of 256 rooms hold a gem, so collecting from the single
            # default start room yields no gem events at all. Spreading starts
            # over gem-bearing rooms is what gives the gem head a signal.
            from env.real_levels import RealLevelSource
            rooms = RealLevelSource(seed=0, repo_root=args.repo_root).rooms_with_gems()
            if rooms:
                level_id = rooms[seed % len(rooms)]
        return MazeBenchEnv(
            bridge=MockMazeBridge(
                seed=seed, level_id=level_id,
                real_levels=getattr(args, "real_levels", False),
                repo_root=args.repo_root,
            ),
            max_actions=args.max_actions,
        )
    from env.bridge import BridgeConfig, MazeBridge
    bridge = MazeBridge(BridgeConfig(
        level_id=args.level, view=args.view, repo_root=args.repo_root, node_bin=args.node_bin
    ))
    return MazeBenchEnv(bridge=bridge, max_actions=args.max_actions)


# -------------------------------------------------------------------- collect
def collect(args, agent=None, policy: str | None = None, n_episodes: int | None = None) -> list[Episode]:
    episodes: list[Episode] = []
    policy = policy or args.policy
    n_episodes = n_episodes or args.episodes
    for i in range(n_episodes):
        env = make_env(args, seed=args.seed + i)
        try:
            ep = collect_episode(
                env, policy=policy, max_steps=args.max_actions,
                window_size=args.window, seed=args.seed + i, agent=agent,
            )
        finally:
            env.close()
        episodes.append(ep)
        print(f"  episode {i + 1}/{n_episodes}: {len(ep):4d} transitions  {ep.summary.get('gems', 0)} gems  "
              f"{ep.summary.get('rooms_visited', 0)} rooms  novelty {ep.summary.get('state_novelty_rate', 0):.2f}")
    return episodes


# ------------------------------------------------------------------- level 1
def train_level1(model: HJEPA, train_ds, val_ds, args, device) -> dict:
    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters()
         if p.requires_grad and not n.startswith("abstract")],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_ds.sampler(num_samples=args.steps_per_epoch * args.batch_size),
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(val_ds) else None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    history = []
    best = {"changed_tile_acc": -1.0, "epoch": -1}
    for epoch in range(args.epochs):
        model.train()
        running: dict[str, float] = {}
        count = 0
        t0 = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["member_mask"] = bootstrap_mask(
                model.ensemble.n_members, batch["action"].shape[0], device, p=args.bootstrap_p
            )
            loss, stats = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            model.update_targets()
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v
            count += 1
        sched.step()
        train_stats = {k: v / max(1, count) for k, v in running.items()}

        val_stats = evaluate(model, val_loader, device) if val_loader else {}
        line = (f"[L1 {epoch + 1}/{args.epochs}] loss {train_stats['loss']:.4f} "
                f"tile {train_stats['tile_acc']:.3f} changed {train_stats['changed_tile_acc']:.3f} "
                f"gem_rec {train_stats['gem_recall']:.3f}/prec {train_stats['gem_precision']:.3f}")
        if val_stats:
            line += (f" | val changed {val_stats['changed_tile_acc']:.3f} "
                     f"gem_rec {val_stats['gem_recall']:.3f}")
        print(line + f"  ({time.time() - t0:.0f}s)")
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})

        score = val_stats.get("changed_tile_acc", train_stats["changed_tile_acc"])
        if score > best["changed_tile_acc"]:
            best = {"changed_tile_acc": score, "epoch": epoch + 1}
            torch.save(
                {"model": model.state_dict(), "config": model.config.__dict__, "epoch": epoch + 1},
                Path(args.out) / "level1_best.pt",
            )
    return {"history": history, "best": best}


@torch.no_grad()
def evaluate(model: HJEPA, loader, device) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    running: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, stats = model.loss(batch)
        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v
        count += 1
    return {k: v / max(1, count) for k, v in running.items()}


# ------------------------------------------------------------------- level 2
def train_hierarchy(model: HJEPA, episodes: list[Episode], args, device) -> dict:
    ds = ChunkDataset(
        episodes, chunk_len=model.config.chunk_len,
        n_levels=model.config.n_levels,
        # Index at the deepest arm being compared so every depth trains on an
        # identical sample set; otherwise a deeper model is starved, not tested.
        index_levels=getattr(args, "index_levels", None) or model.config.n_levels,
    )
    if not len(ds):
        print("[H] no chunks available; skipping")
        return {}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("abstract"))
    print("  chunk samples per level:", ds.level_counts())
    opt = torch.optim.AdamW(model.abstract.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    history = []
    for epoch in range(args.l2_epochs):
        # evaluate() leaves the model in eval mode, and cuDNN refuses to run
        # GRU backward there. Level 2 owns a GRU, so re-assert train mode.
        model.train()
        running: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, stats = model.hierarchy_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.abstract.parameters(), args.grad_clip)
            opt.step()
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v
            count += 1
        stats = {k: v / max(1, count) for k, v in running.items()}
        parts = " ".join(
            f"L{k}:pred {stats[f'l{k}_pred']:.4f}/var {stats[f'l{k}_var']:.4f}"
            for k in range(2, model.config.n_levels + 1)
            if f'l{k}_pred' in stats
        )
        print(f"[H {epoch + 1}/{args.l2_epochs}] {parts}")
        history.append({"epoch": epoch + 1, **stats})

    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.target_encoder.parameters():
        p.requires_grad_(False)
    return {"history": history}


# ----------------------------------------------------------------------- main
def build_args(argv=None):
    p = argparse.ArgumentParser(description="Train the MazeBench H-JEPA / EFE world model")
    p.add_argument("--engine", choices=["mock", "real"], default="mock",
                   help="'real' drives MazeBenchEngine via maze-bridge.js (needs Node)")
    p.add_argument("--repo-root", default=None, help="MazeBenchEngine checkout")
    p.add_argument("--start-in-gem-rooms", action="store_true",
                   help="spread collection starts over the 86 gem-bearing rooms "
                        "(collection only; evaluation starts where the benchmark says)")
    p.add_argument("--collect-only", action="store_true",
                   help="fill a buffer and exit, no training")
    p.add_argument("--real-levels", action="store_true",
                   help="mock dynamics on the 256 real 16x16 level layouts (no Node needed)")
    p.add_argument("--node-bin", default=None)
    p.add_argument("--level", default="level_HxI")
    p.add_argument("--view", default="top-diagonal")

    p.add_argument("--out", default="runs/dev")
    p.add_argument("--buffer", default=None, help="reuse a saved buffer instead of collecting")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--val-episodes", type=int, default=4)
    p.add_argument("--max-actions", type=int, default=256)
    p.add_argument("--policy", choices=["random", "heuristic", "coverage", "agent"], default="coverage")
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--l2-epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bootstrap-p", type=float, default=0.8)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--chunk-len", type=int, default=4)
    p.add_argument("--levels", type=int, default=2, choices=[2, 3, 4],
                   help="total hierarchy levels including the primitive one")
    p.add_argument("--index-levels", type=int, default=None, choices=[2, 3, 4],
                   help="index chunk samples at this depth so a 2/3/4 comparison "
                        "sees identical data; set it to the deepest arm")

    p.add_argument("--dyna-rounds", type=int, default=0,
                   help="rounds of (collect with trained agent -> retrain)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = build_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}   engine: {args.engine}   out: {out}")

    # --- phase 0: data ---
    if args.buffer and Path(args.buffer).exists():
        print(f"loading buffer {args.buffer}")
        episodes = load_episodes(args.buffer)
    else:
        print(f"collecting {args.episodes} episodes with policy={args.policy}")
        episodes = collect(args)
        save_episodes(episodes, out / "buffer.npz")
    if args.collect_only:
        stats = TransitionDataset(episodes).label_stats()
        print("\nbuffer:", out / "buffer.npz")
        print("label stats:", json.dumps(stats, indent=2))
        return

    print(f"collecting {args.val_episodes} validation episodes")
    val_args = build_args(argv)
    val_args.seed = args.seed + 10_000
    val_episodes = collect(val_args, n_episodes=args.val_episodes) if args.val_episodes else []

    train_ds = TransitionDataset(episodes)
    val_ds = TransitionDataset(val_episodes)
    print("label stats:", json.dumps(train_ds.label_stats(), indent=2))

    # --- model ---
    config = WorldModelConfig(
        window_size=args.window, channels=args.channels,
        n_members=args.members, chunk_len=args.chunk_len, n_levels=args.levels,
    )
    model = HJEPA(config).to(device)
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # --- phases 1 and 2 ---
    report = {"args": vars(args), "label_stats": train_ds.label_stats()}
    report["level1"] = train_level1(model, train_ds, val_ds, args, device)
    report["hierarchy"] = train_hierarchy(model, episodes, args, device)

    # --- optional Dyna iteration ---
    for round_idx in range(args.dyna_rounds):
        print(f"\n=== Dyna round {round_idx + 1}/{args.dyna_rounds} ===")
        agent = EFEAgent(model, AgentConfig(window_size=args.window, device=str(device)))
        fresh = collect(args, agent=_ActionAdapter(agent), policy="agent",
                        n_episodes=max(2, args.episodes // 2))
        episodes = episodes + fresh
        train_ds = TransitionDataset(episodes)
        report[f"level1_dyna{round_idx + 1}"] = train_level1(model, train_ds, val_ds, args, device)
        report[f"hierarchy_dyna{round_idx + 1}"] = train_hierarchy(model, episodes, args, device)
        save_episodes(episodes, out / f"buffer_dyna{round_idx + 1}.npz")

    torch.save({"model": model.state_dict(), "config": config.__dict__}, out / "final.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    print(f"\nsaved {out / 'final.pt'}  and  {out / 'report.json'}")


class _ActionAdapter:
    """Expose ``EFEAgent`` as a one-shot ``choose_action`` for the collector."""

    def __init__(self, agent: EFEAgent):
        self.agent = agent

    def choose_action(self, obs: dict) -> int:
        encoded, _ = self.agent.perceive(obs)
        window, scalars = self.agent._tensors(encoded)
        with torch.no_grad():
            z, _ = self.agent.model.encode(window, scalars)
        plan = self.agent.planner.plan(
            z0=z, start_pos=self.agent.memory.est_pos,
            start_room=encoded.room_id, start_yaw=int(obs.get("yaw") or 0),
        )
        self.agent._last_action = plan.action
        return plan.action


if __name__ == "__main__":
    main()
