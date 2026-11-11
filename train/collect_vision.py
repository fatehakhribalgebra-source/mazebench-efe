"""Collect ``(frame, action, next_frame)`` from the **real** MazeBench engine.

Phase 0 of `GOAL.md`. The bridge holds the authoritative game state; the renderer
draws it. We step the bridge, take its ``_render_state``, and hand that to the
renderer -- never driving both with actions, which desynchronises them silently.

    python -m train.collect_vision --episodes 400 --steps 128 --out runs/vision/frames

What this writes, per episode::

    ep_00000/
      frames/000000.png ...      the observation, 512x512, nothing else
      episode.json               actions + the fields the agent may see
      privileged.json            evaluator-only, kept deliberately separate

**Why two files.** ``board_state_hash``, player coordinates, gem ids,
``push_count``, ``moved``, ``room_changed`` and ``_render_state`` are
evaluator-only. Keeping them in a different file from the actions means a later
mistake has to be deliberate rather than accidental -- training code that globs
``episode.json`` cannot reach them.

**Camera.** Four moves only. `GOAL.md` fixes yaw and tilt *within* an episode so
the goal screenshot and the current frame share a camera, and varies yaw *across*
episodes so the model does not overfit one angle. We do not emit camera
rotations: the model has no action token for them, so a rotated frame pair would
be labelled with a move the rotation did not cause, which teaches the predictor
noise. See the note in the report.

**Episode length.** Level 3 has stride 16 and needs
``(context-1)*16 + rollout*16 + 1 = 81`` frames for a single window, so short
episodes silently contribute nothing to the top level. Default 128.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from env.bridge import BridgeConfig, BridgeError, MazeBridge
from env.obs import AGENT_VISIBLE_FIELDS
from env.renderer import FrameRenderer, RendererConfig
from train.collect import CoverageExplorer

#: The four moves, in the order ``model.hjepa.ACTIONS`` indexes them.
MOVES = ("up", "down", "left", "right")

#: The shipped world is a 16x16 grid of rooms, A..P on each axis (256 rooms,
#: confirmed against games/maze/world_map.json).
ROOM_LETTERS = string.ascii_uppercase[:16]
ALL_LEVELS = tuple(f"level_{x}x{y}" for y in ROOM_LETTERS for x in ROOM_LETTERS)


def agent_visible(obs: dict) -> dict:
    """Drop every privileged field. The policy sees what a player sees."""
    return {k: v for k, v in obs.items() if k in AGENT_VISIBLE_FIELDS}


def privileged(obs: dict) -> dict:
    """The evaluator-only half. Never used for training or planning."""
    player = obs.get("player") or {}
    return {
        "board_state_hash": obs.get("board_state_hash"),
        "player_x": player.get("x") if isinstance(player, dict) else None,
        "player_y": player.get("y") if isinstance(player, dict) else None,
        "push_count": obs.get("push_count"),
        "novel_push_count": obs.get("novel_push_count"),
        "room_changed": obs.get("room_changed"),
        "collected_gems": obs.get("collected_gems"),
        "level_id": (obs.get("_render_state") or {}).get("level_id"),
    }


class MoveOnlyCoverage:
    """``CoverageExplorer`` restricted to the four moves.

    The underlying policy's epsilon branch samples over all eight plannable
    actions, half of which are camera rotations. Those are not in this model's
    action space, so an episode containing them would carry transitions no action
    can explain. Resample instead of clamping -- ``action % 4`` would quietly
    double the weight on every move.
    """

    def __init__(self, seed: int = 0, leave_after: int = 60, epsilon: float = 0.22):
        self.inner = CoverageExplorer(seed=seed, leave_after=leave_after,
                                      epsilon=epsilon)
        self.rng = np.random.default_rng(seed + 991)

    def reset(self) -> None:
        self.inner.reset()

    def act(self, obs: dict) -> int:
        action = int(self.inner.act(obs))
        if action >= 4:
            action = int(self.rng.integers(4))
        # Enforced, not merely intended: a rotation reaching the bridge would
        # put a frame pair in the buffer labelled with a move that did not
        # cause it, and nothing downstream could detect that.
        if not 0 <= action < 4:
            raise AssertionError(f"non-move action {action} escaped the policy")
        return action


def collect_episode(
    level_id: str,
    yaw: int,
    steps: int,
    seed: int,
    out_dir: Path,
    view: int | str,
    size: int = 512,
) -> dict:
    """One episode, in its own renderer session. Writes frames and metadata.

    **A fresh session per episode is deliberate.** The renderer advances an
    internal phase once per ``render`` call, so a frame is a function of
    (state, renders so far in this session). Replaying an episode reproduces its
    latents only if that phase lines up, which means starting at render 0.
    Sharing one session across episodes would save ~2s each and make a single
    episode impossible to replay without replaying every episode before it.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer = FrameRenderer(RendererConfig(
        level_id=level_id, width=size, height=size, view=view, yaw=yaw))
    bridge = MazeBridge(BridgeConfig(level_id=level_id, yaw=yaw,
                                     observation_mode="text"))
    policy = MoveOnlyCoverage(seed=seed)
    actions: list[int] = []
    rooms: list[str] = []
    priv: list[dict] = []
    render_seconds = 0.0
    terminated = None

    try:
        renderer.init()
        obs = bridge.observe()
        t0 = time.perf_counter()
        png = renderer.render(obs["_render_state"])
        render_seconds += time.perf_counter() - t0
        (frames_dir / "000000.png").write_bytes(png)
        rooms.append(str(obs.get("current_room") or ""))
        priv.append(privileged(obs))

        for step in range(steps):
            action = policy.act(agent_visible(obs))
            try:
                obs = bridge.move(MOVES[action])
            except BridgeError as error:
                terminated = f"bridge: {error}"
                break

            t0 = time.perf_counter()
            png = renderer.render(obs["_render_state"])
            render_seconds += time.perf_counter() - t0
            (frames_dir / f"{step + 1:06d}.png").write_bytes(png)

            actions.append(action)
            rooms.append(str(obs.get("current_room") or ""))
            priv.append(privileged(obs))

            if obs.get("player_dead"):
                terminated = "player_dead"
                break
            if obs.get("game_won"):
                terminated = "game_won"
                break
    finally:
        bridge.close()
        renderer.close()

    n_frames = len(actions) + 1
    (out_dir / "episode.json").write_text(json.dumps({
        "n_frames": n_frames,
        "actions": actions,
        "action_names": [MOVES[a] for a in actions],
        "rooms": rooms,
        "start_level": level_id,
        "yaw": yaw,
        "view": view,
        "seed": seed,
        "terminated": terminated,
        "render_seconds": round(render_seconds, 3),
    }, indent=1), encoding="utf8")
    (out_dir / "privileged.json").write_text(
        json.dumps({"note": "evaluator-only; never train or plan on this",
                    "steps": priv}, indent=1), encoding="utf8")

    return {
        "frames": n_frames,
        "rooms": sorted(set(r for r in rooms if r)),
        "terminated": terminated,
        "render_seconds": render_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--steps", type=int, default=128,
                        help="moves per episode; 128 keeps stride-16 usable")
    parser.add_argument("--out", default="runs/vision/frames")
    parser.add_argument("--view", default="1",
                        help="neighbour-room rings; 1 is the benchmark default")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", default="all",
                        help="'all' for the 256 shipped rooms, or a comma list")
    parser.add_argument("--target-frames", type=int, default=0,
                        help="stop once this many frames exist (0 = no limit)")
    parser.add_argument("--min-frames", type=int, default=81,
                        help="discard episodes shorter than this")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    view = args.view if args.view == "world" else int(args.view)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    levels = (list(ALL_LEVELS) if args.levels == "all"
              else [s.strip() for s in args.levels.split(",") if s.strip()])
    rng = random.Random(args.seed)

    done = sorted(p.name for p in out_root.glob("ep_*") if (p / "episode.json").is_file())
    start_index = len(done) if args.resume else 0
    total_frames = 0
    if args.resume:
        for name in done:
            try:
                total_frames += json.loads(
                    (out_root / name / "episode.json").read_text(encoding="utf8")
                )["n_frames"]
            except Exception:
                pass
        print(f"resuming: {len(done)} episodes, {total_frames} frames already on disk")

    # One renderer session PER EPISODE, so each episode replays on its own.
    # See collect_episode. Probe once up front to fail fast on a bad browser
    # path rather than after the first episode's worth of work.
    probe = FrameRenderer(RendererConfig(
        level_id=levels[0], width=args.size, height=args.size, view=view, yaw=0))
    print(f"node    : {probe.node}")
    print(f"browser : {probe.browser}")
    print(f"view    : {view}   size: {args.size}x{args.size}")
    probe.close()

    rooms_seen: set[str] = set()
    yaws_used: set[int] = set()
    kept = 0
    discarded = 0
    deaths = 0
    wall_start = time.perf_counter()
    render_total = 0.0

    try:
        for i in range(start_index, start_index + args.episodes):
            if args.target_frames and total_frames >= args.target_frames:
                print(f"reached target of {args.target_frames} frames")
                break
            level_id = rng.choice(levels)
            yaw = rng.randrange(4)          # fixed within the episode, varied across
            ep_dir = out_root / f"ep_{i:05d}"
            seed = args.seed * 100003 + i

            try:
                summary = collect_episode(level_id, yaw, args.steps,
                                          seed, ep_dir, view, args.size)
            except Exception as error:
                print(f"  ep {i:05d} FAILED: {type(error).__name__}: {error}")
                continue

            render_total += summary["render_seconds"]
            if summary["frames"] < args.min_frames:
                discarded += 1
                print(f"  ep {i:05d} {level_id} yaw={yaw} "
                      f"{summary['frames']:4d} frames  DISCARDED "
                      f"({summary['terminated']})")
                continue

            kept += 1
            total_frames += summary["frames"]
            rooms_seen.update(summary["rooms"])
            yaws_used.add(yaw)
            if summary["terminated"] == "player_dead":
                deaths += 1

            elapsed = time.perf_counter() - wall_start
            print(f"  ep {i:05d} {level_id} yaw={yaw} "
                  f"{summary['frames']:4d} frames  "
                  f"rooms={len(summary['rooms'])}  "
                  f"total={total_frames}  "
                  f"{render_total / max(total_frames, 1):.3f} s/frame  "
                  f"elapsed={elapsed / 60:.1f}m"
                  + (f"  [{summary['terminated']}]" if summary["terminated"] else ""),
                  flush=True)
    finally:
        pass

    manifest = {
        "frames": total_frames,
        "episodes_kept": kept,
        "episodes_discarded": discarded,
        "episodes_ended_in_death": deaths,
        "rooms_covered": sorted(rooms_seen),
        "n_rooms_covered": len(rooms_seen),
        "yaws_used": sorted(yaws_used),
        "view": view,
        "size": args.size,
        "steps_requested": args.steps,
        "seconds_per_frame_render": round(render_total / max(total_frames, 1), 4),
        "wall_seconds": round(time.perf_counter() - wall_start, 1),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                            encoding="utf8")
    print("\n=== manifest ===")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rooms_covered"},
                     indent=1))
    print(f"rooms covered: {len(rooms_seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
