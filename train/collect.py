"""Collect transitions for world-model training.

Labels are computed here, at collection time, and every one of them comes from
an agent-visible field or from the agent's own percept:

``gem_delta``    difference of the public ``gem_count``
``room_change``  change of the public ``current_room``
``player_vis``   whether ``P`` is findable in the agent's own board
``death``        the public ``player_dead``
``moved``        whether the egocentric window changed at all -- the benchmark
                 never reports a blocked move, so this self-derived signal is
                 the honest substitute

Transitions that span a *control* action (undo, reset_level, goto_level) are
marked invalid and dropped. The world model only ever claims to predict the
eight plannable primitives; training it on a teleport would teach it a dynamics
it is never asked to roll out.

Three collection policies:

``random``     uniform over the 8 primitives. The baseline's data.
``heuristic``  a blocked-move-aware anti-reversal walk. No model needed, and it
               covers far more distinct states per action than uniform random,
               which matters because phase-0 data quality sets the ceiling on
               everything the planner can later do.
``agent``      the EFE agent itself, for Dyna-style iteration: collect with the
               current model, retrain, repeat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env.actions import N_ACTIONS
from env.mazebench_env import MazeBenchEnv, PlayerDead
from env.obs import encode

REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}      # up<->down, left<->right


class HeuristicExplorer:
    """Model-free coverage policy. Avoids reversing and re-trying blocked moves."""

    def __init__(self, seed: int = 0, rotate_prob: float = 0.12, reverse_prob: float = 0.08):
        self.rng = np.random.default_rng(seed)
        self.rotate_prob = rotate_prob
        self.reverse_prob = reverse_prob
        self.last_move: int | None = None
        self.blocked: set[int] = set()

    def reset(self) -> None:
        self.last_move = None
        self.blocked.clear()

    def observe_result(self, action: int, moved: bool | None) -> None:
        if action < 4:
            if moved is False:
                self.blocked.add(action)
            else:
                self.blocked.clear()
                self.last_move = action

    def act(self) -> int:
        if self.rng.random() < self.rotate_prob:
            return int(self.rng.integers(4, N_ACTIONS))
        options = [a for a in range(4) if a not in self.blocked]
        if not options:
            self.blocked.clear()
            options = list(range(4))
        if self.last_move is not None and len(options) > 1:
            rev = REVERSE[self.last_move]
            if rev in options and self.rng.random() > self.reverse_prob:
                options = [a for a in options if a != rev] or options
        # momentum: keep going straight when possible
        if self.last_move in options and self.rng.random() < 0.45:
            return self.last_move
        return int(options[self.rng.integers(len(options))])


class CoverageExplorer:
    """Pathfinding collector. Uses only what the agent can see.

    Why this exists: on real MazeBench geometry a random or heuristic walk is
    useless as a data source. Measured over 6 episodes of 256 actions, it
    collected **0 gems and never once changed room** -- in ``level_HxI`` the
    only walkable border cells are row 0, columns 6-9, a single narrow door,
    and a random walk from the spawn at (12,4) never reaches it. A buffer
    collected that way contains no gem events and no room changes, so the gem
    and room heads cannot learn anything at all.

    This is not privileged. ``camera_view`` is 16x16, the same as a room, so
    the agent sees the whole board; BFS over the rendered ASCII uses exactly
    the information the policy is allowed. Movement is screen-relative and the
    board renders in screen space, so a path on the board is a path in actions.

    Target priority: a visible gem, then an unvisited cell, then a door out.
    Collection policy only -- the evaluated agent never uses it.
    """

    def __init__(self, seed: int = 0, leave_after: int = 60, epsilon: float = 0.22):
        self.rng = np.random.default_rng(seed)
        self.leave_after = leave_after
        # Pathfinding alone only ever emits the 4 moves, so a buffer collected
        # from it contains no camera rotations at all -- half the plannable
        # action set, which the world model would then never learn and the
        # planner would still propose. Epsilon covers all 8.
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.visited: set[tuple[int, int]] = set()
        self.room: str | None = None
        self.steps_in_room = 0
        # Cells the world refused to let us enter. The rendered board marks a
        # ledge as a wall only relative to where the player is standing, so a
        # cell two steps away can look walkable and turn out not to be. BFS
        # over that board is optimistic; without this the policy bumps into
        # ledges and its move rate collapses (measured: 0.24 vs random's 0.58).
        self.blocked: set[tuple[int, int]] = set()
        self._last_target: tuple[int, int] | None = None
        self._last_pos: tuple[int, int] | None = None

    # -- board reading ------------------------------------------------------
    @staticmethod
    def _walkable(grid: np.ndarray) -> np.ndarray:
        from env.glyphs import GEM_IDS, PLAYER_IDS, WALKABLE_GLYPHS, glyph_id

        ok_ids = {glyph_id(g) for g in WALKABLE_GLYPHS} | set(GEM_IDS) | set(PLAYER_IDS)
        return np.isin(grid, list(ok_ids))

    def _bfs(self, grid: np.ndarray, start: tuple[int, int], blocked=frozenset()):
        """Breadth-first over walkable cells. Returns parent and distance maps."""
        walkable = self._walkable(grid)
        h, w = grid.shape
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        order: list[tuple[int, int]] = [start]
        head = 0
        while head < len(order):
            cell = order[head]
            head += 1
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if nxt in parent or not walkable[nxt] or nxt in blocked:
                    continue
                parent[nxt] = cell
                order.append(nxt)
        return parent, order

    def _first_step(self, parent, target, start) -> int | None:
        """Walk the parent chain back to ``start``; return the action index."""
        node = target
        while parent.get(node) is not None and parent[node] != start:
            node = parent[node]
        if parent.get(node) != start:
            return None
        dr, dc = node[0] - start[0], node[1] - start[1]
        for action, delta in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            if (dr, dc) == delta:
                return action
        return None

    # -- policy -------------------------------------------------------------
    def act(self, obs: dict) -> int:
        from env.glyphs import GEM_IDS
        from env.obs import find_player, parse_board

        if self.rng.random() < self.epsilon:
            self._last_target = None
            return int(self.rng.integers(N_ACTIONS))

        room = str(obs.get("current_room") or "")
        if room != self.room:
            self.room = room
            self.visited = set()
            self.steps_in_room = 0
        self.steps_in_room += 1

        grid = parse_board(str(obs.get("level") or ""))
        start = find_player(grid)
        if start is None:
            return int(self.rng.integers(4))
        # Did the previous step actually happen? If not, that cell is a ledge.
        if self._last_pos is not None and start == self._last_pos and self._last_target:
            self.blocked.add(self._last_target)
        elif start != self._last_pos:
            self.blocked.clear()
        self._last_pos = start
        self.visited.add(start)

        parent, order = self._bfs(grid, start, blocked=self.blocked)
        reachable = [c for c in order if c != start]
        if not reachable:
            return int(self.rng.integers(4))

        h, w = grid.shape
        gems = [c for c in reachable if int(grid[c]) in GEM_IDS]
        doors = [c for c in reachable if c[0] in (0, h - 1) or c[1] in (0, w - 1)]
        unseen = [c for c in reachable if c not in self.visited]

        if gems:
            target = gems[0]                       # BFS order == nearest first
        elif self.steps_in_room >= self.leave_after and doors:
            target = doors[0]                      # room exhausted: find a door
        elif unseen:
            target = unseen[0]
        elif doors:
            target = doors[0]
        else:
            return int(self.rng.integers(4))

        # Standing on a border cell with the room exhausted: step outward.
        # `doors` is built from `reachable`, which excludes the player's own
        # cell, so this has to be tested against `start` directly.
        on_border = start[0] in (0, h - 1) or start[1] in (0, w - 1)
        if on_border and self.steps_in_room >= self.leave_after and not gems:
            if start[0] == 0:
                return 0
            if start[0] == h - 1:
                return 1
            if start[1] == 0:
                return 2
            return 3

        action = self._first_step(parent, target, start)
        if action is None:
            return int(self.rng.integers(4))
        step = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
        self._last_target = (start[0] + step[0], start[1] + step[1])
        return action


@dataclass
class Episode:
    windows: np.ndarray       # (T+1, S, S) int16
    scalars: np.ndarray       # (T+1, D) float32
    actions: np.ndarray       # (T,) int8
    gem_delta: np.ndarray     # (T,) int8
    room_change: np.ndarray   # (T,) int8
    player_vis: np.ndarray    # (T,) int8
    death: np.ndarray         # (T,) int8
    moved: np.ndarray         # (T,) int8
    valid: np.ndarray         # (T,) int8 -- 0 where a control action intervened
    summary: dict

    def __len__(self) -> int:
        return len(self.actions)


def collect_episode(
    env: MazeBenchEnv,
    policy: str = "heuristic",
    max_steps: int = 256,
    window_size: int = 15,
    seed: int = 0,
    agent=None,
) -> Episode:
    rng = np.random.default_rng(seed)
    explorer = HeuristicExplorer(seed=seed) if policy == "heuristic" else None
    coverage = CoverageExplorer(seed=seed) if policy == "coverage" else None

    obs = env.reset()
    encoded = encode(obs, window_size)
    windows = [encoded.window.astype(np.int16)]
    scalars = [encoded.scalars]
    rooms = [encoded.room_id]
    gems = [encoded.gem_count]
    vis = [encoded.player_found]

    actions: list[int] = []
    valid: list[int] = []
    deaths: list[int] = []
    moved_flags: list[int] = []

    for _ in range(max_steps):
        if obs.get("player_dead"):
            obs, done, _ = env.recover()
            encoded = encode(obs, window_size)
            # a control action intervened: mark the *next* transition as the
            # start of a fresh valid chain by recording an invalid filler
            windows.append(encoded.window.astype(np.int16))
            scalars.append(encoded.scalars)
            rooms.append(encoded.room_id)
            gems.append(encoded.gem_count)
            vis.append(encoded.player_found)
            actions.append(0)
            valid.append(0)
            deaths.append(1)
            moved_flags.append(0)
            if explorer:
                explorer.reset()
            if coverage:
                coverage.reset()
            if done:
                break
            continue

        if policy == "random":
            action = int(rng.integers(N_ACTIONS))
        elif policy == "heuristic":
            action = explorer.act()
        elif policy == "coverage":
            action = coverage.act(obs)
        elif policy == "agent":
            if agent is None:
                raise ValueError("policy='agent' needs an agent")
            action = agent.choose_action(obs)
        else:
            raise ValueError(f"unknown policy: {policy}")

        prev_window = encoded.window
        try:
            obs, done, _ = env.step(action)
        except PlayerDead:
            obs, done, _ = env.recover()
            done = bool(done)

        encoded = encode(obs, window_size)
        did_move = not np.array_equal(encoded.window, prev_window)
        if explorer:
            explorer.observe_result(action, did_move if action < 4 else None)

        windows.append(encoded.window.astype(np.int16))
        scalars.append(encoded.scalars)
        rooms.append(encoded.room_id)
        gems.append(encoded.gem_count)
        vis.append(encoded.player_found)
        actions.append(action)
        valid.append(1)
        deaths.append(1 if obs.get("player_dead") else 0)
        moved_flags.append(1 if did_move else 0)

        if done:
            break

    t = len(actions)
    gem_delta = np.clip(np.diff(np.array(gems, dtype=np.int32))[:t], 0, 3).astype(np.int8)
    room_change = np.array(
        [1 if rooms[i + 1] != rooms[i] else 0 for i in range(t)], dtype=np.int8
    )
    return Episode(
        windows=np.stack(windows),
        scalars=np.stack(scalars).astype(np.float32),
        actions=np.array(actions, dtype=np.int8),
        gem_delta=gem_delta,
        room_change=room_change,
        player_vis=np.array([1 if v else 0 for v in vis[1:t + 1]], dtype=np.int8),
        death=np.array(deaths, dtype=np.int8),
        moved=np.array(moved_flags, dtype=np.int8),
        valid=np.array(valid, dtype=np.int8),
        summary=env.log.summary(),
    )


def save_episodes(episodes: list[Episode], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    lengths = []
    for i, ep in enumerate(episodes):
        for field_name in (
            "windows", "scalars", "actions", "gem_delta", "room_change",
            "player_vis", "death", "moved", "valid",
        ):
            payload[f"{field_name}_{i}"] = getattr(ep, field_name)
        lengths.append(len(ep))
    payload["lengths"] = np.array(lengths, dtype=np.int32)
    np.savez_compressed(path, **payload)
    (path.with_suffix(".summaries.json")).write_text(
        json.dumps([ep.summary for ep in episodes], indent=2), encoding="utf8"
    )
    return path


def load_episodes(path: str | Path) -> list[Episode]:
    data = np.load(Path(path), allow_pickle=False)
    n = len(data["lengths"])
    out = []
    for i in range(n):
        out.append(Episode(
            windows=data[f"windows_{i}"], scalars=data[f"scalars_{i}"],
            actions=data[f"actions_{i}"], gem_delta=data[f"gem_delta_{i}"],
            room_change=data[f"room_change_{i}"], player_vis=data[f"player_vis_{i}"],
            death=data[f"death_{i}"], moved=data[f"moved_{i}"], valid=data[f"valid_{i}"],
            summary={},
        ))
    return out
