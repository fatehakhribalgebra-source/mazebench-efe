"""A pure-Python stand-in for maze-bridge.js.

THIS IS NOT THE BENCHMARK. It is a small multi-room Sokoban-with-gems world
that speaks the same command surface and emits the same snapshot fields, so the
world model, the EFE planner, and the tests can all run with no Node runtime and
no browser. Numbers from it say nothing about MazeBench performance; it exists
so that code is exercised before it touches the real engine.

What it reproduces faithfully, because the agent depends on it:
  * the ASCII glyph contract (``env.glyphs``), top vs side projection
  * limited visibility -- unseen tiles render as the contract's unknown glyph,
    which is what makes exploration and camera rotation meaningful
  * camera yaw rotating the rendered board, pitch switching top/side glyphs
  * persistent world state across rooms, monotonic gem score
  * box pushing, holes that kill, room transitions at the edges
  * the agent-visible / evaluator-only field split

What it does not reproduce: ice physics, slopes, clones, punchers, lifts,
buttons, undo depth beyond one step, and the real world's 200+ hand-built rooms.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .glyphs import ACTORS, TERRAIN, UNKNOWN

# Every one of MazeBench's 256 shipped levels is exactly 16x16 tiles
# (counted from games/maze/levels/*.txt, which are space-separated cell
# tokens). Matching it means window sizing and room-traversal distances here
# behave like the real thing.
ROOM_H, ROOM_W = 16, 16
VIEW_NAMES = ["top", "top-diagonal", "diagonal", "side-diagonal", "side"]

FLOOR, WALL, HOLE, GEM, BOX, PLAYER, EMPTY = "floor", "wall", "hole", "gem", "box", "player", "empty"


def _glyph(kind: str, side: bool) -> str:
    idx = 1 if side else 0
    if kind in TERRAIN:
        return TERRAIN[kind][idx]
    if kind in ACTORS:
        return ACTORS[kind][idx]
    return UNKNOWN["terrain"][idx]


@dataclass
class Room:
    """One room: static terrain plus mutable gem/box sets."""

    room_id: str
    terrain: list[list[str]]
    gems: set[tuple[int, int]] = field(default_factory=set)
    boxes: set[tuple[int, int]] = field(default_factory=set)
    spawn: tuple[int, int] = (ROOM_H // 2, ROOM_W // 2)
    #: Surface height per cell, for levels parsed from the real game. MazeBench
    #: is 3D: a `.+#+#` cell is ground at height 2, not a wall. Collapsing
    #: every stack to a wall sealed the real levels -- in level_HxI the only
    #: door became unreachable and no episode ever left the starting room.
    #: With heights, a step is allowed when the surfaces differ by at most 1,
    #: so walls emerge from the terrain instead of being asserted.
    heights: list[list[int]] | None = None

    def clone_dynamic(self) -> tuple[set, set]:
        return set(self.gems), set(self.boxes)


def _build_room(room_id: str, rng: random.Random) -> Room:
    terrain = [[FLOOR] * ROOM_W for _ in range(ROOM_H)]
    for r in range(ROOM_H):
        for c in range(ROOM_W):
            if r in (0, ROOM_H - 1) or c in (0, ROOM_W - 1):
                terrain[r][c] = WALL
    # doorways in the middle of each wall, so rooms connect
    terrain[0][ROOM_W // 2] = FLOOR
    terrain[ROOM_H - 1][ROOM_W // 2] = FLOOR
    terrain[ROOM_H // 2][0] = FLOOR
    terrain[ROOM_H // 2][ROOM_W - 1] = FLOOR

    interior = [
        (r, c)
        for r in range(1, ROOM_H - 1)
        for c in range(1, ROOM_W - 1)
        if not (r == ROOM_H // 2 or c == ROOM_W // 2)
    ]
    rng.shuffle(interior)
    for pos in interior[: rng.randint(4, 10)]:
        terrain[pos[0]][pos[1]] = WALL
    for pos in interior[12:14]:
        terrain[pos[0]][pos[1]] = HOLE

    free = [
        (r, c)
        for r in range(1, ROOM_H - 1)
        for c in range(1, ROOM_W - 1)
        if terrain[r][c] == FLOOR and (r, c) != (ROOM_H // 2, ROOM_W // 2)
    ]
    rng.shuffle(free)
    gems = set(free[: rng.randint(2, 5)])
    boxes = set(free[6:8])
    return Room(room_id=room_id, terrain=terrain, gems=gems, boxes=boxes)


class MockMazeBridge:
    """Duck-typed stand-in for ``env.bridge.MazeBridge``."""

    def __init__(
        self,
        level_id: str = "level_HxI",
        view: str = "top-diagonal",
        yaw: int = 0,
        seed: int = 0,
        world_size: int = 4,
        visibility: int = 3,
        real_levels: bool = False,
        repo_root: str | None = None,
        **_ignored,
    ):
        self.seed = seed
        self.world_size = world_size
        self.visibility = visibility
        # Real MazeBench geometry, parsed from the shipped level files. The
        # layouts, gems and boxes are then the benchmark's own; the dynamics
        # are still this mock's. See env/real_levels.py.
        self.level_source = None
        if real_levels:
            from .real_levels import RealLevelSource

            self.level_source = RealLevelSource(seed=seed, repo_root=repo_root)
            # world_parsing.json: world_size [16,16], level_size [16,16],
            # camera_view [16,16]. So the world is a 16x16 grid of rooms and
            # the camera frames a whole room -- not a radius around the player.
            self.world_size = 16
            if visibility < max(ROOM_H, ROOM_W):
                self.visibility = max(ROOM_H, ROOM_W)
        self.start_level = level_id
        self.start_pitch = VIEW_NAMES.index(view) if view in VIEW_NAMES else 1
        self.start_yaw = int(yaw) % 4
        self._rooms: dict[str, Room] = {}
        self._initial: dict[str, tuple[set, set]] = {}
        self.reset_run()

    # -- world construction -------------------------------------------------
    def _room_coords(self, level_id: str) -> tuple[int, int]:
        raw = level_id.replace("level_", "")
        a, _, b = raw.partition("x")
        return (ord(a.upper()) - 65) % self.world_size, (ord(b.upper()) - 65) % self.world_size

    def _level_id(self, x: int, y: int) -> str:
        return f"level_{chr(65 + x % self.world_size)}x{chr(65 + y % self.world_size)}"

    def _room(self, level_id: str) -> Room:
        if level_id not in self._rooms:
            if self.level_source is not None:
                room = self.level_source.room_for(level_id)
            else:
                x, y = self._room_coords(level_id)
                rng = random.Random((self.seed, x, y).__hash__())
                room = _build_room(level_id, rng)
            self._rooms[level_id] = room
            self._initial[level_id] = room.clone_dynamic()
        return self._rooms[level_id]

    def _dims(self, room: Room) -> tuple[int, int]:
        return len(room.terrain), len(room.terrain[0])

    # -- session state ------------------------------------------------------
    def reset_run(self) -> dict:
        self._rooms.clear()
        self._initial.clear()
        self.level_id = self.start_level
        self.pitch = self.start_pitch
        self.yaw = self.start_yaw
        room = self._room(self.level_id)
        self.pos = room.spawn
        self.collected: set[str] = set()
        self.visited: set[str] = {self.level_id}
        self.action_count = 0
        self.push_count = 0
        self.novel_push_states: set[str] = set()
        self.dead = False
        self._undo: tuple | None = None
        self._closed = False
        return self._snapshot("reset_run")

    # -- rendering ----------------------------------------------------------
    MAX_STEP_UP = 1

    def _height(self, room: Room, r: int, c: int) -> int:
        return 0 if room.heights is None else room.heights[r][c]

    def _can_enter(self, room: Room, r: int, c: int) -> bool:
        """Height-aware passability: a step of more than one level is a wall."""
        if room.terrain[r][c] == WALL:
            return False
        if room.heights is None:
            return True
        here = self._height(room, *self.pos)
        return abs(room.heights[r][c] - here) <= self.MAX_STEP_UP

    def _visible(self, r: int, c: int) -> bool:
        return max(abs(r - self.pos[0]), abs(c - self.pos[1])) <= self.visibility

    def _cell_kind(self, room: Room, r: int, c: int) -> str:
        if (r, c) == self.pos and not self.dead:
            return PLAYER
        if (r, c) in room.boxes:
            return BOX
        if (r, c) in room.gems:
            return GEM
        return room.terrain[r][c]

    def _render(self) -> str:
        room = self._room(self.level_id)
        height, width = self._dims(room)
        side = self.pitch >= 3
        grid = []
        for r in range(height):
            row = []
            for c in range(width):
                if not self._visible(r, c):
                    row.append(UNKNOWN["terrain"][1 if side else 0])
                    continue
                kind = self._cell_kind(room, r, c)
                if (kind == FLOOR and room.heights is not None
                        and not self._can_enter(room, r, c)):
                    # Seen but not steppable from here -- a ledge reads as an
                    # obstacle, which is what the player can actually infer.
                    kind = WALL
                row.append(" " if kind == EMPTY else _glyph(kind, side))
            grid.append(row)
        for _ in range(self.yaw):
            grid = [list(row) for row in zip(*grid[::-1])]
        return "\n".join("".join(row) for row in grid)

    def _board_hash(self) -> str:
        room = self._room(self.level_id)
        payload = "|".join([
            self.level_id,
            str(sorted(room.gems)),
            str(sorted(room.boxes)),
            str(self.pos),
            str(self.dead),
        ])
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def _snapshot(self, action: str, **extra) -> dict:
        room = self._room(self.level_id)
        self.visited.add(self.level_id)
        won = len(self.collected) >= 100
        snapshot = {
            "ok": True,
            "action": action,
            "action_count": self.action_count,
            "allowed_commands": (
                ["undo", "reset_level", "observe"]
                if self.dead
                else ["move", "rotate_camera", "undo", "reset_level", "goto_level", "observe"]
            ),
            "board_state_hash": self._board_hash(),
            "board_state_hash_version": "mock-1",
            "collected_gems": sorted(self.collected),
            "current_room": self.level_id,
            "current_view": VIEW_NAMES[self.pitch],
            "death_message": "You died." if self.dead else "",
            "gem_count": len(self.collected),
            "level": self._render(),
            "ascii_legend": [],
            "novel_push_count": len(self.novel_push_states),
            "push_count": self.push_count,
            "player": None if self.dead else {"x": self.pos[1], "y": self.pos[0]},
            "player_dead": self.dead,
            "visited_levels": sorted(self.visited),
            "yaw": self.yaw,
            "_render_state": {
                "level_id": self.level_id, "pitch": self.pitch, "yaw": self.yaw,
                "action_count": self.action_count, "collected_gems": sorted(self.collected),
                "visited_levels": sorted(self.visited), "push_count": self.push_count,
                "pos": list(self.pos), "gems": sorted(room.gems), "boxes": sorted(room.boxes),
            },
        }
        if won:
            snapshot["game_won"] = True
            snapshot["scorecard"] = self._scorecard()
        snapshot.update(extra)
        return snapshot

    def _scorecard(self) -> dict:
        return {
            "collected_gems": len(self.collected),
            "visited_level_count": len(self.visited),
            "block_pushes": self.push_count,
            "novel_block_positions": len(self.novel_push_states),
            "action_count": self.action_count,
        }

    # -- dynamics -----------------------------------------------------------
    _DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def _screen_to_world(self, direction: str) -> tuple[int, int]:
        """Movement is screen-relative, so yaw rotates the intended delta."""
        dr, dc = self._DELTA[direction]
        for _ in range(self.yaw):
            dr, dc = -dc, dr
        return dr, dc

    def _move(self, direction: str) -> dict:
        if self.dead:
            raise RuntimeError("You died.")
        room = self._room(self.level_id)
        height, width = self._dims(room)
        self._undo = (self.level_id, self.pos, set(room.gems), set(room.boxes), self.dead)
        dr, dc = self._screen_to_world(direction)
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.action_count += 1
        moved, room_changed, collected, pushes, novel = False, False, [], 0, 0

        # leaving through a doorway
        if not (0 <= nr < height and 0 <= nc < width):
            x, y = self._room_coords(self.level_id)
            if dr < 0:
                y -= 1
            elif dr > 0:
                y += 1
            elif dc < 0:
                x -= 1
            else:
                x += 1
            self.level_id = self._level_id(x, y)
            new_room = self._room(self.level_id)
            nh, nw = self._dims(new_room)
            self.pos = (
                (nh - 2 if dr < 0 else 1) if dr else self.pos[0],
                (nw - 2 if dc < 0 else 1) if dc else self.pos[1],
            )
            self.pos = (min(max(self.pos[0], 1), nh - 2), min(max(self.pos[1], 1), nw - 2))
            if new_room.terrain[self.pos[0]][self.pos[1]] == WALL:
                new_room.terrain[self.pos[0]][self.pos[1]] = FLOOR
            self.visited.add(self.level_id)
            return self._snapshot(
                "move", direction=direction, moved=True, room_changed=True,
                collected_this_action=[], pushes_this_action=0, novel_pushes_this_action=0,
            )

        target = room.terrain[nr][nc]

        if (nr, nc) in room.boxes:
            br, bc = nr + dr, nc + dc
            blocked = (
                not (0 <= br < height and 0 <= bc < width)
                or room.terrain[br][bc] == WALL
                or (room.heights is not None
                    and abs(room.heights[br][bc] - room.heights[nr][nc]) > self.MAX_STEP_UP)
                or (br, bc) in room.boxes
            )
            if not blocked:
                room.boxes.discard((nr, nc))
                if room.terrain[br][bc] == HOLE:
                    room.terrain[br][bc] = FLOOR          # box fills the hole
                else:
                    room.boxes.add((br, bc))
                self.pos = (nr, nc)
                moved, pushes = True, 1
                self.push_count += 1
                key = f"{self.level_id}:{br},{bc}"
                if key not in self.novel_push_states:
                    self.novel_push_states.add(key)
                    novel = 1
        elif target == WALL or not self._can_enter(room, nr, nc):
            pass
        elif target == HOLE:
            self.pos = (nr, nc)
            self.dead = True
            moved = True
        else:
            self.pos = (nr, nc)
            moved = True

        if moved and (self.pos in room.gems):
            room.gems.discard(self.pos)
            gid = f"{self.level_id}:{self.pos[0]},{self.pos[1]}"
            if gid not in self.collected:
                self.collected.add(gid)
                collected.append(gid)

        return self._snapshot(
            "move", direction=direction, moved=moved, room_changed=room_changed,
            collected_this_action=collected, pushes_this_action=pushes,
            novel_pushes_this_action=novel,
        )

    # -- command surface ----------------------------------------------------
    def call(self, command: str, **kwargs) -> dict:
        if self._closed:
            raise RuntimeError("bridge closed")
        if command == "observe":
            return self._snapshot("observe")
        if command == "move":
            return self._move(str(kwargs.get("direction")))
        if command == "rotate_camera":
            direction = str(kwargs.get("direction"))
            if direction == "up":
                self.pitch = max(0, self.pitch - 1)
            elif direction == "down":
                self.pitch = min(4, self.pitch + 1)
            elif direction == "left":
                self.yaw = (self.yaw - 1) % 4
            elif direction == "right":
                self.yaw = (self.yaw + 1) % 4
            else:
                raise RuntimeError("bad rotate direction")
            self.action_count += 1
            return self._snapshot("rotate_camera", direction=direction)
        if command == "undo":
            self.action_count += 1
            if self._undo is not None:
                level, pos, gems, boxes, dead = self._undo
                self.level_id, self.pos, self.dead = level, pos, dead
                room = self._room(level)
                room.gems, room.boxes = gems, boxes
                self._undo = None
                return self._snapshot("undo", undone=True)
            return self._snapshot("undo", undone=False)
        if command == "reset_level":
            room = self._room(self.level_id)
            gems, boxes = self._initial[self.level_id]
            room.gems, room.boxes = set(gems), set(boxes)
            self.pos = room.spawn
            self.dead = False
            self.action_count += 1
            return self._snapshot("reset_level", reset=True)
        if command == "goto_level":
            level = f"level_{str(kwargs.get('x')).upper()}x{str(kwargs.get('y')).upper()}"
            if level not in self.visited:
                raise RuntimeError(f"cannot goto unvisited level: {level}")
            self.level_id = level
            self.pos = self._room(level).spawn
            self.dead = False
            self.action_count += 1
            return self._snapshot("goto_level", destination_room=level)
        if command == "no_move":
            self.action_count += 1
            return self._snapshot("no_move", moved=False, synthetic=True)
        if command == "reset_run":
            return self.reset_run()
        if command == "scorecard":
            return self._snapshot("scorecard", scorecard=self._scorecard())
        if command == "quit":
            self.action_count += 1
            return self._snapshot("quit", game_lost=True, quit=True, scorecard=self._scorecard())
        if command == "replay_fast":
            return {"ok": True, "replay_fast": bool(kwargs.get("enabled", True))}
        if command == "close":
            self._closed = True
            return {"ok": True, "action": "close"}
        raise RuntimeError(f"unknown command: {command}")

    # -- parity with MazeBridge --------------------------------------------
    def observe(self) -> dict:
        return self.call("observe")

    def reset_run_cmd(self) -> dict:
        return self.call("reset_run")

    def scorecard(self) -> dict:
        return self.call("scorecard")

    def close(self) -> None:
        self._closed = True

    @property
    def alive(self) -> bool:
        return not self._closed
