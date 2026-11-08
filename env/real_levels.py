"""Load real MazeBench level files into the mock world.

The engine is JavaScript and running it needs a Node subprocess. The *levels*,
though, are plain text data in a documented token format, so real room geometry
can be used without executing anything. That gets training data off
procedurally-generated rooms and onto the 256 hand-built ones the benchmark
actually ships.

Format, from ``games/maze/level_parsing.json``:

* cells are separated by a space, 16 x 16 per level in all 256 shipped files
* ``+`` is the *block adder*: ``.+#+#+#+#`` is a floor with four stacked walls,
  i.e. a wall column four high
* a cell with no base ``.`` is void

**This is a 2D projection of a 3D world, and that is a real approximation.**
MazeBench has elevation, ice physics, slopes, lifts, buttons that lower walls,
punchers and clones. Here a cell is reduced to one of floor / wall / hole /
gem / box / hazard by looking at the top of its stack. Raised platforms become
walls, slopes become floor, and nothing moves vertically. So:

* **geometry, gem placement and box placement are real**
* **dynamics are still the mock's**

Which makes this a strictly better training distribution than random rooms, and
still not the benchmark. Numbers from it are not MazeBench results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .mock import BOX, FLOOR, GEM, HOLE, Room

# token -> mock cell kind. Keys are the *top* of a cell's stack.
WALL_TOKENS = {
    "#", "I", "O",                                   # wall, ice block, orange wall
    "t1", "t2", "t3", "t4", "st1", "st3", "st4",     # trees
    "sh",                                            # shrub
    "b1", "b2", "b3", "b4",                          # block assets
}
BOX_TOKENS = {"b", "f", "M0", "M1", "M2", "M3", "M4"}
HAZARD_TOKENS = {"pr", "pl", "pu", "pd"}             # punchers
FLOOR_TOKENS = {
    ".", "i", "l", "L", "o", "e", "g", "cp",
    "Sr", "Sl", "Su", "Sd", "Sr#", "Sl#", "Su#", "Sd#",
    "SrO", "SlO", "SuO", "SdO",
    "c0", "c1", "c2",
}
PLAYER_TOKEN = "p"
GEM_TOKEN = "G"


@dataclass
class ParsedCell:
    kind: str
    elevation: int
    has_gem: bool = False
    has_box: bool = False
    is_spawn: bool = False
    is_hazard: bool = False


def parse_cell(token: str) -> ParsedCell:
    """One space-separated cell token -> a flattened 2D cell."""
    raw = token.strip()
    if not raw:
        return ParsedCell(kind=HOLE, elevation=0)

    parts = raw.split("+")
    elevation = len(parts) - 1
    base = parts[0]
    stack = [p for p in parts if p]
    top = stack[-1] if stack else ""

    has_gem = GEM_TOKEN in stack
    has_box = any(p in BOX_TOKENS for p in stack)
    is_spawn = PLAYER_TOKEN in stack
    is_hazard = any(p in HAZARD_TOKENS for p in stack)

    if not base and not stack:
        kind = HOLE
    elif top in WALL_TOKENS:
        kind = "wall"
    elif has_box:
        kind = FLOOR            # the box itself is an actor laid on floor
    elif top in FLOOR_TOKENS or top in (PLAYER_TOKEN, GEM_TOKEN):
        kind = FLOOR
    elif base == "" :
        kind = HOLE
    else:
        kind = FLOOR            # unknown token: assume traversable, not a wall

    return ParsedCell(
        kind=kind, elevation=elevation, has_gem=has_gem,
        has_box=has_box, is_spawn=is_spawn, is_hazard=is_hazard,
    )


def parse_level(text: str, room_id: str = "level_AxA") -> tuple[Room, dict]:
    """A level file's contents -> a ``Room`` the mock can run, plus statistics."""
    rows = [line for line in text.split("\n") if line.strip()]
    if not rows:
        raise ValueError("empty level file")

    grid = [[parse_cell(tok) for tok in row.split(" ")] for row in rows]
    height = len(grid)
    width = max(len(r) for r in grid)
    for row in grid:                                  # pad ragged rows with void
        row.extend([ParsedCell(kind=HOLE, elevation=0)] * (width - len(row)))

    terrain = [[FLOOR] * width for _ in range(height)]
    heights = [[0] * width for _ in range(height)]
    gems: set[tuple[int, int]] = set()
    boxes: set[tuple[int, int]] = set()
    spawn: tuple[int, int] | None = None

    for r in range(height):
        for c in range(width):
            cell = grid[r][c]
            heights[r][c] = cell.elevation
            if cell.is_hazard:
                # Punchers really do kill.
                terrain[r][c] = HOLE
            elif cell.kind == HOLE:
                # Void is *impassable*, not lethal. `level_parsing.json` has no
                # hole token at all -- a cell with no floor is simply outside
                # the playable area, and a 16x16 level is mostly that. Treating
                # it as lethal made a random walk die within a few steps.
                terrain[r][c] = "wall"
            else:
                # A stacked `#` column is ground at that elevation, not a wall.
                # Whether it can be entered is decided by the height difference
                # at move time, so walls emerge from the terrain.
                terrain[r][c] = FLOOR
            if cell.has_gem:
                gems.add((r, c))
            if cell.has_box:
                boxes.add((r, c))
            if cell.is_spawn and spawn is None:
                spawn = (r, c)

    if spawn is None:
        candidates = [
            (r, c) for r in range(height) for c in range(width)
            if terrain[r][c] == FLOOR and (r, c) not in boxes
        ]
        spawn = candidates[len(candidates) // 2] if candidates else (0, 0)

    room = Room(room_id=room_id, terrain=terrain, gems=gems, boxes=boxes,
                spawn=spawn, heights=heights)
    stats = {
        "size": (height, width),
        "gems": len(gems),
        "boxes": len(boxes),
        "walls": sum(row.count("wall") for row in terrain),
        "max_height": max(max(row) for row in heights),
        "holes": sum(row.count(HOLE) for row in terrain),
        "has_explicit_spawn": any(cell.is_spawn for row in grid for cell in row),
    }
    return room, stats


def find_levels_dir(repo_root: str | Path | None = None) -> Path:
    candidates = []
    if repo_root:
        candidates.append(Path(repo_root) / "games" / "maze" / "levels")
    candidates += [
        Path.home() / "Documents" / "mazebench_src" / "repo" / "games" / "maze" / "levels",
    ]
    import os

    env_repo = os.environ.get("MAZEBENCH_REPO")
    if env_repo:
        candidates.insert(0, Path(env_repo) / "games" / "maze" / "levels")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    raise FileNotFoundError(
        "No MazeBench levels found. Clone MazeBenchEngine and set MAZEBENCH_REPO."
    )


@lru_cache(maxsize=1)
def list_levels(repo_root: str | None = None) -> tuple[Path, ...]:
    return tuple(sorted(find_levels_dir(repo_root).glob("*.txt")))


def load_level(path: str | Path, room_id: str = "level_AxA") -> tuple[Room, dict]:
    return parse_level(Path(path).read_text(encoding="utf8"), room_id=room_id)


def load_world_map(repo_root: str | Path | None = None) -> dict[str, str]:
    """``world_map.json`` -> {``level_HxI``: filename}.

    This is the real topology: a 16x16 grid of rooms, A..P on each axis, one
    level file per cell, all 256 placed with no duplicates. Using it means the
    mock's room adjacency *is* MazeBench's, so walking east from ``level_HxI``
    arrives where it really would.
    """
    levels_dir = find_levels_dir(repo_root)
    world_map_path = levels_dir.parent / "world_map.json"
    if not world_map_path.is_file():
        return {}
    import json

    placed = json.loads(world_map_path.read_text(encoding="utf8")).get("levels", {})
    out: dict[str, str] = {}
    for filename, coords in placed.items():
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            out[f"level_{coords[0]}x{coords[1]}"] = filename
    return out


class RealLevelSource:
    """Serves real levels to ``MockMazeBridge`` in place of generated rooms.

    Prefers ``world_map.json``, so the world has MazeBench's own layout and
    adjacency. Falls back to a deterministic seeded shuffle only if the map is
    missing, in which case geometry is real but neighbours are not.
    """

    def __init__(self, seed: int = 0, repo_root: str | None = None, limit: int | None = None):
        paths = list(list_levels(repo_root))
        if not paths:
            raise FileNotFoundError("no level files found")
        self.by_room = load_world_map(repo_root)
        self.levels_dir = find_levels_dir(repo_root)
        self.authentic_layout = bool(self.by_room)
        rng = random.Random(seed)
        shuffled = list(paths)
        rng.shuffle(shuffled)
        self.paths = shuffled[:limit] if limit else shuffled
        self.seed = seed
        self._cache: dict[str, Room] = {}

    def __len__(self) -> int:
        return len(self.by_room) or len(self.paths)

    def room_for(self, room_id: str) -> Room:
        if room_id not in self._cache:
            filename = self.by_room.get(room_id)
            if filename and (self.levels_dir / filename).is_file():
                path = self.levels_dir / filename
            else:
                index = (hash((self.seed, room_id)) % len(self.paths) + len(self.paths)) % len(self.paths)
                path = self.paths[index]
            room, _ = load_level(path, room_id=room_id)
            self._cache[room_id] = room
        return self._cache[room_id]

    def rooms_with_gems(self) -> list[str]:
        """Room ids holding at least one gem.

        Only 90 gems exist across 256 rooms, so an episode that visits two or
        three rooms almost never meets one -- measured 0 gems across 5 full
        episodes. Starting collection in gem-bearing rooms is what gives the
        gem head anything to learn from. Collection only; evaluation still
        starts where the benchmark says.
        """
        out = []
        for room_id, filename in self.by_room.items():
            path = self.levels_dir / filename
            if path.is_file() and load_level(path)[1]["gems"] > 0:
                out.append(room_id)
        return sorted(out)

    def total_gems(self) -> int:
        """Gems existing in the whole placed world.

        For the record: this is **90**, against a win condition of 100. The
        official threshold is not reachable from the shipped level set, which
        fits the reported ~10-13% best scores being ~10-13 gems out of 90.
        """
        total = 0
        for filename in (self.by_room.values() or [p.name for p in self.paths]):
            path = self.levels_dir / filename
            if not path.is_file():
                continue
            _, stats = load_level(path)
            total += stats["gems"]
        return total
