"""Turn an agent-visible MazeBench observation into tensors.

Strict rule enforced here: this module only ever reads fields the benchmark
shows the model. See ``AGENT_VISIBLE_FIELDS``. Privileged engine fields
(``board_state_hash``, ``player`` coordinates, ``collected_gems`` ids,
``push_count``) are dropped by ``env.mazebench_env.ObservationFirewall`` before
anything reaches this code, so they cannot leak into the policy by accident.

The egocentric window is centred on the player glyph, which the agent locates
itself by scanning its own ASCII board for ``P``/``p`` -- a fixed glyph the
engine never permutes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .glyphs import EMPTY_ID, PAD_ID, PLAYER_GLYPHS, glyph_id

AGENT_VISIBLE_FIELDS = frozenset({
    "level", "ascii_legend", "gem_count", "current_room", "current_view",
    "yaw", "visited_levels", "allowed_commands", "player_dead",
    "death_message", "action_count", "game_won", "json_observation",
})


def parse_board(level: str) -> np.ndarray:
    """ASCII board string -> (H, W) int array of glyph ids, right-padded."""
    lines = (level or "").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return np.full((1, 1), EMPTY_ID, dtype=np.int64)
    width = max(len(line) for line in lines)
    grid = np.full((len(lines), width), EMPTY_ID, dtype=np.int64)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            grid[r, c] = glyph_id(ch)
    return grid


_PLAYER_ID_SET = {glyph_id(g) for g in PLAYER_GLYPHS}


def find_player(grid: np.ndarray) -> tuple[int, int] | None:
    """Locate the player glyph. Returns (row, col) or None if off-screen."""
    for pid in _PLAYER_ID_SET:
        hits = np.argwhere(grid == pid)
        if len(hits):
            r, c = hits[0]
            return int(r), int(c)
    return None


#: MazeBench rooms are exactly 16x16 tiles in all 256 shipped levels. An
#: egocentric window only covers a whole room from any standing position if it
#: is 2*16-1 = 31 wide; 15 is a deliberate speed/coverage trade for local
#: iteration. See README "Window sizing".
ROOM_TILES = 16
FULL_ROOM_WINDOW = 2 * ROOM_TILES - 1


def egocentric_window(grid: np.ndarray, size: int, centre: tuple[int, int] | None = None) -> np.ndarray:
    """Crop a (size, size) window centred on the player, PAD outside the board.

    Padding is a distinct token, so the model can tell "wall" from "edge of
    what I can see".
    """
    if centre is None:
        centre = find_player(grid)
    if centre is None:
        centre = (grid.shape[0] // 2, grid.shape[1] // 2)
    half = size // 2
    out = np.full((size, size), PAD_ID, dtype=np.int64)
    r0, c0 = centre[0] - half, centre[1] - half
    for i in range(size):
        rr = r0 + i
        if rr < 0 or rr >= grid.shape[0]:
            continue
        cc0 = max(0, c0)
        cc1 = min(grid.shape[1], c0 + size)
        if cc1 <= cc0:
            continue
        out[i, cc0 - c0 : cc1 - c0] = grid[rr, cc0:cc1]
    return out


def room_to_vec(room_id: str) -> np.ndarray:
    """``level_HxI`` -> two normalised coordinates in [0, 1].

    Room ids are a letter pair, so this is a legitimate reading of a public
    field, not a hidden coordinate.
    """
    out = np.zeros(2, dtype=np.float32)
    raw = (room_id or "").replace("level_", "")
    if "x" in raw:
        a, _, b = raw.partition("x")
        if len(a) == 1 and a.isalpha():
            out[0] = (ord(a.upper()) - 65) / 25.0
        if len(b) == 1 and b.isalpha():
            out[1] = (ord(b.upper()) - 65) / 25.0
    return out


VIEW_NAMES = ["top", "top-diagonal", "diagonal", "side-diagonal", "side"]
SCALAR_DIM = 2 + 5 + 4 + 4  # room xy, view one-hot, yaw one-hot, misc


@dataclass
class EncodedObs:
    """One encoded timestep."""

    window: np.ndarray                 # (size, size) int64 glyph ids
    scalars: np.ndarray                # (SCALAR_DIM,) float32
    gem_count: int
    room_id: str
    obs_hash: str                      # agent's OWN hash, for novelty
    n_visited: int
    player_found: bool
    raw_grid: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1), np.int64))


def encode(obs: dict, window_size: int) -> EncodedObs:
    """Encode an agent-visible observation dict."""
    grid = parse_board(obs.get("level", ""))
    centre = find_player(grid)
    window = egocentric_window(grid, window_size, centre)

    view = str(obs.get("current_view") or "top-diagonal")
    yaw = int(obs.get("yaw") or 0) % 4
    visited = obs.get("visited_levels") or []
    gem_count = int(obs.get("gem_count") or 0)

    scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
    scalars[0:2] = room_to_vec(str(obs.get("current_room") or ""))
    if view in VIEW_NAMES:
        scalars[2 + VIEW_NAMES.index(view)] = 1.0
    scalars[7 + yaw] = 1.0
    scalars[11] = min(gem_count, 100) / 100.0
    scalars[12] = min(len(visited), 200) / 200.0
    scalars[13] = 1.0 if obs.get("player_dead") else 0.0
    scalars[14] = 1.0 if centre is not None else 0.0

    return EncodedObs(
        window=window,
        scalars=scalars,
        gem_count=gem_count,
        room_id=str(obs.get("current_room") or ""),
        obs_hash=hash_observation(obs),
        n_visited=len(visited),
        player_found=centre is not None,
        raw_grid=grid,
    )


def hash_observation(obs: dict) -> str:
    """Novelty key built ONLY from what the agent can see.

    The engine exposes an authoritative ``board_state_hash``, but that is an
    evaluator-only field. Using it as an exploration signal would be privileged
    information, so the agent hashes its own percept instead: the ASCII board
    plus the room it believes it is in. Coarser than the engine's hash (it
    cannot see behind itself, and camera pitch changes the glyphs) but honest.
    """
    board = str(obs.get("level") or "")
    room = str(obs.get("current_room") or "")
    view = str(obs.get("current_view") or "")
    yaw = str(obs.get("yaw") or 0)
    payload = f"{room}|{view}|{yaw}|{board}".encode("utf8", "replace")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def room_signature(obs: dict) -> str:
    """Novelty key for a room, ignoring camera. Used for room-level frontier."""
    board = str(obs.get("level") or "")
    room = str(obs.get("current_room") or "")
    return hashlib.blake2b(f"{room}|{board}".encode("utf8", "replace"), digest_size=8).hexdigest()
