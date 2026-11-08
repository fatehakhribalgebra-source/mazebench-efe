"""MazeBench ASCII glyph vocabulary.

Transcribed from the engine's own contract at
``shared/maze-observation-contract.js`` (MazeBenchEngine, MIT).

Every tile in an ASCII observation is exactly one Unicode code point. Each
terrain/actor type has a ``top`` glyph (seen from above) and a ``side`` glyph
(seen from the side), selected by the camera pitch.

Two facts drive the agent design:

* ``P``/``p`` (player) and ``G``/``g`` (gem) are FIXED glyphs. The engine's
  ``hide_names`` mode permutes every other symbol per rollout but deliberately
  leaves these four alone. So a gem-seeking explorer keeps working under glyph
  permutation, while box/terrain identity has to be learned.
* Everything else is a stable permutation, which a learned embedding adapts to.
"""

from __future__ import annotations

# --- canonical glyph pairs, (top, side) -------------------------------------

TERRAIN = {
    "block_asset": ("&", "7"),
    "empty": (" ", " "),
    "exit": ("E", "e"),
    "floor": ("A", "a"),
    "hole": ("H", "h"),
    "ice": ("I", "i"),
    "ice_block": ("K", "k"),
    "ice_slope": ("~", "-"),
    "orange_wall": ("O", "o"),
    "player_gate": ("Y", "y"),
    "shrub": ("S", "s"),
    "tree": ("T", "t"),
    "wall": ("W", "w"),
}

PLAYER_LIFT = {"lowered": (">", "l"), "raised": ("L", "l")}
ORANGE_BUTTON = {"orange_button": ("8", " ")}
BLOCK_ASSET = {"1": ("!", "1"), "2": ("@", "2"), "3": ("#", "3"), "4": ("$", "4")}

ICE_SLOPE_DIR = {"down": ("V", "v"), "left": ("<", ","), "right": ("R", "r"), "up": ("^", "6")}
BLACK_ICE_SLOPE_DIR = {
    "down": ("▼", "▽"),
    "left": ("◀", "◁"),
    "right": ("▶", "▷"),
    "up": ("▲", "△"),
}
ORANGE_ICE_SLOPE_DIR = {
    "down": ("↓", "⇩"),
    "left": ("←", "⇦"),
    "right": ("→", "⇨"),
    "up": ("↑", "⇧"),
}

ACTORS = {
    "box": ("B", "b"),
    "clone": ("{", "["),
    "floating_floor": ("F", "f"),
    "gem": ("G", "g"),
    "player": ("P", "p"),
    "puncher": ("}", "]"),
    "weightless_box": (";", "_"),
}

CLONES = {"c0": ("C", "c"), "c1": ("D", "d"), "c2": ("J", "j")}
WEIGHTLESS_BOXES = {
    "M0": ("U", "u"),
    "M1": ("0", "9"),
    "M2": ("(", ")"),
    "M3": ("+", "="),
    "M4": (".", ":"),
}
PUNCHER_DIR = {"down": ("%", "5"), "left": ("X", "x"), "right": ("Q", "q"), "up": ("Z", "z")}
UNKNOWN = {"actor": ("|", "\\"), "terrain": ("`", "'")}

_ALL_GROUPS = (
    TERRAIN, PLAYER_LIFT, ORANGE_BUTTON, BLOCK_ASSET, ICE_SLOPE_DIR,
    BLACK_ICE_SLOPE_DIR, ORANGE_ICE_SLOPE_DIR, ACTORS, CLONES,
    WEIGHTLESS_BOXES, PUNCHER_DIR, UNKNOWN,
)


def _collect() -> list[str]:
    seen: set[str] = set()
    for group in _ALL_GROUPS:
        for pair in group.values():
            seen.update(pair)
    # "?" is the extra target the engine's hidden-glyph permutation can emit.
    seen.add("?")
    seen.discard(" ")
    return sorted(seen)


# --- vocabulary --------------------------------------------------------------
# id 0 = padding (outside the board), id 1 = empty space, id 2 = out-of-vocab.

PAD_ID = 0
EMPTY_ID = 1
OOV_ID = 2
_FIRST = 3

CANONICAL_GLYPHS = _collect()
GLYPH_TO_ID: dict[str, int] = {" ": EMPTY_ID}
for _i, _g in enumerate(CANONICAL_GLYPHS):
    GLYPH_TO_ID[_g] = _FIRST + _i
ID_TO_GLYPH = {v: k for k, v in GLYPH_TO_ID.items()}
VOCAB_SIZE = _FIRST + len(CANONICAL_GLYPHS)


def glyph_id(ch: str) -> int:
    return GLYPH_TO_ID.get(ch, OOV_ID)


# --- semantic sets -----------------------------------------------------------
# Only PLAYER_GLYPHS and GEM_GLYPHS are guaranteed stable under hide_names.
# The rest are a convenience for the canonical palette and for the mock env;
# the learned model never depends on them.

PLAYER_GLYPHS = frozenset(ACTORS["player"])
GEM_GLYPHS = frozenset(ACTORS["gem"])

PUSHABLE_GLYPHS = frozenset(
    set(ACTORS["box"])
    | set(ACTORS["floating_floor"])
    | set(ACTORS["weightless_box"])
    | set(ACTORS["clone"])
    | {g for pair in CLONES.values() for g in pair}
    | {g for pair in WEIGHTLESS_BOXES.values() for g in pair}
)

SOLID_GLYPHS = frozenset(
    set(TERRAIN["wall"]) | set(TERRAIN["tree"]) | set(TERRAIN["shrub"])
    | set(TERRAIN["orange_wall"]) | set(TERRAIN["ice_block"]) | set(TERRAIN["block_asset"])
    | {g for pair in BLOCK_ASSET.values() for g in pair}
)

HAZARD_GLYPHS = frozenset(
    set(TERRAIN["hole"]) | set(ACTORS["puncher"])
    | {g for pair in PUNCHER_DIR.values() for g in pair}
)

WALKABLE_GLYPHS = frozenset(
    set(TERRAIN["floor"]) | set(TERRAIN["exit"]) | set(TERRAIN["ice"])
    | set(ORANGE_BUTTON["orange_button"]) | GEM_GLYPHS | PLAYER_GLYPHS
)

PLAYER_IDS = frozenset(glyph_id(g) for g in PLAYER_GLYPHS)
GEM_IDS = frozenset(glyph_id(g) for g in GEM_GLYPHS)
PUSHABLE_IDS = frozenset(glyph_id(g) for g in PUSHABLE_GLYPHS)
SOLID_IDS = frozenset(glyph_id(g) for g in SOLID_GLYPHS)
HAZARD_IDS = frozenset(glyph_id(g) for g in HAZARD_GLYPHS)

__all__ = [
    "VOCAB_SIZE", "PAD_ID", "EMPTY_ID", "OOV_ID", "GLYPH_TO_ID", "ID_TO_GLYPH",
    "glyph_id", "CANONICAL_GLYPHS", "PLAYER_GLYPHS", "GEM_GLYPHS",
    "PUSHABLE_GLYPHS", "SOLID_GLYPHS", "HAZARD_GLYPHS", "WALKABLE_GLYPHS",
    "PLAYER_IDS", "GEM_IDS", "PUSHABLE_IDS", "SOLID_IDS", "HAZARD_IDS",
]
