"""Action space.

Split deliberately into two tiers.

**Plannable primitives (8)** -- the only actions the CEM planner searches over.
Four moves plus four camera rotations. Camera rotation is in here on purpose:
it cannot change world state, so it has zero pragmatic value, but it *does*
reveal occluded tiles and change the top/side glyph projection. That makes it
the cleanest possible test of whether the EFE epistemic term is doing real
work -- an agent that never rotates is ignoring free information, and one that
only rotates has collapsed into novelty-seeking.

**Control actions** -- ``undo``, ``reset_level``, ``goto_level``, ``quit``.
These are issued by a supervisor (``agent.supervisor``), never searched. Reason:
``undo``/``reset`` are history-dependent, so a Markov latent world model cannot
represent them honestly, and letting the planner treat ``reset`` as an ordinary
move lets it farm a reset loop. Keeping them out is also what this project
concluded the first time round: reset is an episode-control decision for a
stuck detector, not a movement.
"""

from __future__ import annotations

from typing import Iterator

MOVES = ("up", "down", "left", "right")
ROTATIONS = ("up", "down", "left", "right")

# index -> (command, kwargs)
PLANNABLE: tuple[tuple[str, dict], ...] = tuple(
    [("move", {"direction": d}) for d in MOVES]
    + [("rotate_camera", {"direction": d}) for d in ROTATIONS]
)
N_ACTIONS = len(PLANNABLE)

MOVE_IDS = tuple(range(0, 4))
ROTATE_IDS = tuple(range(4, 8))

ACTION_NAMES = tuple(
    f"{'move' if i < 4 else 'rotate'}_{PLANNABLE[i][1]['direction']}" for i in range(N_ACTIONS)
)

CONTROL_ACTIONS = ("undo", "reset_level", "goto_level", "quit", "no_move")


def is_move(action: int) -> bool:
    return 0 <= action < 4


def is_rotation(action: int) -> bool:
    return 4 <= action < 8


def describe(action: int) -> str:
    return ACTION_NAMES[action]


def iter_actions() -> Iterator[int]:
    return iter(range(N_ACTIONS))
