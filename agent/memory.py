"""Persistent episodic memory.

MazeBench's world persists across rooms for the whole rollout, so memory is not
an optimisation here -- it is the thing being tested. The research question this
project narrowed to is whether persistent memory plus an action-conditioned
world model beats a memoryless baseline at reward-free exploration, and this
class is the "memory" half of that comparison.

Everything stored here is derived from agent-visible observations and the
agent's own actions. The engine's authoritative ``board_state_hash`` would be a
better novelty key and is right there in the snapshot -- it is deliberately not
used, because it is an evaluator-only field.

Three levels of novelty, matching the three timescales the planner works at:

``cell_counts``    (room, estimated position) -> visit count. Fine-grained,
                   needs dead reckoning, and is what drives local exploration.
``pattern_counts`` hash of the egocentric window -> count. Translation
                   invariant, so it generalises across rooms, and it can be
                   evaluated on an *imagined* decoded window, which is what
                   makes it usable inside a planning rollout.
``rooms``          per-room statistics and a frontier score, for the
                   room-level ``goto_level`` decision.

Dead reckoning: the agent is never told its coordinates. It integrates its own
intended moves, and cancels the integration when the percept says the move was
blocked (an identical window). Drift is real -- a room transition resets the
estimate -- which is why pattern novelty is kept as an independent signal.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


def window_hash(window: np.ndarray) -> str:
    """Stable hash of a glyph window. Works on real and imagined windows."""
    return hashlib.blake2b(np.ascontiguousarray(window, dtype=np.int16).tobytes(), digest_size=8).hexdigest()


# screen-relative move deltas in (row, col); yaw rotation applied by the caller
MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def rotate_delta(delta: tuple[int, int], yaw: int) -> tuple[int, int]:
    dr, dc = delta
    for _ in range(yaw % 4):
        dr, dc = -dc, dr
    return dr, dc


@dataclass
class RoomRecord:
    room_id: str
    first_seen_step: int = 0
    visits: int = 0
    steps_spent: int = 0
    distinct_patterns: set[str] = field(default_factory=set)
    distinct_cells: set[tuple[int, int]] = field(default_factory=set)
    gems_collected_here: int = 0
    gem_glyphs_last_seen: int = 0
    neighbours: set[str] = field(default_factory=set)
    exits_tried: set[int] = field(default_factory=set)
    deaths: int = 0

    @property
    def novelty_rate(self) -> float:
        """Fraction of time in this room that produced a new local pattern."""
        if self.steps_spent <= 0:
            return 1.0
        return len(self.distinct_patterns) / self.steps_spent


class EpisodicMemory:
    def __init__(self, novelty_scale: float = 1.0, n_abstract: int = 2048):
        self.cell_counts: dict[tuple[str, int, int], int] = defaultdict(int)
        self.pattern_counts: dict[str, int] = defaultdict(int)
        self.rooms: dict[str, RoomRecord] = {}
        self.novelty_scale = novelty_scale
        self.step = 0

        # dead-reckoned position belief, reset on room change
        self.est_pos: tuple[int, int] = (0, 0)
        self.current_room: str = ""
        self.yaw: int = 0

        # abstract (level-2) coverage memory, for the hierarchical terminal value
        self.abstract_states: dict[int, np.ndarray] = {}
        self._abstract_buffer: dict[int, list[np.ndarray]] = {}
        self.n_abstract = n_abstract

        # gems seen but not yet taken, per room, in agent-estimated coordinates
        self.known_gems: dict[str, set[tuple[int, int]]] = defaultdict(set)

        # box-configuration counts, keyed by (room, layout hash). The benchmark
        # rewards *novel block positions*, and repeating a layout does not farm
        # it -- so the agent needs its own count of layouts it has produced.
        self.box_counts: dict[tuple[str, int], int] = defaultdict(int)

    # -- observation ingest -------------------------------------------------
    def observe(
        self,
        room_id: str,
        window: np.ndarray,
        gem_count: int,
        yaw: int,
        moved: bool | None,
        last_action: int | None,
        died: bool = False,
    ) -> dict[str, float]:
        """Fold one real observation into memory. Returns novelty diagnostics."""
        self.step += 1
        room_changed = room_id != self.current_room

        if room_changed:
            self.current_room = room_id
            self.est_pos = (0, 0)      # fresh frame of reference per room
        elif last_action is not None and last_action < 4 and moved:
            dr, dc = rotate_delta(MOVE_DELTAS[last_action], self.yaw)
            self.est_pos = (self.est_pos[0] + dr, self.est_pos[1] + dc)
        self.yaw = yaw % 4

        record = self.rooms.get(room_id)
        if record is None:
            record = RoomRecord(room_id=room_id, first_seen_step=self.step)
            self.rooms[room_id] = record
        record.visits += 1 if room_changed else 0
        record.steps_spent += 1
        if died:
            record.deaths += 1

        pattern = window_hash(window)
        was_new_pattern = self.pattern_counts[pattern] == 0
        was_new_cell = self.cell_counts[(room_id, *self.est_pos)] == 0

        self.pattern_counts[pattern] += 1
        self.cell_counts[(room_id, *self.est_pos)] += 1
        record.distinct_patterns.add(pattern)
        record.distinct_cells.add(self.est_pos)

        return {
            "new_pattern": float(was_new_pattern),
            "new_cell": float(was_new_cell),
            "room_changed": float(room_changed),
            "pattern_novelty": self.pattern_novelty(pattern),
            "cell_novelty": self.cell_novelty(room_id, self.est_pos),
        }

    def note_room_link(self, from_room: str, to_room: str, action: int | None) -> None:
        if from_room and to_room and from_room != to_room:
            self.rooms.setdefault(from_room, RoomRecord(from_room)).neighbours.add(to_room)
            self.rooms.setdefault(to_room, RoomRecord(to_room)).neighbours.add(from_room)
            if action is not None:
                self.rooms[from_room].exits_tried.add(action)

    # -- novelty ------------------------------------------------------------
    def pattern_novelty(self, pattern: str) -> float:
        """1/sqrt(1+n) count bonus. 1.0 for unseen, decaying with revisits."""
        return self.novelty_scale / math.sqrt(1.0 + self.pattern_counts.get(pattern, 0))

    def cell_novelty(self, room_id: str, pos: tuple[int, int]) -> float:
        return self.novelty_scale / math.sqrt(1.0 + self.cell_counts.get((room_id, *pos), 0))

    def novelty_of_imagined(self, room_id: str, pos: tuple[int, int], window: np.ndarray) -> float:
        """Score an imagined state. Uses only keys computable without the engine."""
        return 0.5 * self.cell_novelty(room_id, pos) + 0.5 * self.pattern_novelty(window_hash(window))

    # -- box layouts --------------------------------------------------------
    def box_novelty(self, room_id: str, layout_key: int) -> float:
        """Novelty of a box layout the agent has pushed blocks into.

        ``block_progress_score`` rewards reaching *novel* block positions, and
        explicitly does not pay for repeating a layout. So this is a count
        bonus over layouts, not a bonus for pushing.
        """
        return self.novelty_scale / math.sqrt(1.0 + self.box_counts.get((room_id, int(layout_key)), 0))

    def record_box_layout(self, room_id: str, layout_key: int) -> bool:
        """Fold a real layout into memory. Returns True if it was new."""
        key = (room_id, int(layout_key))
        was_new = self.box_counts[key] == 0
        self.box_counts[key] += 1
        return was_new

    # -- abstract coverage, one store per abstract level --------------------
    # Each level sees the world at its own timescale, so their coverage is not
    # interchangeable: level 2 distinguishes places a few steps apart, level 4
    # distinguishes whole regions. Mixing them into one store would let the
    # coarsest level's distances swamp the others.
    def add_abstract(self, state: np.ndarray, level: int = 0) -> None:
        self._abstract_buffer.setdefault(level, []).append(
            np.asarray(state, dtype=np.float32).ravel()
        )
        if len(self._abstract_buffer[level]) >= 32:
            self._flush_abstract(level)

    def add_abstract_all(self, states: list[np.ndarray]) -> None:
        for level, state in enumerate(states):
            self.add_abstract(state, level=level)

    def _flush_abstract(self, level: int | None = None) -> None:
        levels = [level] if level is not None else list(self._abstract_buffer)
        for lvl in levels:
            buffered = self._abstract_buffer.get(lvl)
            if not buffered:
                continue
            new = np.stack(buffered)
            buffered.clear()
            existing = self.abstract_states.get(lvl)
            merged = new if existing is None else np.concatenate([existing, new])
            if len(merged) > self.n_abstract:
                idx = np.random.choice(len(merged), self.n_abstract, replace=False)
                merged = merged[idx]
            self.abstract_states[lvl] = merged

    def abstract_novelty(self, state: np.ndarray, level: int = 0) -> np.ndarray:
        """Distance to the nearest visited abstract state at ``level``.

        This is the hierarchy's contribution to exploration: a continuous
        novelty measure in a coarse latent space, scoring *where a long action
        chunk lands* without simulating every primitive step. A deeper level
        scores where a much longer stretch lands, which is the whole argument
        for adding one.
        """
        self._flush_abstract(level)
        query = np.atleast_2d(np.asarray(state, dtype=np.float32))
        seen = self.abstract_states.get(level)
        if seen is None or len(seen) == 0:
            return np.ones(len(query), dtype=np.float32)
        d = np.linalg.norm(query[:, None, :] - seen[None, :, :], axis=-1)
        return d.min(axis=1)

    # -- room-level frontier ------------------------------------------------
    def frontier_scores(self, exclude: str | None = None) -> dict[str, float]:
        """How promising each visited room is to return to.

        Rewards rooms that still yield new patterns, that have unexplored exits,
        and that have not been camped in. This is what the supervisor consults
        when it decides the current room is exhausted and issues ``goto_level``.
        """
        scores: dict[str, float] = {}
        for room_id, record in self.rooms.items():
            if room_id == exclude:
                continue
            unexplored_exits = max(0, 4 - len(record.exits_tried))
            scores[room_id] = (
                2.0 * record.novelty_rate
                + 0.5 * unexplored_exits
                + 1.0 / math.sqrt(1.0 + record.steps_spent)
                - 0.5 * record.deaths
            )
        return scores

    def best_frontier_room(self, exclude: str | None = None) -> tuple[str, float] | None:
        scores = self.frontier_scores(exclude=exclude)
        if not scores:
            return None
        best = max(scores.items(), key=lambda kv: kv[1])
        return best

    # -- diagnostics --------------------------------------------------------
    def summary(self) -> dict[str, float]:
        return {
            "steps": self.step,
            "rooms_known": len(self.rooms),
            "distinct_patterns": len(self.pattern_counts),
            "distinct_cells": len(self.cell_counts),
            "pattern_novelty_rate": len(self.pattern_counts) / max(1, self.step),
            "abstract_states": sum(len(v) for v in self.abstract_states.values()),
            "abstract_levels": len(self.abstract_states),
        }
