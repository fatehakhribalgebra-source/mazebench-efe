"""Tests that need no Node runtime and no GPU.

Run with ``pytest tests`` or ``python tests/test_all.py``.

The load-bearing ones:

``test_firewall_never_leaks``        the whole scientific validity of the project
``test_movement_is_screen_relative`` the assumption the safety mask rests on
``test_latent_does_not_collapse``    the thing JEPA-style training gets wrong
``test_planner_respects_mask``       that safety actually binds
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.efe import EFE, EFEConfig, advance_positions, advance_yaws  # noqa: E402
from agent.memory import MOVE_DELTAS, EpisodicMemory, window_hash  # noqa: E402
from agent.planner import CEMPlanner, PlannerConfig  # noqa: E402
from env import glyphs  # noqa: E402
from env.actions import ACTION_NAMES, N_ACTIONS, PLANNABLE  # noqa: E402
from env.mazebench_env import PRIVILEGED_FIELDS, MazeBenchEnv, split_observation  # noqa: E402
from env.mock import MockMazeBridge  # noqa: E402
from env.obs import (AGENT_VISIBLE_FIELDS, egocentric_window, encode,  # noqa: E402
                     find_player, parse_board)
from model.world_model import HJEPA, WorldModelConfig  # noqa: E402


# ------------------------------------------------------------------- contract
def test_glyph_vocab_is_injective():
    ids = [glyphs.glyph_id(g) for g in glyphs.CANONICAL_GLYPHS]
    assert len(set(ids)) == len(ids), "two glyphs share an id"
    assert glyphs.VOCAB_SIZE == 3 + len(glyphs.CANONICAL_GLYPHS)
    for special in (glyphs.PAD_ID, glyphs.EMPTY_ID, glyphs.OOV_ID):
        assert special not in ids


def test_fixed_glyphs_are_player_and_gem():
    # The engine permutes every glyph except these under hide_names, which is
    # why the agent can rely on finding itself and on seeing gems.
    assert glyphs.PLAYER_GLYPHS == {"P", "p"}
    assert glyphs.GEM_GLYPHS == {"G", "g"}
    assert glyphs.PLAYER_IDS.isdisjoint(glyphs.GEM_IDS)


def test_parse_and_centre_window():
    board = "WWWWW\nW P W\nW G W\nWWWWW"
    grid = parse_board(board)
    assert grid.shape == (4, 5)
    centre = find_player(grid)
    assert centre == (1, 2)
    window = egocentric_window(grid, 7, centre)
    assert window.shape == (7, 7)
    assert int(window[3, 3]) == glyphs.glyph_id("P"), "player must sit at the centre"
    assert int(window[0, 0]) == glyphs.PAD_ID, "outside the board must be PAD, not empty"


def test_ragged_board_is_padded_not_truncated():
    grid = parse_board("WWW\nW\nWWWWW")
    assert grid.shape == (3, 5)
    assert int(grid[1, 4]) == glyphs.EMPTY_ID


# ------------------------------------------------------------------- firewall
def test_firewall_never_leaks():
    bridge = MockMazeBridge(seed=0)
    snapshot = bridge.observe()
    for field in ("board_state_hash", "player", "collected_gems", "push_count"):
        assert field in snapshot, f"mock must emit {field} for the firewall to have work to do"

    visible, privileged = split_observation(snapshot)
    leaked = PRIVILEGED_FIELDS & set(visible)
    assert not leaked, f"privileged fields leaked to the agent: {leaked}"
    assert set(visible) <= AGENT_VISIBLE_FIELDS
    assert "level" in visible and "gem_count" in visible
    assert "board_state_hash" in privileged


def test_env_step_returns_only_visible_obs():
    env = MazeBenchEnv(bridge=MockMazeBridge(seed=0), max_actions=20)
    obs = env.reset()
    assert not (PRIVILEGED_FIELDS & set(obs))
    obs, _, privileged = env.step(4)        # a camera rotation: always legal
    assert not (PRIVILEGED_FIELDS & set(obs))
    assert "board_state_hash" in privileged


# ------------------------------------------------------------------------ env
def test_movement_is_screen_relative():
    """After rotating the camera, moves stay relative to the *screen*.

    The safety mask reads the cell above the window centre for "up". That is
    only correct if the rendered board is in screen space, so the rendered
    displacement of the player must equal the raw delta with no yaw
    correction. If this test fails the mask is masking the wrong direction.
    """
    for yaw_turns in range(4):
        bridge = MockMazeBridge(seed=5)
        for _ in range(yaw_turns):
            bridge.call("rotate_camera", direction="right")

        checked = 0
        for action in range(4):
            before = bridge.observe()
            grid_before = parse_board(before["level"])
            pos_before = find_player(grid_before)
            result = bridge.call("move", direction=PLANNABLE[action][1]["direction"])
            if result.get("room_changed") or result.get("player_dead"):
                bridge.call("undo")
                continue
            grid_after = parse_board(result["level"])
            pos_after = find_player(grid_after)
            if pos_before is None or pos_after is None:
                bridge.call("undo")
                continue
            if pos_after != pos_before:
                delta = (pos_after[0] - pos_before[0], pos_after[1] - pos_before[1])
                assert delta == MOVE_DELTAS[action], (
                    f"yaw={yaw_turns} {ACTION_NAMES[action]}: rendered delta {delta} "
                    f"!= screen delta {MOVE_DELTAS[action]}"
                )
                checked += 1
            bridge.call("undo")
        assert checked > 0, f"yaw={yaw_turns}: no move succeeded, test proved nothing"


def test_death_requires_recovery_and_keeps_gems():
    env = MazeBenchEnv(bridge=MockMazeBridge(seed=2), max_actions=400)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(400):
        if env.dead:
            break
        try:
            env.step(int(rng.integers(4)))
        except Exception:
            break
    if not env.dead:
        return                      # no hole reached; nothing to assert
    gems_before = env.log.gem_count
    env.recover()
    assert not env.dead, "recover() must clear death"
    assert env.log.gem_count >= gems_before, "gem score must stay monotonic"


def test_goto_level_rejects_unvisited():
    bridge = MockMazeBridge(seed=0)
    try:
        bridge.call("goto_level", x="Z", y="Z")
    except RuntimeError as error:
        assert "unvisited" in str(error)
    else:
        raise AssertionError("goto_level accepted an unvisited room")


# --------------------------------------------------------------------- memory
def test_novelty_decays_with_revisits():
    memory = EpisodicMemory()
    window = np.zeros((9, 9), dtype=np.int64)
    first = memory.pattern_novelty(window_hash(window))
    for _ in range(8):
        memory.observe("level_AxA", window, 0, 0, True, None)
    later = memory.pattern_novelty(window_hash(window))
    assert first == 1.0
    assert later < first, "revisiting a pattern must reduce its novelty"


def test_dead_reckoning_tracks_moves_and_blocks():
    memory = EpisodicMemory()
    window = np.zeros((9, 9), dtype=np.int64)
    memory.observe("level_AxA", window, 0, 0, None, None)
    assert memory.est_pos == (0, 0)
    memory.observe("level_AxA", window, 0, 0, True, 1)        # moved down
    assert memory.est_pos == (1, 0)
    memory.observe("level_AxA", window, 0, 0, False, 1)       # blocked: no change
    assert memory.est_pos == (1, 0)
    memory.observe("level_BxA", window, 0, 0, True, 1)        # new room resets frame
    assert memory.est_pos == (0, 0)


def test_frontier_prefers_rooms_with_novelty_left():
    memory = EpisodicMemory()
    window = np.zeros((9, 9), dtype=np.int64)
    for i in range(30):                                       # stale room
        w = window.copy()
        memory.observe("level_AxA", w, 0, 0, True, 0)
    for i in range(4):                                        # fresh room
        w = window.copy()
        w[0, i] = i + 5
        memory.observe("level_BxB", w, 0, 0, True, 0)
    scores = memory.frontier_scores()
    assert scores["level_BxB"] > scores["level_AxA"]


def test_advance_positions_only_moves_on_success():
    positions = np.zeros((3, 2), dtype=np.int64)
    actions = np.array([0, 0, 5])                 # up, up, rotate
    moved = np.array([0.9, 0.1, 0.9])
    yaws = np.zeros(3, dtype=np.int64)
    out = advance_positions(positions, actions, moved, yaws)
    assert tuple(out[0]) == (-1, 0), "a successful move must displace"
    assert tuple(out[1]) == (0, 0), "a predicted-blocked move must not"
    assert tuple(out[2]) == (0, 0), "a rotation must not displace"
    assert tuple(advance_yaws(np.zeros(1, dtype=np.int64), np.array([7]))) == (1,)


# ---------------------------------------------------------------------- model
def _toy_batch(batch: int = 8, size: int = 15) -> dict[str, torch.Tensor]:
    return {
        "window": torch.randint(0, glyphs.VOCAB_SIZE, (batch, size, size)),
        "scalars": torch.randn(batch, 15),
        "action": torch.randint(0, N_ACTIONS, (batch,)),
        "next_window": torch.randint(0, glyphs.VOCAB_SIZE, (batch, size, size)),
        "next_scalars": torch.randn(batch, 15),
        "gem_delta": torch.randint(0, 4, (batch,)),
        "room_change": torch.randint(0, 2, (batch,)).float(),
        "player_vis": torch.ones(batch),
        "death": torch.zeros(batch),
        "moved": torch.ones(batch),
    }


def test_model_shapes_and_backward():
    model = HJEPA(WorldModelConfig(n_members=2))
    batch = _toy_batch()
    loss, stats = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    for key in ("tile_acc", "changed_tile_acc", "gem_recall", "gem_precision"):
        assert key in stats
    z, h = model.encode(batch["window"], batch["scalars"])
    assert z.shape[0] == 8 and z.shape[2] == z.shape[3] == 15
    assert h.shape == (8, model.config.pooled_dim)


def test_predictor_starts_as_identity():
    # Zero-init output head: before training, imagining must be a no-op rather
    # than injecting noise into the planner.
    model = HJEPA(WorldModelConfig(n_members=1))
    z, _ = model.encode(_toy_batch()["window"], _toy_batch()["scalars"])
    predicted = model.ensemble.members[0](z, torch.zeros(z.shape[0], dtype=torch.long))
    assert torch.allclose(predicted, z, atol=1e-6)


def test_latent_does_not_collapse():
    """A few optimisation steps must not drive the latent to a constant."""
    torch.manual_seed(0)
    model = HJEPA(WorldModelConfig(n_members=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _toy_batch(batch=16)
    for _ in range(25):
        loss, _ = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_targets()
    with torch.no_grad():
        _, h = model.encode(batch["window"], batch["scalars"])
    spread = h.std(dim=0).mean().item()
    assert spread > 0.05, f"latent collapsed (mean per-dim std {spread:.4f})"


def test_ensemble_disagreement_is_nonnegative():
    model = HJEPA(WorldModelConfig(n_members=3))
    batch = _toy_batch()
    z, _ = model.encode(batch["window"], batch["scalars"])
    _, disagreement = model.ensemble.mean_and_disagreement(z, batch["action"])
    assert disagreement.shape == (8,)
    assert bool((disagreement >= 0).all())


# ------------------------------------------------------------------------ efe
def test_gem_proximity_peaks_on_the_player_cell():
    efe = EFE(EpisodicMemory(), EFEConfig())
    gem_id = glyphs.glyph_id("G")
    size = 15
    on_centre = torch.full((1, size, size), glyphs.EMPTY_ID, dtype=torch.long)
    on_centre[0, size // 2, size // 2] = gem_id
    far = torch.full((1, size, size), glyphs.EMPTY_ID, dtype=torch.long)
    far[0, 0, 0] = gem_id
    none = torch.full((1, size, size), glyphs.EMPTY_ID, dtype=torch.long)

    ids = torch.tensor(sorted(glyphs.GEM_IDS))
    near_score = float(efe.proximity(on_centre, ids, 8.0)[0])
    far_score = float(efe.proximity(far, ids, 8.0)[0])
    none_score = float(efe.proximity(none, ids, 8.0)[0])
    assert near_score == 1.0
    assert none_score == 0.0
    assert near_score > far_score >= none_score


def test_box_layout_key_tracks_block_positions():
    """Moving a block must change the layout key; moving the player must not.

    This is the 'imagine the cube on top of the diamond' term. It has to react
    to where blocks are and ignore everything else, or it just duplicates the
    pattern-novelty signal.
    """
    efe = EFE(EpisodicMemory(), EFEConfig())
    size = 15
    box_id = glyphs.glyph_id("B")
    player_id = glyphs.glyph_id("P")

    base = torch.full((1, size, size), glyphs.EMPTY_ID, dtype=torch.long)
    base[0, 4, 4] = box_id
    base[0, 7, 7] = player_id

    moved_box = base.clone()
    moved_box[0, 4, 4] = glyphs.EMPTY_ID
    moved_box[0, 4, 5] = box_id

    moved_player = base.clone()
    moved_player[0, 7, 7] = glyphs.EMPTY_ID
    moved_player[0, 8, 7] = player_id

    key_base = int(efe.box_layout_keys(base)[0])
    assert int(efe.box_layout_keys(moved_box)[0]) != key_base, "a pushed block must change the key"
    assert int(efe.box_layout_keys(moved_player)[0]) == key_base, "player motion must not"


def test_box_novelty_decays_once_a_layout_is_produced():
    memory = EpisodicMemory()
    assert memory.box_novelty("level_AxA", 123) == 1.0
    assert memory.record_box_layout("level_AxA", 123) is True
    assert memory.record_box_layout("level_AxA", 123) is False
    assert memory.box_novelty("level_AxA", 123) < 1.0
    assert memory.box_novelty("level_AxA", 999) == 1.0, "a different layout stays novel"


def test_expected_gem_delta_matches_distribution():
    logits = torch.tensor([[0.0, 100.0, 0.0, 0.0]])      # all mass on "1 gem"
    assert abs(float(EFE.expected_gem_delta(logits)[0]) - 1.0) < 1e-4


def test_ambiguity_is_higher_for_a_flat_prediction():
    flat = torch.zeros(1, glyphs.VOCAB_SIZE, 5, 5)
    sharp = torch.full((1, glyphs.VOCAB_SIZE, 5, 5), -10.0)
    sharp[0, 3] = 10.0
    assert float(EFE.tile_entropy(flat)[0]) > float(EFE.tile_entropy(sharp)[0])


def test_efe_prefers_gems_when_exploiting():
    """Zeroing the gem weights must change the objective, not just the logs."""
    base = EFEConfig(w_gem=10.0, w_gem_proximity=1.0)
    explore_only = base.scaled(0.0)
    assert explore_only.w_gem == 0.0
    assert explore_only.w_gem_proximity == 0.0
    assert explore_only.w_cell_novelty >= base.w_cell_novelty


# -------------------------------------------------------------------- planner
def test_planner_respects_mask():
    torch.manual_seed(0)
    model = HJEPA(WorldModelConfig(n_members=2)).eval()
    efe = EFE(EpisodicMemory(), EFEConfig(horizon=3, terminal_chunks=1))
    planner = CEMPlanner(
        model, efe, PlannerConfig(horizon=3, n_samples=24, n_elite=6, n_iters=2, keep_top_k_traces=1)
    )
    z = torch.zeros(1, model.config.channels, 15, 15)
    allowed = torch.zeros(N_ACTIONS, dtype=torch.bool)
    allowed[5] = True                       # only one legal action
    result = planner.plan(z, (0, 0), "level_AxA", 0, action_mask=allowed)
    assert result.action == 5
    assert set(result.sequence.tolist()) == {5}


def test_planner_returns_renderable_imagination():
    torch.manual_seed(0)
    model = HJEPA(WorldModelConfig(n_members=2)).eval()
    efe = EFE(EpisodicMemory(), EFEConfig(horizon=4, terminal_chunks=1))
    planner = CEMPlanner(
        model, efe, PlannerConfig(horizon=4, n_samples=16, n_elite=4, n_iters=1, keep_top_k_traces=1)
    )
    z = torch.zeros(1, model.config.channels, 15, 15)
    result = planner.plan(z, (0, 0), "level_AxA", 0)
    assert len(result.sequence) == 4
    assert len(result.imagined) == 4, "every imagined step must be decodable"
    assert result.imagined[0].shape == (15, 15)
    assert np.isfinite(result.efe)


# ------------------------------------------------------------------ collector
def test_collector_labels_and_validity():
    from train.collect import collect_episode
    from train.dataset import TransitionDataset

    env = MazeBenchEnv(bridge=MockMazeBridge(seed=7), max_actions=120)
    episode = collect_episode(env, policy="heuristic", max_steps=120, seed=7)
    env.close()

    assert len(episode) > 0
    assert len(episode.windows) == len(episode) + 1
    assert set(np.unique(episode.gem_delta)) <= {0, 1, 2, 3}
    assert set(np.unique(episode.valid)) <= {0, 1}

    # `moved` must agree with whether the window actually changed
    for t in range(len(episode)):
        if not episode.valid[t]:
            continue
        changed = not np.array_equal(episode.windows[t], episode.windows[t + 1])
        assert bool(episode.moved[t]) == changed

    dataset = TransitionDataset([episode])
    assert len(dataset) == int(episode.valid.sum())
    sample = dataset[0]
    assert sample["window"].dtype == torch.int64
    assert sample["gem_delta"].dtype == torch.int64


def test_chunk_dataset_never_spans_a_control_action():
    from train.collect import collect_episode
    from train.dataset import ChunkDataset

    env = MazeBenchEnv(bridge=MockMazeBridge(seed=11), max_actions=150)
    episode = collect_episode(env, policy="random", max_steps=150, seed=11)
    env.close()
    dataset = ChunkDataset([episode], chunk_len=4)
    for e, t in dataset.index:
        assert episode.valid[t : t + 4].all()


def test_oversampler_raises_gem_frequency():
    from train.collect import collect_episode
    from train.dataset import TransitionDataset

    env = MazeBenchEnv(bridge=MockMazeBridge(seed=3), max_actions=200)
    episode = collect_episode(env, policy="heuristic", max_steps=200, seed=3)
    env.close()
    dataset = TransitionDataset([episode])
    if dataset.label_stats()["gem_rate"] == 0:
        return
    weights = dataset.event_weights()
    gem_rows = [i for i, (e, t) in enumerate(dataset.index) if episode.gem_delta[t] > 0]
    assert weights[gem_rows].mean() > weights.mean()


# ---------------------------------------------------------------- hierarchy
def test_only_2_3_4_levels_and_strides_are_geometric():
    for n_levels, expected in ((2, [4]), (3, [4, 16]), (4, [4, 16, 64])):
        model = HJEPA(WorldModelConfig(n_levels=n_levels, n_members=2))
        assert len(model.abstract) == n_levels - 1
        assert model.strides == expected, f"n_levels={n_levels} gave {model.strides}"
    two = HJEPA(WorldModelConfig(n_levels=2, n_members=2))
    assert two.level3 is None, "a 2-level model has no third level"
    assert HJEPA(WorldModelConfig(n_levels=3, n_members=2)).level3 is not None


def test_each_level_is_trained_and_reported_separately():
    """A deeper stack must actually learn, not just add parameters.

    Per-level stats are what show whether level 3 or 4 is pulling its weight,
    so the loss has to emit them and every level has to receive gradient.
    """
    torch.manual_seed(0)
    model = HJEPA(WorldModelConfig(n_levels=4, n_members=2))
    batch = {
        "window": torch.randint(0, glyphs.VOCAB_SIZE, (6, 15, 15)),
        "scalars": torch.randn(6, 15),
    }
    for k, stride in enumerate(model.strides):
        batch[f"chunk_actions_{k}"] = torch.randint(0, N_ACTIONS, (6, stride))
        batch[f"future_window_{k}"] = torch.randint(0, glyphs.VOCAB_SIZE, (6, 15, 15))
        batch[f"future_scalars_{k}"] = torch.randn(6, 15)

    loss, stats = model.hierarchy_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    for level in (2, 3, 4):
        assert f"l{level}_pred" in stats, f"level {level} must be reported separately"
    assert stats["l2_stride"] == 4 and stats["l4_stride"] == 64

    for k, level in enumerate(model.abstract):
        grads = [p.grad for p in level.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), \
            f"abstract level {k + 2} received no gradient"


def test_deeper_hierarchy_has_fewer_training_samples():
    """A 4-level model needs 64 clean consecutive transitions per sample.

    That is a real cost of depth and the comparison has to account for it, so
    the dataset reports the counts rather than hiding them.
    """
    from env.mazebench_env import MazeBenchEnv
    from env.mock import MockMazeBridge
    from train.collect import collect_episode
    from train.dataset import ChunkDataset

    env = MazeBenchEnv(bridge=MockMazeBridge(seed=4), max_actions=200)
    episode = collect_episode(env, policy="heuristic", max_steps=200, seed=4)
    env.close()

    shallow = ChunkDataset([episode], chunk_len=4, n_levels=2)
    deep = ChunkDataset([episode], chunk_len=4, n_levels=4)
    assert shallow.strides == [4] and deep.strides == [4, 16, 64]
    assert len(deep) < len(shallow), "a deeper stack must yield fewer joint samples"
    counts = deep.level_counts()
    assert counts["level2_stride4"] > counts["level4_stride64"]

    sample = deep[0]
    for k, stride in enumerate(deep.strides):
        assert sample[f"chunk_actions_{k}"].shape == (stride,)
        assert f"future_window_{k}" in sample


def test_index_levels_equalises_data_across_depths():
    """The 2-vs-3-vs-4 comparison has to hold the sample set fixed.

    Indexed at its own depth, a 4-level model gets ~40% of the samples a
    2-level one does, because it needs 64 clean consecutive transitions per
    sample instead of 4. The deeper arm would then lose for being starved
    rather than for being deep. ``index_levels`` pins the sample set.
    """
    from env.mazebench_env import MazeBenchEnv
    from env.mock import MockMazeBridge
    from train.collect import collect_episode
    from train.dataset import ChunkDataset

    env = MazeBenchEnv(bridge=MockMazeBridge(seed=9), max_actions=250)
    episode = collect_episode(env, policy="heuristic", max_steps=250, seed=9)
    env.close()

    own_depth = [len(ChunkDataset([episode], n_levels=n)) for n in (2, 3, 4)]
    assert own_depth[0] > own_depth[2], "indexing at own depth should starve the deep arm"

    pinned = [ChunkDataset([episode], n_levels=n, index_levels=4) for n in (2, 3, 4)]
    counts = {len(d) for d in pinned}
    assert len(counts) == 1, f"pinned indexing must equalise sample counts, got {counts}"
    assert [d.strides for d in pinned] == [[4], [4, 16], [4, 16, 64]], \
        "each arm still emits futures for its own levels only"


def test_terminal_lookahead_grows_with_depth():
    """The case for another level is geometric reach at linear cost."""
    from agent.planner import CEMPlanner, PlannerConfig

    reach = {}
    for n_levels in (2, 3, 4):
        model = HJEPA(WorldModelConfig(n_levels=n_levels, n_members=2)).eval()
        efe = EFE(EpisodicMemory(), EFEConfig(horizon=2, terminal_chunks=4))
        planner = CEMPlanner(
            model, efe,
            PlannerConfig(horizon=2, n_samples=8, n_elite=2, n_iters=1, keep_top_k_traces=1),
        )
        result = planner.plan(
            torch.zeros(1, model.config.channels, 15, 15), (0, 0), "level_AxA", 0
        )
        assert np.isfinite(result.efe)
        reach[n_levels] = sum(s * 4 for s in model.strides)
    assert reach[2] < reach[3] < reach[4], f"lookahead did not grow: {reach}"
    assert reach[2] == 16 and reach[3] == 80 and reach[4] == 336


def test_abstract_coverage_is_kept_per_level():
    memory = EpisodicMemory()
    memory.add_abstract_all([np.ones(8), np.zeros(4)])
    # states are buffered until 32 accumulate or a query forces a flush
    assert float(memory.abstract_novelty(np.ones(8), level=0)[0]) == 0.0
    assert float(memory.abstract_novelty(np.ones(4), level=1)[0]) > 0.0
    assert set(memory.abstract_states) == {0, 1}
    # the two levels hold different widths, so they cannot have been merged
    assert memory.abstract_states[0].shape[1] == 8
    assert memory.abstract_states[1].shape[1] == 4


# ------------------------------------------------------------- ablation arms
def test_arms_switch_off_the_right_terms():
    """Each arm must remove exactly what it claims and nothing else.

    `memoryless` is the arm the research question turns on: no persistent
    record of anything seen, but ensemble disagreement stays, because model
    uncertainty is not episodic memory. If disagreement were removed too, the
    comparison would conflate curiosity with memory.
    """
    from agent.agent import apply_arm

    base = EFEConfig()
    assert base.w_cell_novelty > 0 and base.w_disagreement > 0

    full = apply_arm(base, "full")
    assert full.w_cell_novelty == base.w_cell_novelty
    assert full.w_disagreement == base.w_disagreement

    memoryless = apply_arm(base, "memoryless")
    for attr in ("w_cell_novelty", "w_pattern_novelty", "w_abstract_novelty", "w_box_novelty"):
        assert getattr(memoryless, attr) == 0.0, f"{attr} must be off in memoryless"
    assert memoryless.w_disagreement == base.w_disagreement, "disagreement is not memory"
    assert memoryless.w_gem == base.w_gem, "preferences are unchanged"

    no_epi = apply_arm(base, "no_epistemic")
    assert no_epi.w_disagreement == 0.0
    assert no_epi.w_ambiguity == base.w_ambiguity, "ambiguity is not an epistemic bonus"

    pragmatic = apply_arm(base, "pragmatic_only")
    assert pragmatic.w_ambiguity == 0.0 and pragmatic.w_disagreement == 0.0

    try:
        apply_arm(base, "nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown arm must be rejected, not silently ignored")


def test_memoryless_arm_cannot_use_the_frontier_jump():
    from agent.agent import AgentConfig, EFEAgent
    from agent.planner import PlannerConfig

    model = HJEPA(WorldModelConfig(n_members=2))
    agent = EFEAgent(model, AgentConfig(
        device="cpu", arm="memoryless",
        planner=PlannerConfig(horizon=2, n_samples=8, n_elite=2, n_iters=1),
        efe=EFEConfig(horizon=2, terminal_chunks=0),
    ))
    agent._steps_since_jump = 10 ** 9
    agent._steps_since_new_pattern = 10 ** 9
    agent._last_room = "level_AxA"
    jump, _ = agent._should_jump({})
    assert jump is False, "the frontier jump reads the room graph, which is memory"


# --------------------------------------------------------------- real levels
def _levels_available() -> bool:
    try:
        from env.real_levels import list_levels

        return len(list_levels()) > 0
    except Exception:
        return False


def test_real_levels_are_all_16x16():
    if not _levels_available():
        return                      # engine checkout absent; nothing to assert
    from env.real_levels import list_levels, load_level

    paths = list_levels()
    assert len(paths) >= 200, f"expected the shipped level set, found {len(paths)}"
    sizes = {load_level(p)[1]["size"] for p in paths}
    assert sizes == {(16, 16)}, f"levels are not uniformly 16x16: {sizes}"


def test_level_token_parsing():
    from env.real_levels import parse_cell

    assert parse_cell(".").kind == "floor"
    assert parse_cell(".+#+#+#+#").kind == "wall", "a stacked wall column is a wall"
    assert parse_cell(".+#+#+#+#").elevation == 4
    assert parse_cell("").kind == "hole", "a cell with nothing in it is void"
    assert parse_cell(".+G").has_gem
    assert parse_cell(".+b").has_box
    assert parse_cell(".+M3").has_box, "weightless boxes are pushable too"
    assert parse_cell(".+p").is_spawn
    assert parse_cell(".+pr").is_hazard, "punchers kill"
    assert parse_cell(".+Sl#").kind == "floor", "slopes are traversable"


def test_void_is_impassable_not_lethal():
    """`level_parsing.json` has no hole token: a floorless cell is out of
    bounds, not a death trap. Treating it as lethal killed a random walk in a
    handful of steps, since a 16x16 level is mostly void."""
    if not _levels_available():
        return
    from env.mazebench_env import MazeBenchEnv
    from env.mock import MockMazeBridge

    env = MazeBenchEnv(bridge=MockMazeBridge(seed=1, real_levels=True), max_actions=120)
    env.reset()
    rng = np.random.default_rng(0)
    done = False
    while not done:
        try:
            _, done, _ = env.step(int(rng.integers(8)))
        except Exception:
            break
    assert env.log.deaths == 0, f"a random walk died {env.log.deaths} times on real geometry"


def test_real_level_rooms_are_deterministic_per_seed():
    if not _levels_available():
        return
    from env.real_levels import RealLevelSource

    a = RealLevelSource(seed=7).room_for("level_HxI")
    b = RealLevelSource(seed=7).room_for("level_HxI")
    assert a.terrain == b.terrain and a.gems == b.gems


# ---------------------------------------------------------------- stuck guard
def _tiny_agent():
    from agent.agent import AgentConfig, EFEAgent
    from agent.planner import PlannerConfig

    model = HJEPA(WorldModelConfig(n_members=2))
    return EFEAgent(model, AgentConfig(
        device="cpu",
        planner=PlannerConfig(horizon=2, n_samples=8, n_elite=2, n_iters=1, keep_top_k_traces=1),
        efe=EFEConfig(horizon=2, terminal_chunks=0, use_pattern_novelty=False),
    ))


def test_refused_move_is_masked_then_released():
    """A move the world refuses must stop being proposed.

    Regression test. Without this the planner re-derives the same refused
    action from the same state forever -- an observed run spent 150 actions at
    a 0.98 blocked rate inside 4 distinct states.
    """
    agent = _tiny_agent()
    obs = {"current_room": "level_AxA", "gem_count": 0, "yaw": 0}

    def board(player_row: int) -> str:
        # Floor everywhere, not blank: parse_board strips trailing blank lines,
        # so an all-empty board changes shape on re-encode. Real MazeBench
        # boards are walled, so this is the realistic case.
        rows = [["A"] * 15 for _ in range(15)]
        rows[player_row][7] = "P"
        return "\n".join("".join(r) for r in rows)

    stay = board(7)
    agent.perceive({**obs, "level": stay})     # establishes _last_window
    agent._last_action = 0                     # tried to move up

    agent.perceive({**obs, "level": stay})     # identical board: refused
    assert 0 in agent._blocked, "an unchanged window after a move means blocked"

    mask = agent._action_mask(agent._last_window)
    assert not bool(mask[0]), "the refused move must be masked"
    assert bool(mask[4]), "rotations stay available"

    # a successful move clears the whole set
    agent._last_action = 1
    agent.perceive({**obs, "level": board(8)})
    assert not agent._blocked, "a successful move must clear the blocked set"


def test_all_moves_blocked_does_not_deadlock():
    agent = _tiny_agent()
    window = np.full((15, 15), glyphs.EMPTY_ID, dtype=np.int64)
    window[7, 7] = glyphs.glyph_id("P")
    agent._blocked = {0, 1, 2, 3}
    mask = agent._action_mask(window)
    assert bool(mask[:4].any()), "masking every move would leave nothing to plan"
    assert not agent._blocked, "the blocked set must be cleared, not left stale"


def test_hazard_mask_blocks_a_step_into_a_hole():
    agent = _tiny_agent()
    window = np.full((15, 15), glyphs.EMPTY_ID, dtype=np.int64)
    window[7, 7] = glyphs.glyph_id("P")
    window[6, 7] = glyphs.glyph_id("H")      # a hole directly "up"
    mask = agent._action_mask(window)
    assert not bool(mask[0]), "stepping into a visible hole must be masked"
    assert bool(mask[1]), "other directions stay open"


def test_stuck_with_no_frontier_arms_an_escape():
    agent = _tiny_agent()
    agent._steps_since_jump = 10 ** 9
    agent._steps_since_new_pattern = 10 ** 9
    agent._last_room = "level_AxA"
    from agent.memory import RoomRecord

    record = RoomRecord(room_id="level_AxA")
    record.steps_spent = 500
    record.distinct_patterns = {"a"}
    agent.memory.rooms["level_AxA"] = record

    jump, room = agent._should_jump({})
    assert jump is False, "there is nowhere to jump to"
    assert agent._escape_steps > 0, "being stuck with no frontier must arm an escape"


# --------------------------------------------------------------------- vision
def test_patch_grid_refuses_to_guess_a_video_grid():
    """A video backbone's token count factorises more than one way.

    V-JEPA 2 returns 2048 tokens = 8x16x16. A square-root heuristic reads that
    as 2x32x32 and silently builds the wrong grid from half-temporal features.
    So an ambiguous count without an explicit side must raise, not guess.
    """
    from model.vision import _patch_grid

    assert tuple(_patch_grid(torch.randn(2, 196, 768), side=14).shape) == (2, 768, 14, 14)
    assert tuple(
        _patch_grid(torch.randn(2, 197, 768), drop_prefix=1, side=14).shape
    ) == (2, 768, 14, 14)
    assert tuple(_patch_grid(torch.randn(2, 2048, 1024), side=16).shape) == (2, 1024, 16, 16)

    try:
        _patch_grid(torch.randn(2, 2048, 1024))
    except ValueError:
        pass
    else:
        raise AssertionError("an ambiguous token count must not be silently guessed")


def test_vision_backbones_are_pinned():
    from model.vision import PINNED

    for key, (name, revision) in PINNED.items():
        assert len(revision) == 40, f"{key} must pin a full commit sha"
        assert "/" in name


# ------------------------------------------------------------------- renderer
def test_replay_html_is_self_contained(tmp_path=None):
    import tempfile

    from agent.agent import StepRecord
    from viz.render import write_replay

    out = Path(tmp_path or tempfile.mkdtemp()) / "replay.html"
    record = StepRecord(
        step=0, kind="plan", action=0, action_name="move_up", room="level_AxA",
        gem_count=1, board="WWW\nWPW\nWWW", efe=-1.25,
        terms={"risk": -2.0, "epistemic": 0.75}, plan=[0, 1],
        imagined=[np.zeros((5, 5), dtype=np.int16)],
        real_window=np.zeros((5, 5), dtype=np.int64),
    )
    path = write_replay([record], out, summary={"gems": 1})
    html = path.read_text(encoding="utf8")
    assert "<!doctype html>" in html
    assert "move_up" in html
    assert "src=" not in html, "the replay must not depend on external files"


# ----------------------------------------------------------------------- main
def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as error:                        # noqa: BLE001
            failures.append((name, error))
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
