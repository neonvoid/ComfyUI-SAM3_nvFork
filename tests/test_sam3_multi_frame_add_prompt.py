"""Tests for SAM3MultiFrameAddPrompt (D-354).

Covers schema validation, pre-scan duplicate rejection, state chaining,
coord epsilon clamp, NaN/inf reject, collision dedup, and fail-soft policy.

Uses a fake VideoState that mimics the real SAM3 contract (.prompts list,
.with_prompt method, .num_frames attribute) without needing the actual
SAM3 model + torch.
"""

import importlib.util
import json
import pathlib
import sys

import pytest


# Stub torch / numpy minimally if not installed. The video_state module
# imports torch lazily inside create_mask, which we don't exercise here.
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules.setdefault("torch", type("torch_stub", (), {})())


# Bypass package __init__ chain; load video_state then the target node directly.
_nvfork_dir = pathlib.Path(__file__).resolve().parent.parent
_video_state_path = _nvfork_dir / "nodes" / "video_state.py"
_video_state_spec = importlib.util.spec_from_file_location(
    "video_state_test_stub", str(_video_state_path)
)
_video_state_mod = importlib.util.module_from_spec(_video_state_spec)
_video_state_spec.loader.exec_module(_video_state_mod)
VideoPrompt = _video_state_mod.VideoPrompt


# Load the target node — but it imports `.video_state` (relative). Need to
# create a synthetic package alias so the relative import resolves.
_pkg_alias_name = "_sam3_multi_frame_test_pkg"
_pkg = type(sys)(_pkg_alias_name)
_pkg.__path__ = [str(_nvfork_dir / "nodes")]
sys.modules[_pkg_alias_name] = _pkg
sys.modules[f"{_pkg_alias_name}.video_state"] = _video_state_mod

_target_path = _nvfork_dir / "nodes" / "sam3_multi_frame_add_prompt.py"
_target_spec = importlib.util.spec_from_file_location(
    f"{_pkg_alias_name}.sam3_multi_frame_add_prompt", str(_target_path)
)
_target_mod = importlib.util.module_from_spec(_target_spec)
_target_spec.loader.exec_module(_target_mod)

SAM3MultiFrameAddPrompt = _target_mod.SAM3MultiFrameAddPrompt
_validate_schema_version = _target_mod._validate_schema_version
_validate_required_keys = _target_mod._validate_required_keys
_prescan_duplicates = _target_mod._prescan_duplicates


# ---------------------------------------------------------------------------
# Fake video state — minimal SAM3_VIDEO_STATE surface
# ---------------------------------------------------------------------------

class FakeVideoState:
    """Mimics the immutable-state pattern of SAM3's VideoTrackingState.

    Real state has .prompts (list of VideoPrompt) + .with_prompt() returning
    new state + .num_frames. We replicate that minimal surface.
    """
    def __init__(self, prompts=None, num_frames=100):
        self.prompts = tuple(prompts or ())
        self.num_frames = num_frames

    def with_prompt(self, prompt):
        return FakeVideoState(
            prompts=tuple(self.prompts) + (prompt,),
            num_frames=self.num_frames,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(
    seeds=None,
    schema_version="1.0.0",
    schema_minor_compatible_with="1.x",
    schema_type="sam3_seed_prompts",
    accepted_frames=None,
    dwpose_person_index=0,
    drop_keys=None,
):
    """Build a SAM3_SEED_PROMPTS payload with defaults."""
    seeds = seeds if seeds is not None else _default_seeds()
    payload = {
        "schema_type": schema_type,
        "schema_version": schema_version,
        "schema_minor_compatible_with": schema_minor_compatible_with,
        "generator_node": "NV_DWPoseToSAM3Seeds",
        "subject_class": "face",
        "face_anchor_preset": "5_face_core",
        "anchor_indices_used": [36, 45, 48, 54, 30],
        "frame_width": 1920,
        "frame_height": 1080,
        "total_frames": 100,
        "seed_strategy": "pose_change_keyed",
        "normalization_space": "unit_xy",
        "dwpose_person_index": dwpose_person_index,
        "identity_lock_mode": "strict",
        "identity_lock_threshold_pct": 15.0,
        "identity_lock_subject_bbox": None,
        "ambiguity_eps": 0.02,
        "max_pose_score_in_run": 0.05,
        "flagged_frame_mode": "restrict_only",
        "flagged_frame_count": 0,
        "effective_min_frames_between_seeds": 10,
        "effective_force_include_first_frame": True,
        "lr_flip_count": 0,
        "accepted_frames": accepted_frames if accepted_frames is not None else [s["frame_idx"] for s in seeds],
        "candidate_events_log": [],
        "seeds": seeds,
    }
    for k in (drop_keys or []):
        payload.pop(k, None)
    return json.dumps(payload)


def _default_seeds():
    """Single-seed default at frame 0 with 5 anchor points."""
    return [{
        "frame_idx": 0,
        "obj_id": 1,
        "pos_pts": [[0.5, 0.5], [0.4, 0.5], [0.6, 0.5], [0.45, 0.6], [0.55, 0.6]],
        "neg_pts": [],
        "anchor_ids": [36, 45, 48, 54, 30],
        "anchor_confidences": [1.0, 1.0, 1.0, 1.0, 1.0],
        "anchor_count": 5,
        "pose_score": 0.0,
        "shape_score": 0.0,
        "seed_quality": 1.0,
        "selection_reason": "initial_frame",
    }]


def _make_seed(frame_idx, obj_id=1, pos_pts=None, neg_pts=None):
    return {
        "frame_idx": frame_idx, "obj_id": obj_id,
        "pos_pts": pos_pts if pos_pts is not None else [[0.5, 0.5]],
        "neg_pts": neg_pts or [],
        "anchor_ids": [30], "anchor_confidences": [1.0],
        "anchor_count": 1, "pose_score": 0.05, "shape_score": 0.0,
        "seed_quality": 1.0, "selection_reason": "pose_threshold",
    }


def _call_node(payload_json, state=None, **kwargs):
    node = SAM3MultiFrameAddPrompt()
    if state is None:
        state = FakeVideoState()
    return node.add_prompts(
        video_state=state, seed_prompts=payload_json, **kwargs,
    )


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


def test_schema_type_must_match():
    bad = _make_payload(schema_type="something_else")
    with pytest.raises(ValueError, match="schema_type"):
        _call_node(bad)


def test_schema_version_major_too_new_raises():
    bad = _make_payload(schema_version="2.0.0")
    with pytest.raises(ValueError, match="major>1"):
        _call_node(bad)


def test_schema_version_minor_newer_with_compat_warns_proceeds():
    """1.1.0 with schema_minor_compatible_with='1.x' should warn + proceed."""
    payload = _make_payload(
        schema_version="1.1.0",
        schema_minor_compatible_with="1.x",
    )
    new_state, applied, skipped_log = _call_node(payload)
    assert applied == 1


def test_schema_version_minor_newer_without_compat_raises():
    payload = _make_payload(
        schema_version="1.1.0",
        schema_minor_compatible_with="2.x",
    )
    with pytest.raises(ValueError, match="compatible_with"):
        _call_node(payload)


def test_missing_required_top_level_key_raises():
    payload = _make_payload(drop_keys=["seeds"])
    with pytest.raises(ValueError, match="missing required"):
        _call_node(payload)


def test_malformed_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        _call_node("not valid json {")


def test_root_not_dict_raises():
    with pytest.raises(ValueError, match="root must be JSON object"):
        _call_node("[]")


# ---------------------------------------------------------------------------
# Pre-scan duplicate tests
# ---------------------------------------------------------------------------


def test_prescan_duplicate_frame_obj_raises():
    """Payload with two seeds at same (frame_idx, obj_id) must raise BEFORE
    any state mutation."""
    seeds = [_make_seed(5), _make_seed(5)]
    payload = _make_payload(seeds=seeds)
    with pytest.raises(ValueError, match="duplicate"):
        _call_node(payload)


def test_prescan_different_obj_id_same_frame_ok():
    """Same frame_idx but different obj_id is NOT a duplicate."""
    seeds = [_make_seed(5, obj_id=1), _make_seed(5, obj_id=2)]
    payload = _make_payload(seeds=seeds)
    new_state, applied, _ = _call_node(payload)
    assert applied == 2


# ---------------------------------------------------------------------------
# State-chaining tests
# ---------------------------------------------------------------------------


def test_basic_seed_application():
    """Single seed applies; new state has 1 prompt; original state untouched."""
    initial_state = FakeVideoState()
    payload = _make_payload()
    new_state, applied, _ = _call_node(payload, state=initial_state)
    assert applied == 1
    assert len(new_state.prompts) == 1
    assert len(initial_state.prompts) == 0
    assert new_state.prompts[0].frame_idx == 0
    assert new_state.prompts[0].obj_id == 1


def test_multi_seed_state_chains_correctly():
    """N seeds → N prompts on returned state. Each seed sees prior state."""
    seeds = [_make_seed(0), _make_seed(10), _make_seed(20), _make_seed(30)]
    payload = _make_payload(seeds=seeds)
    new_state, applied, _ = _call_node(payload)
    assert applied == 4
    assert len(new_state.prompts) == 4
    frame_indices = [p.frame_idx for p in new_state.prompts]
    assert frame_indices == [0, 10, 20, 30]


def test_seeds_applied_in_sorted_order():
    """Seeds in non-sorted JSON should be applied in frame_idx asc order."""
    seeds = [_make_seed(20), _make_seed(0), _make_seed(10)]
    payload = _make_payload(seeds=seeds)
    new_state, applied, _ = _call_node(payload)
    assert applied == 3
    frame_indices = [p.frame_idx for p in new_state.prompts]
    assert frame_indices == [0, 10, 20]


# ---------------------------------------------------------------------------
# Collision tests
# ---------------------------------------------------------------------------


def test_init_frame_dup_skipped_with_other_seed_applies():
    """Existing prompt at (0, 1); payload has seeds at (0, 1) and (10, 1).
    First skipped, second applied."""
    existing_prompt = VideoPrompt.create_point(
        frame_idx=0, obj_id=1, points=[[0.5, 0.5]], labels=[1]
    )
    state = FakeVideoState(prompts=[existing_prompt])
    seeds = [_make_seed(0, obj_id=1), _make_seed(10, obj_id=1)]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(
        payload, state=state, skip_init_frame_dups=True,
    )
    assert applied == 1, "Second seed should apply, first should skip"
    assert len(new_state.prompts) == 2  # 1 existing + 1 new
    skipped = json.loads(skipped_log)
    dup_skips = [s for s in skipped if s["reason"] == "duplicate_existing_prompt"]
    assert len(dup_skips) == 1
    assert dup_skips[0]["frame_idx"] == 0


def test_init_frame_dup_raises_when_skip_false():
    """skip_init_frame_dups=False should raise loud on collision."""
    existing_prompt = VideoPrompt.create_point(
        frame_idx=0, obj_id=1, points=[[0.5, 0.5]], labels=[1]
    )
    state = FakeVideoState(prompts=[existing_prompt])
    payload = _make_payload()
    with pytest.raises(ValueError, match="collides"):
        _call_node(payload, state=state, skip_init_frame_dups=False)


def test_collision_set_includes_only_point_prompts():
    """Non-point prompts (e.g. box) on existing state don't trigger
    collision for a point seed at same (frame, obj_id)."""
    box_prompt = VideoPrompt.create_box(
        frame_idx=0, obj_id=1, box=[0.1, 0.1, 0.5, 0.5], is_positive=True
    )
    state = FakeVideoState(prompts=[box_prompt])
    payload = _make_payload()
    new_state, applied, _ = _call_node(payload, state=state)
    assert applied == 1


# ---------------------------------------------------------------------------
# Coord validation tests
# ---------------------------------------------------------------------------


def test_nan_coord_rejected_softly():
    """fail_soft_per_seed=True (default): NaN coord skips seed + logs."""
    bad_seed = _make_seed(5, pos_pts=[[float("nan"), 0.5]])
    seeds = [_make_seed(0), bad_seed]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload, fail_soft_per_seed=True)
    assert applied == 1
    skipped = json.loads(skipped_log)
    nan_skips = [s for s in skipped if s["reason"] == "coord_nan_inf"]
    assert len(nan_skips) == 1


def test_nan_coord_raises_when_soft_false():
    seeds = [_make_seed(5, pos_pts=[[float("inf"), 0.5]])]
    payload = _make_payload(seeds=seeds)
    with pytest.raises(ValueError, match="non-finite"):
        _call_node(payload, fail_soft_per_seed=False)


def test_out_of_range_coord_rejected():
    """Coord at 1.5 (out of [-eps, 1+eps]) rejected with coord_out_of_range."""
    bad_seed = _make_seed(5, pos_pts=[[1.5, 0.5]])
    seeds = [_make_seed(0), bad_seed]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload, fail_soft_per_seed=True)
    assert applied == 1
    skipped = json.loads(skipped_log)
    oor_skips = [s for s in skipped if s["reason"] == "coord_out_of_range"]
    assert len(oor_skips) == 1


def test_epsilon_drift_clamps_to_unit():
    """Coord at 1.0005 (within epsilon) clamps to 1.0, seed applies."""
    drift_seed = _make_seed(5, pos_pts=[[1.0005, 0.5]])
    seeds = [drift_seed]
    payload = _make_payload(seeds=seeds)
    new_state, applied, _ = _call_node(payload)
    assert applied == 1
    # Confirm clamped value got stored
    prompt = new_state.prompts[0]
    points, labels = prompt.data
    x = points[0][0]
    assert 0.0 <= x <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_seeds_raises_loudly():
    """Payload with seeds=[] + valid schema → must raise loud (chain broken)."""
    payload = _make_payload(seeds=[], accepted_frames=[])
    with pytest.raises(RuntimeError, match="no seeds applied"):
        _call_node(payload)


def test_all_seeds_collide_raises_loudly():
    """If ALL seeds get skipped due to collisions, raise loud."""
    existing = [
        VideoPrompt.create_point(0, 1, [[0.5, 0.5]], [1]),
        VideoPrompt.create_point(10, 1, [[0.5, 0.5]], [1]),
    ]
    state = FakeVideoState(prompts=existing)
    seeds = [_make_seed(0), _make_seed(10)]
    payload = _make_payload(seeds=seeds)
    with pytest.raises(RuntimeError, match="no seeds applied"):
        _call_node(payload, state=state)


def test_frame_idx_beyond_num_frames_softly_skipped():
    """Seed with frame_idx >= video_state.num_frames is fail-soft skipped."""
    state = FakeVideoState(num_frames=10)
    seeds = [_make_seed(5), _make_seed(20)]  # 20 >= 10
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload, state=state)
    assert applied == 1
    skipped = json.loads(skipped_log)
    oor = [s for s in skipped if s["reason"] == "frame_idx_out_of_range"]
    assert len(oor) == 1
    assert oor[0]["frame_idx"] == 20


def test_neg_pts_applied_with_label_0():
    """negative_points are stored as label=0 in VideoPrompt."""
    seed = _make_seed(0, pos_pts=[[0.5, 0.5]], neg_pts=[[0.1, 0.1]])
    payload = _make_payload(seeds=[seed])
    new_state, applied, _ = _call_node(payload)
    assert applied == 1
    prompt = new_state.prompts[0]
    _, labels = prompt.data
    assert 1 in labels
    assert 0 in labels


def test_seed_with_no_points_skipped():
    """A seed with empty pos_pts and neg_pts can't be applied."""
    bad = _make_seed(5, pos_pts=[], neg_pts=[])
    seeds = [_make_seed(0), bad]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload)
    assert applied == 1
    skipped = json.loads(skipped_log)
    no_pts = [s for s in skipped if s["reason"] == "no_points_in_seed"]
    assert len(no_pts) == 1


# ---------------------------------------------------------------------------
# State immutability sanity
# ---------------------------------------------------------------------------


def test_input_state_not_mutated():
    """Original state object should be unchanged after call (immutable chaining)."""
    state = FakeVideoState()
    original_prompts = state.prompts
    payload = _make_payload()
    new_state, _, _ = _call_node(payload, state=state)
    assert state.prompts is original_prompts  # identity preserved
    assert new_state is not state


def test_widget_default_positions_d_201():
    """D-201: required block + optional block widget order pinning."""
    schema = SAM3MultiFrameAddPrompt.INPUT_TYPES()
    required = list(schema["required"].keys())
    assert required == ["video_state", "seed_prompts"]
    optional = list(schema["optional"].keys())
    assert optional == [
        "skip_init_frame_dups", "fail_soft_per_seed",
        "error_on_noop", "verbose_debug",
    ]


# ---------------------------------------------------------------------------
# R1 code-review follow-up tests — fail-soft non-numeric, malformed IDs,
# negative frame_idx, malformed neg_pts container
# ---------------------------------------------------------------------------


def test_non_numeric_coord_rejected_softly():
    """fail_soft_per_seed=True: non-numeric coord member (e.g. string) skips
    seed + logs coord_not_numeric. R3-code-review fix."""
    bad = _make_seed(5, pos_pts=[["x", 0.5]])
    seeds = [_make_seed(0), bad]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload, fail_soft_per_seed=True)
    assert applied == 1
    skipped = json.loads(skipped_log)
    non_num = [s for s in skipped if s["reason"] == "coord_not_numeric"]
    assert len(non_num) == 1


def test_non_numeric_coord_raises_when_soft_false():
    """fail_soft_per_seed=False: non-numeric coord raises immediately."""
    bad = _make_seed(5, pos_pts=[["x", 0.5]])
    seeds = [bad]
    payload = _make_payload(seeds=seeds)
    with pytest.raises(ValueError, match="not numeric"):
        _call_node(payload, fail_soft_per_seed=False)


def test_bool_coord_rejected():
    """bool member in coord pair rejected as coord_not_numeric (bool is
    subclass of int → float() would silently accept True/False)."""
    bad = _make_seed(5, pos_pts=[[True, 0.5]])
    seeds = [_make_seed(0), bad]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload, fail_soft_per_seed=True)
    skipped = json.loads(skipped_log)
    non_num = [s for s in skipped if s["reason"] == "coord_not_numeric"]
    assert len(non_num) == 1


def test_malformed_frame_idx_raises_in_prescan():
    """seed with missing frame_idx must raise loud at pre-scan, NOT
    propagate to main loop. R3-code-review fix."""
    bad_seed = {
        "obj_id": 1, "pos_pts": [[0.5, 0.5]], "neg_pts": [],
        "anchor_ids": [30], "anchor_confidences": [1.0],
        "anchor_count": 1, "pose_score": 0.0, "shape_score": 0.0,
        "seed_quality": 1.0, "selection_reason": "pose_threshold",
    }
    payload = _make_payload(seeds=[bad_seed], accepted_frames=[])
    with pytest.raises(ValueError, match="missing 'frame_idx'"):
        _call_node(payload)


def test_malformed_obj_id_raises_in_prescan():
    """seed with non-integer obj_id raises loud at pre-scan."""
    bad_seed = _make_seed(0)
    bad_seed["obj_id"] = "not_an_int"
    payload = _make_payload(seeds=[bad_seed])
    with pytest.raises(ValueError, match="not integer-coercible"):
        _call_node(payload)


def test_negative_frame_idx_raises_in_prescan():
    """Negative frame_idx rejected at pre-scan. R3-code-review fix."""
    bad_seed = _make_seed(-1)
    payload = _make_payload(seeds=[bad_seed])
    with pytest.raises(ValueError, match="negative frame_idx"):
        _call_node(payload)


def test_bool_frame_idx_rejected_in_prescan():
    """bool frame_idx rejected (would be silently coerced to 0/1 by int())."""
    bad_seed = _make_seed(0)
    bad_seed["frame_idx"] = True
    payload = _make_payload(seeds=[bad_seed])
    with pytest.raises(ValueError, match="must not be bool"):
        _call_node(payload)


def test_non_dict_seed_raises_in_prescan():
    """seed not a dict raises at pre-scan."""
    payload = _make_payload(seeds=["not a dict"], accepted_frames=[])
    with pytest.raises(ValueError, match="not a dict"):
        _call_node(payload)


def test_pos_pts_not_list_rejected_softly():
    """seed.pos_pts not a list → fail-soft skip with reason."""
    bad_seed = _make_seed(5)
    bad_seed["pos_pts"] = "not a list"
    seeds = [_make_seed(0), bad_seed]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload)
    skipped = json.loads(skipped_log)
    pos_skips = [s for s in skipped if s["reason"] == "pos_pts_not_list"]
    assert len(pos_skips) == 1


def test_neg_pts_not_list_rejected_softly():
    """seed.neg_pts not a list → fail-soft skip. R2-code-review-r2 fix."""
    bad_seed = _make_seed(5)
    bad_seed["neg_pts"] = "not a list"
    seeds = [_make_seed(0), bad_seed]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload)
    skipped = json.loads(skipped_log)
    neg_skips = [s for s in skipped if s["reason"] == "neg_pts_not_list"]
    assert len(neg_skips) == 1


def test_neg_pts_none_rejected_softly():
    """seed.neg_pts = None (instead of empty list) → fail-soft skip."""
    bad_seed = _make_seed(5)
    bad_seed["neg_pts"] = None
    seeds = [_make_seed(0), bad_seed]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(payload)
    skipped = json.loads(skipped_log)
    neg_skips = [s for s in skipped if s["reason"] == "neg_pts_not_list"]
    assert len(neg_skips) == 1


def test_neg_pts_not_list_raises_when_soft_false():
    """neg_pts scalar with fail_soft=False raises loud (symmetric to pos_pts)."""
    bad_seed = _make_seed(5)
    bad_seed["neg_pts"] = 7
    seeds = [bad_seed]
    payload = _make_payload(seeds=seeds)
    with pytest.raises(ValueError, match="neg_pts must be a list"):
        _call_node(payload, fail_soft_per_seed=False)


# ---------------------------------------------------------------------------
# R3 integration-audit follow-up tests — error_on_noop + stream mismatch
# ---------------------------------------------------------------------------


def test_error_on_noop_false_returns_unchanged_state():
    """All seeds skipped due to collisions + error_on_noop=False →
    return unchanged video_state + applied=0, NO raise.

    Integration audit fix: benign no-op (2-pass refinement with empty
    flagged_frames) shouldn't crash the chain.
    """
    existing = [
        VideoPrompt.create_point(0, 1, [[0.5, 0.5]], [1]),
        VideoPrompt.create_point(10, 1, [[0.5, 0.5]], [1]),
    ]
    state = FakeVideoState(prompts=existing)
    seeds = [_make_seed(0), _make_seed(10)]
    payload = _make_payload(seeds=seeds)
    new_state, applied, skipped_log = _call_node(
        payload, state=state, error_on_noop=False,
    )
    assert applied == 0
    # Returned state is the original (no mutations applied)
    assert len(new_state.prompts) == 2  # only the original 2


def test_error_on_noop_true_raises_default():
    """Default error_on_noop=True preserves loud-fail UX."""
    existing = [VideoPrompt.create_point(0, 1, [[0.5, 0.5]], [1])]
    state = FakeVideoState(prompts=existing)
    seeds = [_make_seed(0)]
    payload = _make_payload(seeds=seeds)
    with pytest.raises(RuntimeError, match="no seeds applied"):
        _call_node(payload, state=state, error_on_noop=True)


def test_stream_mismatch_warning_total_frames():
    """payload total_frames > video_state.num_frames triggers warning
    log (not raise) — integration audit A3 fix."""
    state = FakeVideoState(num_frames=20)
    seeds = [_make_seed(0), _make_seed(15)]
    # Build payload that claims total_frames=60 (mismatch vs state's 20)
    payload_dict = json.loads(_make_payload(seeds=seeds, accepted_frames=[0, 15]))
    payload_dict["total_frames"] = 60
    payload = json.dumps(payload_dict)
    # Should NOT raise; warning printed to stdout
    new_state, applied, _ = _call_node(payload, state=state)
    assert applied == 2  # both seeds within state's num_frames=20
