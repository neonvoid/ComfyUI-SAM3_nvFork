"""Tests for the name-addressed mask pipeline (Phase 2 output side).

Covers:
  - resolve_subject_name() policy: exact match, unique-normalized fallback,
    ambiguous error, not-found error, untracked-obj_id error, no-subject_map error
  - SAM3SelectMask select_by="name" mode (back-compat id mode untouched)
  - SAM3MaskRouter 8-slot name routing + frame modes + batch output

End-to-end where practical: build a real track_info via SAM3MaskTracks (with a
subject_map), then drive SelectMask / MaskRouter off that exact output — this is
the contract operators actually wire.

Loads sam3_mask_tracks.py directly via importlib (bypasses package __init__).
"""

import importlib.util
import json
import pathlib

import pytest
import torch


_nvfork_dir = pathlib.Path(__file__).resolve().parent.parent
_target_path = _nvfork_dir / "nodes" / "sam3_mask_tracks.py"
_spec = importlib.util.spec_from_file_location("sam3_mask_tracks_name_stub", str(_target_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SAM3MaskTracks = _mod.SAM3MaskTracks
SAM3SelectMask = _mod.SAM3SelectMask
SAM3MaskRouter = _mod.SAM3MaskRouter
resolve_subject_name = _mod.resolve_subject_name
_parse_names_list = _mod._parse_names_list


class FakeVideoState:
    def __init__(self, num_frames, height, width):
        self.num_frames = num_frames
        self.height = height
        self.width = width


def _make_mask(h, w, region):
    m = torch.zeros(h, w)
    y0, y1, x0, x1 = region
    m[y0:y1, x0:x1] = 1.0
    return m


def _build_head_body_hair():
    """Run SAM3MaskTracks with a 3-subject map -> (all_masks, track_info dict).

    Channels (sorted obj_id): ch0=obj1=head, ch1=obj2=body, ch2=obj3=hair.
    Distinct quadrants per identity so we can assert which mask landed where.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=2, height=h, width=w)
    head = _make_mask(h, w, (0, 2, 0, 2))   # top-left
    body = _make_mask(h, w, (2, 4, 0, 2))   # bottom-left
    hair = _make_mask(h, w, (0, 2, 2, 4))   # top-right
    masks = {
        0: {"mask": torch.stack([head, body, hair], dim=0), "obj_ids": [1, 2, 3]},
        1: {"mask": torch.stack([head, body, hair], dim=0), "obj_ids": [1, 2, 3]},
    }
    sm = json.dumps({"subjects": [
        {"obj_id": 1, "name": "head"},
        {"obj_id": 2, "name": "body"},
        {"obj_id": 3, "name": "hair"},
    ]})
    node = SAM3MaskTracks()
    all_masks, _bbox, track_info_json, _n = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1, subject_map=sm,
    )
    return all_masks, track_info_json, {"head": head, "body": body, "hair": hair}


# ---------------------------------------------------------------------------
# resolve_subject_name policy
# ---------------------------------------------------------------------------

_INFO = {
    "subject_to_obj_id": {"head": 1, "body": 2, "hair": 3},
    "obj_id_to_channel": {"1": 0, "2": 1, "3": 2},
    "subjects_present": ["head", "body", "hair"],
}


def test_resolver_exact_match():
    assert resolve_subject_name(_INFO, "body") == (2, 1)


def test_resolver_accepts_json_string():
    assert resolve_subject_name(json.dumps(_INFO), "hair") == (3, 2)


def test_resolver_unique_normalized_fallback():
    # Different case / spacing resolves uniquely.
    assert resolve_subject_name(_INFO, "  HEAD ") == (1, 0)


def test_resolver_ambiguous_normalized_raises():
    info = {
        "subject_to_obj_id": {"Left Arm": 1, "left arm": 2},
        "obj_id_to_channel": {"1": 0, "2": 1},
    }
    # exact match wins when present...
    assert resolve_subject_name(info, "left arm") == (2, 1)
    # ...but a non-exact query that normalizes to both is ambiguous.
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_subject_name(info, "LEFT ARM")


def test_resolver_not_found_lists_available():
    with pytest.raises(ValueError, match="Available subjects: body, hair, head"):
        resolve_subject_name(_INFO, "hed")


def test_resolver_untracked_obj_id_raises():
    info = {
        "subject_to_obj_id": {"head": 1, "hair": 3},
        "obj_id_to_channel": {"1": 0},   # obj 3 not tracked this run
        "subjects_present": ["head"],
    }
    with pytest.raises(ValueError, match="produced no mask track"):
        resolve_subject_name(info, "hair")


def test_resolver_no_subject_map_raises():
    with pytest.raises(ValueError, match="no subject_map"):
        resolve_subject_name({"obj_id_to_channel": {"1": 0}}, "head")


def test_resolver_derives_table_from_subject_map_only():
    # Only subject_map (obj_id->name) present; table derived.
    info = {"subject_map": {"1": "head", "2": "body"}, "obj_id_to_channel": {"1": 0, "2": 1}}
    assert resolve_subject_name(info, "body") == (2, 1)


def test_parse_names_list_newline_and_comma():
    assert _parse_names_list("head\nhair") == ["head", "hair"]
    assert _parse_names_list("head, body ,hair") == ["head", "body", "hair"]
    assert _parse_names_list("head\nhead\n  ") == ["head"]   # dedup + drop blanks
    assert _parse_names_list("") == []


# ---------------------------------------------------------------------------
# SAM3SelectMask name mode
# ---------------------------------------------------------------------------

def test_select_by_name_single_subject_returns_correct_channel():
    all_masks, track_info_json, q = _build_head_body_hair()
    node = SAM3SelectMask()
    selected, ids_str, _viz = node.select(
        all_masks, select_by="name", subject_names="body", track_info=track_info_json,
    )
    # body lives at channel 1; union of one channel -> [F, H, W]
    assert selected.dim() == 3
    torch.testing.assert_close(selected[0], q["body"])
    assert ids_str == "1"


def test_select_by_name_multiple_union():
    all_masks, track_info_json, q = _build_head_body_hair()
    node = SAM3SelectMask()
    selected, _ids, _viz = node.select(
        all_masks, select_by="name", subject_names="head\nhair",
        combine_mode="union", track_info=track_info_json,
    )
    expected = torch.max(q["head"], q["hair"])
    torch.testing.assert_close(selected[0], expected)


def test_select_by_name_separate_keeps_channels():
    all_masks, track_info_json, _q = _build_head_body_hair()
    node = SAM3SelectMask()
    selected, _ids, _viz = node.select(
        all_masks, select_by="name", subject_names="head, body, hair",
        combine_mode="separate", track_info=track_info_json,
    )
    assert selected.shape[1] == 3  # [F, 3, H, W]


def test_select_by_name_missing_error_policy_raises():
    all_masks, track_info_json, _q = _build_head_body_hair()
    node = SAM3SelectMask()
    with pytest.raises(ValueError, match="not found"):
        node.select(all_masks, select_by="name", subject_names="elbow",
                    missing_name_policy="error", track_info=track_info_json)


def test_select_by_name_missing_empty_warn_skips():
    all_masks, track_info_json, q = _build_head_body_hair()
    node = SAM3SelectMask()
    selected, _ids, _viz = node.select(
        all_masks, select_by="name", subject_names="elbow\nbody",
        missing_name_policy="empty_warn", track_info=track_info_json,
    )
    # elbow skipped, body kept
    torch.testing.assert_close(selected[0], q["body"])


def test_select_by_name_all_missing_empty_warn_returns_empty():
    all_masks, track_info_json, _q = _build_head_body_hair()
    node = SAM3SelectMask()
    selected, ids_str, _viz = node.select(
        all_masks, select_by="name", subject_names="elbow\nknee",
        missing_name_policy="empty_warn", track_info=track_info_json,
    )
    assert selected.sum().item() == 0.0
    assert ids_str == ""


def test_select_by_id_mode_unchanged_ignores_name_fields():
    all_masks, track_info_json, q = _build_head_body_hair()
    node = SAM3SelectMask()
    # Legacy path: select_by defaults to 'id'; object_ids picks channel 2 (hair).
    selected, ids_str, _viz = node.select(all_masks, object_ids="2")
    torch.testing.assert_close(selected[0], q["hair"])
    assert ids_str == "2"


def test_select_by_name_requires_track_info():
    all_masks, _track_info_json, _q = _build_head_body_hair()
    node = SAM3SelectMask()
    with pytest.raises(ValueError, match="requires track_info"):
        node.select(all_masks, select_by="name", subject_names="head", track_info="")


def test_select_by_name_stale_track_info_channel_overflow_raises():
    # track_info says 'ghost' -> channel 5, but tensor has only 3 channels.
    all_masks, _ti, _q = _build_head_body_hair()
    stale = json.dumps({
        "subject_to_obj_id": {"ghost": 9},
        "obj_id_to_channel": {"9": 5},
    })
    node = SAM3SelectMask()
    with pytest.raises(ValueError, match="different runs"):
        node.select(all_masks, select_by="name", subject_names="ghost", track_info=stale)


# ---------------------------------------------------------------------------
# SAM3MaskRouter
# ---------------------------------------------------------------------------

def test_router_routes_named_slots_to_correct_masks():
    all_masks, track_info_json, q = _build_head_body_hair()
    node = SAM3MaskRouter()
    out = node.route_masks(
        all_masks, track_info=track_info_json,
        slot_1_name="head", slot_2_name="hair", slot_3_name="body",
        slot_4_name="", slot_5_name="", slot_6_name="",
        slot_7_name="", slot_8_name="",
    )
    slot1, slot2, slot3 = out[0], out[1], out[2]
    torch.testing.assert_close(slot1[0], q["head"])
    torch.testing.assert_close(slot2[0], q["hair"])
    torch.testing.assert_close(slot3[0], q["body"])


def test_router_blank_slot_is_zero_and_excluded_from_batch():
    all_masks, track_info_json, _q = _build_head_body_hair()
    node = SAM3MaskRouter()
    out = node.route_masks(
        all_masks, track_info=track_info_json,
        slot_1_name="head", slot_2_name="", slot_3_name="",
        slot_4_name="", slot_5_name="", slot_6_name="",
        slot_7_name="", slot_8_name="",
    )
    names, masks_batch = out[8], out[9]
    assert out[1].sum().item() == 0.0          # blank slot 2 -> zeros
    assert names == "head"                      # only head in names
    assert masks_batch.shape[0] == 2            # F=2 frames for one resolved lane


def test_router_names_align_with_batch_subject_major():
    all_masks, track_info_json, _q = _build_head_body_hair()
    node = SAM3MaskRouter()
    out = node.route_masks(
        all_masks, track_info=track_info_json,
        slot_1_name="head", slot_2_name="body", slot_3_name="hair",
        slot_4_name="", slot_5_name="", slot_6_name="",
        slot_7_name="", slot_8_name="",
    )
    names, masks_batch = out[8], out[9]
    assert names == "head\nbody\nhair"
    # 3 subjects * 2 frames, subject-major
    assert masks_batch.shape[0] == 6


def test_router_frame_mode_first_frame_emits_single_frame_lanes():
    all_masks, track_info_json, q = _build_head_body_hair()
    node = SAM3MaskRouter()
    out = node.route_masks(
        all_masks, track_info=track_info_json, frame_mode="first_frame",
        slot_1_name="head", slot_2_name="", slot_3_name="",
        slot_4_name="", slot_5_name="", slot_6_name="",
        slot_7_name="", slot_8_name="",
    )
    slot1 = out[0]
    assert slot1.shape[0] == 1
    torch.testing.assert_close(slot1[0], q["head"])


def test_router_missing_name_error_raises():
    all_masks, track_info_json, _q = _build_head_body_hair()
    node = SAM3MaskRouter()
    with pytest.raises(ValueError, match="not found"):
        node.route_masks(
            all_masks, track_info=track_info_json, missing_name_policy="error",
            slot_1_name="elbow", slot_2_name="", slot_3_name="",
            slot_4_name="", slot_5_name="", slot_6_name="",
            slot_7_name="", slot_8_name="",
        )


def test_router_missing_name_empty_warn_zero_lane_excluded():
    all_masks, track_info_json, q = _build_head_body_hair()
    node = SAM3MaskRouter()
    out = node.route_masks(
        all_masks, track_info=track_info_json, missing_name_policy="empty_warn",
        slot_1_name="elbow", slot_2_name="body", slot_3_name="",
        slot_4_name="", slot_5_name="", slot_6_name="",
        slot_7_name="", slot_8_name="",
    )
    names, masks_batch = out[8], out[9]
    assert out[0].sum().item() == 0.0   # elbow lane zeroed
    assert names == "body"              # elbow excluded
    torch.testing.assert_close(out[1][0], q["body"])
    assert masks_batch.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
