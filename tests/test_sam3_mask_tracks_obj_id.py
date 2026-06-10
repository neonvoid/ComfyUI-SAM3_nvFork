"""Tests for SAM3MaskTracks obj_id mapping fix (2026-05-20 audit Bug #1).

The pre-fix code assumed channel-index == SAM3 obj_id, which silently swapped
identities when SAM3 returned out-of-order or sparse obj_ids per frame. These
tests pin the post-fix behavior:

  - When mask dict carries `obj_ids` metadata, channels are stable across
    frames by SAM3 obj_id (not local channel position).
  - When raw-tensor / empty-obj_ids input is supplied, legacy channel-index
    semantics are preserved (back-compat).
  - track_info exposes both legacy `id` (channel index) and new `sam3_obj_id`
    + `obj_id_mapping` + `id_source` fields.

Loads sam3_mask_tracks.py directly via importlib (bypasses package __init__).
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
import torch


_nvfork_dir = pathlib.Path(__file__).resolve().parent.parent
_target_path = _nvfork_dir / "nodes" / "sam3_mask_tracks.py"
_target_spec = importlib.util.spec_from_file_location(
    "sam3_mask_tracks_test_stub", str(_target_path)
)
_target_mod = importlib.util.module_from_spec(_target_spec)
_target_spec.loader.exec_module(_target_mod)
SAM3MaskTracks = _target_mod.SAM3MaskTracks


class FakeVideoState:
    """Minimal SAM3_VIDEO_STATE surface used by extract_tracks."""
    def __init__(self, num_frames, height, width):
        self.num_frames = num_frames
        self.height = height
        self.width = width


def _make_mask(h, w, value=1.0, region=None):
    """Build a [H, W] float mask. region = (y0, y1, x0, x1) or full-frame if None."""
    m = torch.zeros(h, w)
    if region is None:
        m.fill_(value)
    else:
        y0, y1, x0, x1 = region
        m[y0:y1, x0:x1] = value
    return m


# ---------------------------------------------------------------------------
# Test 1: THE BUG — sparse/reordered obj_ids must not swap identity
# ---------------------------------------------------------------------------

def test_obj_id_stays_at_stable_channel_across_frames():
    """Frame 0 has obj_ids=[1, 3]; Frame 1 has obj_ids=[3, 5].
    obj_id 3's mask must land at the SAME global channel on both frames.
    Pre-fix: frame 0 channel 1 = obj 3, frame 1 channel 0 = obj 3 → SWAP.
    Post-fix: obj 3 lives at a stable global channel on both frames.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=2, height=h, width=w)

    # Distinguishable mask patterns per identity:
    mask_obj1 = _make_mask(h, w, region=(0, 2, 0, 2))   # top-left quadrant
    mask_obj3 = _make_mask(h, w, region=(0, 2, 2, 4))   # top-right quadrant
    mask_obj5 = _make_mask(h, w, region=(2, 4, 0, 2))   # bottom-left quadrant

    masks = {
        0: {
            "mask": torch.stack([mask_obj1, mask_obj3], dim=0),
            "obj_ids": [1, 3],
        },
        1: {
            "mask": torch.stack([mask_obj3, mask_obj5], dim=0),
            "obj_ids": [3, 5],
        },
    }

    node = SAM3MaskTracks()
    all_masks, _bbox, track_info_json, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )

    assert num_objects == 3, "global obj_ids should be [1, 3, 5]"
    info = json.loads(track_info_json)
    mapping = info["obj_id_mapping"]
    # Channels sorted by obj_id ascending: ch0=1, ch1=3, ch2=5
    assert mapping["0"] == 1
    assert mapping["1"] == 3
    assert mapping["2"] == 5
    assert info["id_source"] == "sam3_obj_ids"

    # Critical: obj_id 3 (channel 1) on BOTH frames must equal mask_obj3.
    torch.testing.assert_close(all_masks[0, 1], mask_obj3)
    torch.testing.assert_close(all_masks[1, 1], mask_obj3)
    # obj_id 1 only present on frame 0 → channel 0
    torch.testing.assert_close(all_masks[0, 0], mask_obj1)
    # Frame 1 channel 0 (obj_id 1) should be ZEROS (obj 1 left the scene)
    assert all_masks[1, 0].sum().item() == 0.0
    # obj_id 5 only present on frame 1 → channel 2
    torch.testing.assert_close(all_masks[1, 2], mask_obj5)
    assert all_masks[0, 2].sum().item() == 0.0


def test_track_info_exposes_sam3_obj_id_per_channel():
    """track_info.objects[] should include both `id` (channel) and `sam3_obj_id`."""
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    masks = {
        0: {
            "mask": torch.stack([
                _make_mask(h, w, region=(0, 4, 0, 4)),
                _make_mask(h, w, region=(0, 4, 0, 4)),
            ], dim=0),
            "obj_ids": [7, 42],   # deliberately non-consecutive
        },
    }
    node = SAM3MaskTracks()
    _all_masks, _bbox, track_info_json, _n = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    objs = {o["id"]: o for o in info["objects"]}
    assert objs[0]["sam3_obj_id"] == 7
    assert objs[1]["sam3_obj_id"] == 42


# ---------------------------------------------------------------------------
# Test 2: Legacy back-compat — raw tensor (no dict) keeps channel-index mode
# ---------------------------------------------------------------------------

def test_legacy_raw_tensor_input_preserves_channel_index_semantics():
    """Raw tensor (no obj_ids metadata) → legacy channel-index mode.
    Existing workflows wiring raw tensors must keep working unchanged.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=2, height=h, width=w)
    mask_a = _make_mask(h, w, region=(0, 2, 0, 2))
    mask_b = _make_mask(h, w, region=(2, 4, 2, 4))

    # No dict wrapper, no obj_ids — raw tensor mode.
    masks = {
        0: torch.stack([mask_a, mask_b], dim=0),
        1: torch.stack([mask_a, mask_b], dim=0),
    }
    node = SAM3MaskTracks()
    all_masks, _bbox, track_info_json, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    assert info["id_source"] == "legacy_channel_index"
    assert num_objects == 2
    # Channel-index obj_id == channel position in legacy mode
    assert info["obj_id_mapping"]["0"] == 0
    assert info["obj_id_mapping"]["1"] == 1
    torch.testing.assert_close(all_masks[0, 0], mask_a)
    torch.testing.assert_close(all_masks[0, 1], mask_b)


def test_empty_obj_ids_list_falls_back_to_legacy_mode():
    """Dict with `obj_ids: []` should be treated as legacy (channel-index)."""
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    masks = {
        0: {
            "mask": torch.stack([_make_mask(h, w, region=(0, 2, 0, 2))], dim=0),
            "obj_ids": [],
        },
    }
    node = SAM3MaskTracks()
    _all_masks, _bbox, track_info_json, _n = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    assert info["id_source"] == "legacy_channel_index"


# ---------------------------------------------------------------------------
# Test 3: Single-mask [H, W] frame works in obj_id mode
# ---------------------------------------------------------------------------

def test_single_mask_with_obj_ids_lands_in_correct_channel():
    """Frame mask is [H, W] (2D, single object) with obj_ids=[7].
    Should map to the global channel for obj 7, not channel 0 blindly.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=2, height=h, width=w)
    # Frame 0: only obj 7 (single mask, 2D)
    # Frame 1: obj 3 and obj 7 (multi, 3D) — gives global = [3, 7]
    masks = {
        0: {
            "mask": _make_mask(h, w, region=(0, 2, 0, 2)),  # 2D [H, W]
            "obj_ids": [7],
        },
        1: {
            "mask": torch.stack([
                _make_mask(h, w, region=(2, 4, 0, 2)),  # obj 3
                _make_mask(h, w, region=(0, 2, 2, 4)),  # obj 7
            ], dim=0),
            "obj_ids": [3, 7],
        },
    }
    node = SAM3MaskTracks()
    all_masks, _bbox, track_info_json, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    assert num_objects == 2
    # Sorted global: ch0=3, ch1=7
    assert info["obj_id_mapping"]["0"] == 3
    assert info["obj_id_mapping"]["1"] == 7
    # Frame 0's 2D mask is obj 7 → channel 1
    assert all_masks[0, 1].sum().item() > 0
    # Frame 0 channel 0 (obj 3) should be empty
    assert all_masks[0, 0].sum().item() == 0


# ---------------------------------------------------------------------------
# Test 4: Mismatched length (more obj_ids than channels or vice versa)
# ---------------------------------------------------------------------------

def test_more_obj_ids_than_tensor_channels_truncates_safely():
    """obj_ids list longer than tensor channel dim — should not crash."""
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    masks = {
        0: {
            "mask": torch.stack([_make_mask(h, w, region=(0, 2, 0, 2))], dim=0),  # 1 channel
            "obj_ids": [1, 2, 3],  # 3 ids — extras must be ignored gracefully
        },
    }
    node = SAM3MaskTracks()
    # Should not crash; first-pass uses obj_ids[:local_n].
    all_masks, _bbox, _info, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    # Only obj_id 1 should be claimed (single channel available).
    assert num_objects == 1
    assert all_masks.shape == (1, 1, h, w)


# ---------------------------------------------------------------------------
# Test 5: Empty input produces empty result, doesn't crash
# ---------------------------------------------------------------------------

def test_empty_masks_input_returns_empty_track_info():
    """No frames have masks — return empty result cleanly."""
    state = FakeVideoState(num_frames=5, height=4, width=4)
    node = SAM3MaskTracks()
    _all_masks, _bbox, track_info_json, num_objects = node.extract_tracks(
        masks={}, video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    assert num_objects == 0
    assert info["objects"] == []


# ---------------------------------------------------------------------------
# Test 6: Bbox output respects stable channel mapping
# ---------------------------------------------------------------------------

def test_bbox_output_uses_global_channel_for_each_obj_id():
    """When output_bboxes=True, the bbox at frame T for obj N lands in the
    global channel for N, regardless of where it appeared in the local mask
    tensor that frame.
    """
    h, w = 8, 8
    state = FakeVideoState(num_frames=2, height=h, width=w)
    mask_obj1 = _make_mask(h, w, region=(0, 4, 0, 4))
    mask_obj3 = _make_mask(h, w, region=(4, 8, 4, 8))
    masks = {
        0: {"mask": torch.stack([mask_obj1, mask_obj3], dim=0), "obj_ids": [1, 3]},
        # Frame 1: obj 3 appears FIRST in tensor channel order; with the bug
        # this would land its bbox at channel 0 instead of channel 1.
        1: {"mask": torch.stack([mask_obj3], dim=0), "obj_ids": [3]},
    }
    node = SAM3MaskTracks()
    _all_masks, bbox_masks, _info, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1,
        output_bboxes=True, bbox_padding=0,
    )
    assert num_objects == 2
    # Frame 1 channel 1 (obj 3) should have a filled bbox.
    assert bbox_masks[1, 1].sum().item() > 0
    # Frame 1 channel 0 (obj 1) should be empty (obj 1 not present).
    assert bbox_masks[1, 0].sum().item() == 0


# ---------------------------------------------------------------------------
# Test 7: Numpy input in dict format still works
# ---------------------------------------------------------------------------

def test_numpy_mask_input_with_obj_ids_works():
    """SAM3Propagate emits torch tensors but tests/legacy paths may use numpy."""
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    np_mask = np.zeros((2, h, w), dtype=np.float32)
    np_mask[0, 0:2, 0:2] = 1.0
    np_mask[1, 2:4, 2:4] = 1.0
    masks = {0: {"mask": np_mask, "obj_ids": [5, 11]}}
    node = SAM3MaskTracks()
    all_masks, _bbox, track_info_json, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    assert num_objects == 2
    assert info["obj_id_mapping"]["0"] == 5
    assert info["obj_id_mapping"]["1"] == 11


# ---------------------------------------------------------------------------
# Test 8: Mixed-mode metadata — Codex R2 finding 2026-05-20
# ---------------------------------------------------------------------------

def test_mixed_mode_raw_frame_in_realid_mode_is_skipped_not_silently_mapped():
    """Frame 0 has obj_ids=[1, 3]; Frame 1 is a raw tensor (no obj_ids).
    Pre-R2 fix: raw frame's local channels would silently positional-map into
    sorted real-id channels — placing an UNKNOWN identity at channel 0 (which
    represents SAM3 obj_id 1). Post-R2 fix: the raw frame is SKIPPED with a
    warning, and `inconsistent_metadata_frames` increments in track_info.
    Empty masks for one frame are always safer than wrong identity.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=2, height=h, width=w)
    mask_obj1 = _make_mask(h, w, region=(0, 2, 0, 2))
    mask_obj3 = _make_mask(h, w, region=(0, 2, 2, 4))
    mask_legacy = _make_mask(h, w, region=(2, 4, 0, 2))

    masks = {
        0: {
            "mask": torch.stack([mask_obj1, mask_obj3], dim=0),
            "obj_ids": [1, 3],
        },
        # Frame 1: raw tensor — no obj_ids metadata. Channels could be ANY identity.
        1: torch.stack([mask_legacy, mask_legacy], dim=0),
    }

    node = SAM3MaskTracks()
    all_masks, _bbox, track_info_json, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )

    info = json.loads(track_info_json)
    assert info["id_source"] == "sam3_obj_ids"
    assert info["inconsistent_metadata_frames"] == 1
    assert num_objects == 2
    # Frame 0 lands correctly.
    torch.testing.assert_close(all_masks[0, 0], mask_obj1)
    torch.testing.assert_close(all_masks[0, 1], mask_obj3)
    # Frame 1 is SKIPPED — both channels remain zero. NOT silently mis-mapped.
    assert all_masks[1, 0].sum().item() == 0
    assert all_masks[1, 1].sum().item() == 0


def test_mixed_mode_empty_obj_ids_frame_in_realid_mode_is_also_skipped():
    """Same as above but the legacy-shape frame uses dict with empty obj_ids
    instead of raw tensor. Same safety contract applies.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=2, height=h, width=w)
    masks = {
        0: {
            "mask": torch.stack([
                _make_mask(h, w, region=(0, 2, 0, 2)),
                _make_mask(h, w, region=(0, 2, 2, 4)),
            ], dim=0),
            "obj_ids": [1, 3],
        },
        1: {
            "mask": torch.stack([_make_mask(h, w, region=(2, 4, 0, 2))], dim=0),
            "obj_ids": [],   # Explicitly empty — treated same as missing.
        },
    }
    node = SAM3MaskTracks()
    all_masks, _bbox, track_info_json, _n = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    assert info["inconsistent_metadata_frames"] == 1
    # Frame 1 mask should be zeros (skipped).
    assert all_masks[1, 0].sum().item() == 0
    assert all_masks[1, 1].sum().item() == 0


# ---------------------------------------------------------------------------
# Test 9: 4D tensor input — Codex R2 finding
# ---------------------------------------------------------------------------

def test_4d_input_with_batch_size_one_is_supported():
    """A frame mask shaped [1, N, H, W] (single batch) should be squeezed and
    processed correctly. Common output shape from SAM3 propagate.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    inner = torch.stack([
        _make_mask(h, w, region=(0, 2, 0, 2)),
        _make_mask(h, w, region=(2, 4, 2, 4)),
    ], dim=0)
    # Add leading batch dim — shape becomes [1, 2, H, W]
    batched = inner.unsqueeze(0)
    assert batched.dim() == 4 and batched.shape[0] == 1

    masks = {0: {"mask": batched, "obj_ids": [1, 2]}}
    node = SAM3MaskTracks()
    all_masks, _bbox, _info, num_objects = node.extract_tracks(
        masks=masks, video_state=state, min_visible_pixels=1
    )
    assert num_objects == 2
    assert all_masks[0, 0].sum().item() > 0
    assert all_masks[0, 1].sum().item() > 0


def test_scores_in_mixed_mode_do_not_positional_map():
    """Both R3 reviewers flagged: scores block had the same mixed-mode hole as
    masks. If `scores` includes a frame whose mask dict has no obj_ids (raw
    tensor / empty list) but other frames DO have obj_ids, pre-fix code
    positional-mapped the scores into stable real-id channels — silently
    skewing avg_score. Post-fix: scores from such frames are dropped.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=2, height=h, width=w)
    # Frame 0: real-id mode with obj_ids [1, 3]
    # Frame 1: raw tensor (skipped for masks per earlier test)
    masks = {
        0: {
            "mask": torch.stack([
                _make_mask(h, w, region=(0, 2, 0, 2)),
                _make_mask(h, w, region=(0, 2, 2, 4)),
            ], dim=0),
            "obj_ids": [1, 3],
        },
        1: torch.stack([_make_mask(h, w, region=(2, 4, 0, 2))], dim=0),  # raw
    }
    # Scores: frame 0 = [0.9, 0.8] real-id, frame 1 = [0.5] raw
    scores = {
        0: torch.tensor([[0.9, 0.8]]),
        1: torch.tensor([[0.5]]),   # would have positional-mapped to channel 0
    }
    node = SAM3MaskTracks()
    _all_masks, _bbox, track_info_json, _n = node.extract_tracks(
        masks=masks, video_state=state, scores=scores, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    # Channel 0 (SAM3 obj_id 1) should have ONE score (0.9), not two.
    # Pre-fix: would also append 0.5 from frame 1, dragging avg down to 0.7.
    objs = {o["id"]: o for o in info["objects"]}
    assert objs[0]["avg_score"] == pytest.approx(0.9, abs=1e-4), (
        f"Channel 0 avg_score should be 0.9 (only frame 0's score), "
        f"got {objs[0]['avg_score']} — scores from raw-mode frame leaked in."
    )


def test_4d_input_with_batch_size_greater_than_one_raises():
    """A frame mask shaped [B>1, N, H, W] violates the SAM3 propagate contract.
    Pre-R2 fix: squeeze(0) was a no-op for B>1; tensor stayed 4D; neither
    extraction branch ran; frame was silently dropped. R2 fix: raise loud.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    bad = torch.zeros(2, 2, h, w)  # B=2, N=2 — invalid
    masks = {0: {"mask": bad, "obj_ids": [1, 2]}}
    node = SAM3MaskTracks()
    with pytest.raises(ValueError, match="batch>1"):
        node.extract_tracks(masks=masks, video_state=state, min_visible_pixels=1)


# ---------------------------------------------------------------------------
# Test 10: track_info v2 — name<->obj_id + obj_id<->channel tables
# (linchpin for NV_SAM3SelectMask name mode + NV_SAM3MaskRouter)
# ---------------------------------------------------------------------------

def _two_obj_masks(h, w):
    """Frame with obj_ids [1, 2] → sorted global channels ch0=1, ch1=2."""
    return {
        0: {
            "mask": torch.stack([
                _make_mask(h, w, region=(0, 2, 0, 2)),
                _make_mask(h, w, region=(2, 4, 2, 4)),
            ], dim=0),
            "obj_ids": [1, 2],
        },
    }


def test_v2_always_emits_obj_id_channel_tables_without_subject_map():
    """Even with no subject_map wired, track_info v2 carries the obj_id<->channel
    tables + version + subject_map_source='none'. Back-compat keys still present.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    node = SAM3MaskTracks()
    _m, _b, track_info_json, _n = node.extract_tracks(
        masks=_two_obj_masks(h, w), video_state=state, min_visible_pixels=1
    )
    info = json.loads(track_info_json)
    assert info["version"] == "nv_sam3_mask_tracks.v2"
    assert info["subject_map_source"] == "none"
    assert info["obj_id_to_channel"] == {"1": 0, "2": 1}
    assert info["channel_to_obj_id"] == {"0": 1, "1": 2}
    # back-compat keys untouched
    assert info["obj_id_mapping"] == {"0": 1, "1": 2}
    assert "subject_map" not in info  # not emitted when unwired


def test_v2_subject_map_seedbuilder_form_embeds_name_tables():
    """SeedBuilder {"subjects":[{obj_id,name}]} form → subject_map +
    subject_to_obj_id embedded; declared-and-tracked → subjects_present.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    sm = json.dumps({"subjects": [
        {"obj_id": 1, "name": "head"},
        {"obj_id": 2, "name": "body"},
    ]})
    node = SAM3MaskTracks()
    _m, _b, track_info_json, _n = node.extract_tracks(
        masks=_two_obj_masks(h, w), video_state=state,
        min_visible_pixels=1, subject_map=sm,
    )
    info = json.loads(track_info_json)
    assert info["subject_map_source"] == "input"
    assert info["subject_map"] == {"1": "head", "2": "body"}
    assert info["subject_to_obj_id"] == {"head": 1, "body": 2}
    assert sorted(info["subjects_present"]) == ["body", "head"]
    assert info["subjects_missing"] == []
    # Full name->obj_id->channel chain resolvable from track_info alone:
    oid = info["subject_to_obj_id"]["body"]            # 2
    ch = info["obj_id_to_channel"][str(oid)]           # 1
    assert ch == 1


def test_v2_subject_map_plain_dict_form_accepted():
    """Plain {obj_id: name} dict form is also accepted."""
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    node = SAM3MaskTracks()
    _m, _b, track_info_json, _n = node.extract_tracks(
        masks=_two_obj_masks(h, w), video_state=state,
        min_visible_pixels=1, subject_map=json.dumps({"1": "head", "2": "body"}),
    )
    info = json.loads(track_info_json)
    assert info["subject_to_obj_id"] == {"head": 1, "body": 2}


def test_v2_declared_subject_not_tracked_lands_in_missing():
    """A subject declared upstream but never produced by SAM3 (obj_id 3 here)
    must appear in subjects_missing, NOT subjects_present, so the downstream
    resolver can raise a clear error instead of silently emitting zeros.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    sm = json.dumps({"subjects": [
        {"obj_id": 1, "name": "head"},
        {"obj_id": 2, "name": "body"},
        {"obj_id": 3, "name": "hair"},   # never tracked
    ]})
    node = SAM3MaskTracks()
    _m, _b, track_info_json, _n = node.extract_tracks(
        masks=_two_obj_masks(h, w), video_state=state,
        min_visible_pixels=1, subject_map=sm,
    )
    info = json.loads(track_info_json)
    assert "hair" in info["subjects_missing"]
    assert "hair" not in info["subjects_present"]
    # hair has no channel — resolver must not find one.
    assert "3" not in info["obj_id_to_channel"]


def test_v2_malformed_subject_map_json_does_not_crash_render():
    """A malformed subject_map must be ignored with a warning, never crash —
    the segmentation result is far more expensive than the name metadata.
    """
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    node = SAM3MaskTracks()
    _m, _b, track_info_json, num_objects = node.extract_tracks(
        masks=_two_obj_masks(h, w), video_state=state,
        min_visible_pixels=1, subject_map="{not valid json",
    )
    info = json.loads(track_info_json)
    assert num_objects == 2  # render unaffected
    # source is "input" (a map was supplied) but tables are empty + warned.
    assert info["subject_map_source"] == "input"
    assert info["subject_to_obj_id"] == {}
    assert any("not valid JSON" in w for w in info.get("subject_map_warnings", []))


def test_v2_duplicate_name_in_subject_map_is_warned_first_wins():
    """Duplicate subject name → first obj_id kept, collision warned (the
    resolver relies on a unique name->obj_id table)."""
    h, w = 4, 4
    state = FakeVideoState(num_frames=1, height=h, width=w)
    sm = json.dumps({"subjects": [
        {"obj_id": 1, "name": "head"},
        {"obj_id": 2, "name": "head"},   # dup name
    ]})
    node = SAM3MaskTracks()
    _m, _b, track_info_json, _n = node.extract_tracks(
        masks=_two_obj_masks(h, w), video_state=state,
        min_visible_pixels=1, subject_map=sm,
    )
    info = json.loads(track_info_json)
    assert info["subject_to_obj_id"] == {"head": 1}  # first wins
    assert any("duplicate name" in w for w in info.get("subject_map_warnings", []))


def test_v2_empty_objects_path_still_emits_v2_keys():
    """When SAM3 produced no tracks, the empty-return track_info must still
    carry v2 keys (+ declared subjects as missing) so a downstream name select
    fails with a helpful 'available subjects' message rather than KeyError.
    """
    state = FakeVideoState(num_frames=3, height=4, width=4)
    sm = json.dumps({"subjects": [{"obj_id": 1, "name": "head"}]})
    node = SAM3MaskTracks()
    _m, _b, track_info_json, num_objects = node.extract_tracks(
        masks={}, video_state=state, min_visible_pixels=1, subject_map=sm,
    )
    info = json.loads(track_info_json)
    assert num_objects == 0
    assert info["version"] == "nv_sam3_mask_tracks.v2"
    assert info["obj_id_to_channel"] == {}
    assert info["subject_to_obj_id"] == {"head": 1}
    assert info["subjects_missing"] == ["head"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
