"""Tests for SAM3MaskRefine compose helpers (2026-05-20 audit Bug #2).

Covers the pure-function helpers added to honor BOTH mask + corrective
prompts at the same (frame_idx, obj_id) without violating SAM3's tracker-
level mutual-exclusion contract:

  - _resolve_compose_op: mode dispatch + auto-detection from prompt polarity
  - _compose_masks: union / intersect / replace / diff element-wise ops

End-to-end refine() integration (which calls real SAM3 image-mode + video
propagation) is left to runtime testing — not unit-testable without the
full inference state.

Bypasses sam3_video_nodes.py's heavy imports (folder_paths, comfy) via
importlib spec + targeted module stubs.
"""

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


# --- Stub heavy imports before loading sam3_video_nodes -----------------------

# folder_paths (needs `base_path` which conftest's mock doesn't supply)
_fp = MagicMock()
_fp.base_path = "/tmp"
_fp.models_dir = "/tmp/models"
_fp.get_folder_paths = lambda x: ["/tmp"]
sys.modules.setdefault("folder_paths", _fp)

# comfy + comfy.model_management
_comfy = MagicMock()
_comfy_mm = MagicMock()
_comfy_mm.get_torch_device = lambda: "cpu"
_comfy_mm.unet_offload_device = lambda: "cpu"
_comfy_mm.throw_exception_if_processing_interrupted = lambda: None
_comfy.model_management = _comfy_mm
sys.modules.setdefault("comfy", _comfy)
sys.modules.setdefault("comfy.model_management", _comfy_mm)
sys.modules.setdefault("comfy.utils", MagicMock())


# A working comfy_image_to_pil stub — must return an object with `.size = (W, H)`.
class _FakePIL:
    """Minimal stand-in for PIL.Image — exposes only the `.size` tuple
    `_run_image_mode_segmentation` reads.
    """
    def __init__(self, w, h):
        self.size = (w, h)


def _fake_image_to_pil(img_batch):
    """img_batch is a [B, H, W, C] tensor; return a fake PIL with .size = (W, H)."""
    if hasattr(img_batch, "shape") and len(img_batch.shape) >= 3:
        # Tensor [B, H, W, C] → (W, H)
        h, w = int(img_batch.shape[1]), int(img_batch.shape[2])
    else:
        w, h = 8, 8
    return _FakePIL(w, h)


# --- Load video_state then sam3_video_nodes via importlib ---------------------

_nvfork_dir = pathlib.Path(__file__).resolve().parent.parent
_pkg_alias = "_sam3_refine_compose_test_pkg"
_pkg = type(sys)(_pkg_alias)
_pkg.__path__ = [str(_nvfork_dir / "nodes")]
sys.modules[_pkg_alias] = _pkg

# Load video_state as the submodule (sam3_video_nodes imports `.video_state`)
_video_state_path = _nvfork_dir / "nodes" / "video_state.py"
_video_state_spec = importlib.util.spec_from_file_location(
    f"{_pkg_alias}.video_state", str(_video_state_path)
)
_video_state_mod = importlib.util.module_from_spec(_video_state_spec)
_video_state_spec.loader.exec_module(_video_state_mod)
sys.modules[f"{_pkg_alias}.video_state"] = _video_state_mod

# Stub the other relative imports sam3_video_nodes pulls
sys.modules[f"{_pkg_alias}.inference_reconstructor"] = MagicMock()
sys.modules[f"{_pkg_alias}.sam3_model_patcher"] = MagicMock()
# utils stub with a working comfy_image_to_pil (the
# _run_image_mode_segmentation helper does `from .utils import ...` at call
# time, so we wire the real lambda onto the module before any test runs).
_utils_mod = MagicMock()
_utils_mod.comfy_image_to_pil = _fake_image_to_pil
sys.modules[f"{_pkg_alias}.utils"] = _utils_mod

# Load sam3_video_nodes
_target_path = _nvfork_dir / "nodes" / "sam3_video_nodes.py"
_target_spec = importlib.util.spec_from_file_location(
    f"{_pkg_alias}.sam3_video_nodes", str(_target_path)
)
_target_mod = importlib.util.module_from_spec(_target_spec)
_target_spec.loader.exec_module(_target_mod)

_resolve_compose_op = _target_mod._resolve_compose_op
_compose_masks = _target_mod._compose_masks
_validate_compose_mode = _target_mod._validate_compose_mode
_VALID_COMPOSE_MODES = _target_mod._VALID_COMPOSE_MODES
_run_image_mode_segmentation = _target_mod._run_image_mode_segmentation
SAM3MaskRefine = _target_mod.SAM3MaskRefine


# ============================================================================
# _resolve_compose_op — mode dispatch + auto-detection
# ============================================================================

@pytest.mark.parametrize("mode", ["union", "intersect", "replace", "diff"])
def test_explicit_mode_returned_verbatim(mode):
    """Explicit modes always return their own name regardless of polarity."""
    assert _resolve_compose_op(mode, has_positive=True, has_negative=True) == mode
    assert _resolve_compose_op(mode, has_positive=True, has_negative=False) == mode
    assert _resolve_compose_op(mode, has_positive=False, has_negative=True) == mode
    assert _resolve_compose_op(mode, has_positive=False, has_negative=False) == mode


def test_auto_positive_only_uses_union():
    """positive-only corrective prompts → extend mask via union."""
    assert _resolve_compose_op("auto", has_positive=True, has_negative=False) == "union"


def test_auto_negative_only_uses_diff():
    """negative-only corrective prompts → subtract region from mask."""
    assert _resolve_compose_op("auto", has_positive=False, has_negative=True) == "diff"


def test_auto_mixed_uses_replace():
    """Both polarities present → trust image-mode result (replace)."""
    assert _resolve_compose_op("auto", has_positive=True, has_negative=True) == "replace"


def test_auto_no_prompts_falls_through_to_union():
    """Edge case: auto called with no polarity flags. Falls through to union
    (caller should also have short-circuited; this is defensive).
    """
    assert _resolve_compose_op("auto", has_positive=False, has_negative=False) == "union"


# ============================================================================
# _compose_masks — element-wise ops
# ============================================================================

def _quad_mask(h, w, region):
    """Build [H, W] float mask with given region filled."""
    m = torch.zeros(h, w)
    y0, y1, x0, x1 = region
    m[y0:y1, x0:x1] = 1.0
    return m


def test_union_takes_max_of_both_masks():
    """Union = element-wise max. Result covers both regions."""
    h, w = 4, 4
    base = _quad_mask(h, w, (0, 2, 0, 2))    # top-left
    corrective = _quad_mask(h, w, (2, 4, 2, 4))  # bottom-right
    out = _compose_masks(base, corrective, "union")
    # Both quadrants present
    assert out[0, 0].item() == 1.0  # top-left from base
    assert out[3, 3].item() == 1.0  # bottom-right from corrective
    assert out[1, 3].item() == 0.0  # neither
    assert out.sum().item() == 8.0  # 4 + 4


def test_intersect_takes_min_of_both_masks():
    """Intersect = element-wise min. Result is the overlap only."""
    h, w = 4, 4
    base = _quad_mask(h, w, (0, 3, 0, 3))    # 3x3 top-left
    corrective = _quad_mask(h, w, (1, 4, 1, 4))  # 3x3 shifted
    out = _compose_masks(base, corrective, "intersect")
    # Overlap is 2x2 region (1:3, 1:3)
    assert out[1, 1].item() == 1.0
    assert out[2, 2].item() == 1.0
    assert out[0, 0].item() == 0.0  # base only
    assert out[3, 3].item() == 0.0  # corrective only
    assert out.sum().item() == 4.0


def test_replace_discards_base_entirely():
    """Replace returns corrective mask verbatim, base ignored."""
    h, w = 4, 4
    base = _quad_mask(h, w, (0, 2, 0, 2))
    corrective = _quad_mask(h, w, (2, 4, 2, 4))
    out = _compose_masks(base, corrective, "replace")
    # Base's top-left is gone; only corrective's bottom-right remains
    assert out[0, 0].item() == 0.0
    assert out[3, 3].item() == 1.0
    assert out.sum().item() == 4.0


def test_replace_returns_clone_not_alias():
    """Replace must return a clone — mutating output should not affect input."""
    h, w = 4, 4
    base = torch.zeros(h, w)
    corrective = torch.ones(h, w)
    out = _compose_masks(base, corrective, "replace")
    out.fill_(0.5)
    assert corrective[0, 0].item() == 1.0, "replace returned an alias, not a clone"


def test_diff_subtracts_corrective_from_base():
    """Diff = base AND NOT corrective. Removes the corrective region from base."""
    h, w = 4, 4
    base = torch.ones(h, w)                   # full mask
    corrective = _quad_mask(h, w, (0, 2, 0, 2))  # top-left to subtract
    out = _compose_masks(base, corrective, "diff")
    # Top-left zeroed, rest of base preserved
    assert out[0, 0].item() == 0.0
    assert out[3, 3].item() == 1.0
    assert out.sum().item() == 12.0  # 16 - 4


def test_unknown_op_raises():
    """Defensive: unknown op string should raise with clear message."""
    base = torch.zeros(4, 4)
    corrective = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="Unknown compose op"):
        _compose_masks(base, corrective, "bogus")


# ============================================================================
# refine() input validation — mode whitelist
# ============================================================================

def test_validate_compose_mode_accepts_all_valid_modes():
    """R1 review fold-in (Codex): validation factored into a helper so each
    mode can be tested directly without dragging in the refine() pipeline.
    """
    for mode in _VALID_COMPOSE_MODES:
        # Should not raise.
        _validate_compose_mode(mode)


def test_validate_compose_mode_rejects_invalid_mode():
    """Helper raises ValueError with the expected diagnostic on bad input."""
    with pytest.raises(ValueError, match="corrective_compose_mode"):
        _validate_compose_mode("bogus_mode")


def test_refine_rejects_invalid_compose_mode():
    """Invalid corrective_compose_mode must raise before any heavy work."""
    node = SAM3MaskRefine()
    fake_model = MagicMock()
    fake_frames = torch.zeros(2, 4, 4, 3)
    fake_masks = torch.zeros(2, 4, 4)
    with pytest.raises(ValueError, match="corrective_compose_mode"):
        node.refine(
            sam3_model=fake_model,
            video_frames=fake_frames,
            input_masks=fake_masks,
            corrective_compose_mode="bogus_mode",
        )


def test_refine_accepts_all_valid_compose_modes_at_validation_stage():
    """All 6 valid modes should pass the early validation. We don't run the
    full refine() pipeline (needs real SAM3 model + inference state), but the
    validation block is the first thing refine() does, so we can verify mode
    acceptance by triggering a DIFFERENT downstream failure (e.g. invalid
    tensor) AFTER mode validation passes.
    """
    node = SAM3MaskRefine()
    fake_model = MagicMock()
    # Empty masks → known downstream "no valid prompts" branch eventually.
    # We just need NOT to hit the mode-validation raise.
    fake_frames = torch.zeros(1, 4, 4, 3)
    fake_masks = torch.zeros(1, 4, 4)
    for mode in ("skip", "auto", "union", "intersect", "replace", "diff"):
        with pytest.raises(Exception) as excinfo:
            node.refine(
                sam3_model=fake_model,
                video_frames=fake_frames,
                input_masks=fake_masks,
                corrective_compose_mode=mode,
            )
        # The raise we're catching here should NOT be the mode-validation one.
        assert "corrective_compose_mode" not in str(excinfo.value), (
            f"Mode '{mode}' was rejected by validation when it should be accepted: "
            f"{excinfo.value}"
        )


def test_refine_none_compose_mode_defaults_to_skip():
    """When compose mode comes in as None (older workflow), default to skip."""
    node = SAM3MaskRefine()
    fake_model = MagicMock()
    fake_frames = torch.zeros(1, 4, 4, 3)
    fake_masks = torch.zeros(1, 4, 4)
    # Should not raise on validation; will fail downstream for other reasons.
    with pytest.raises(Exception) as excinfo:
        node.refine(
            sam3_model=fake_model,
            video_frames=fake_frames,
            input_masks=fake_masks,
            corrective_compose_mode=None,
        )
    assert "corrective_compose_mode" not in str(excinfo.value)


# ============================================================================
# _run_image_mode_segmentation — multi-object parsing, neg-box-as-point,
# no-positive-anchor short-circuit, exception cleanup
# (all R1 review fold-ins from Codex + Gemini)
# ============================================================================

class _FakeModel:
    """Stand-in for sam3_model.processor.model. Captures predict_inst args."""
    def __init__(self, raise_on_predict=False):
        self.inst_interactive_predictor = object()  # truthy
        self.predict_inst_calls = []
        self._raise = raise_on_predict

    def predict_inst(self, state, point_coords=None, point_labels=None,
                     box=None, mask_input=None, multimask_output=True,
                     normalize_coords=True):
        self.predict_inst_calls.append({
            "point_coords": point_coords,
            "point_labels": point_labels,
            "box": box,
            "mask_input": mask_input,
            "multimask_output": multimask_output,
            "normalize_coords": normalize_coords,
        })
        if self._raise:
            raise RuntimeError("simulated predict_inst failure")
        masks = np.ones((1, 4, 4), dtype=np.float32)
        scores = np.array([1.0], dtype=np.float32)
        return masks, scores, None


class _FakeProcessor:
    def __init__(self, model):
        self.model = model

    def set_image(self, pil_image):
        return {"backbone_out": {}}

    def sync_device_with_model(self):
        pass


class _FakeSAM3Model:
    def __init__(self, raise_on_predict=False):
        self.processor = _FakeProcessor(_FakeModel(raise_on_predict=raise_on_predict))


def test_multi_object_dict_extracts_both_polarities():
    """R1 review fold-in (Gemini critical): pre-fix parser dropped negatives
    embedded in the multi-object dict (`objects[].negative_points`). Post-fix:
    when multi-object format is encountered, BOTH polarities are extracted.
    """
    sam3 = _FakeSAM3Model()
    frame = torch.zeros(1, 8, 8, 3)
    pos_dict = {"objects": [{
        "obj_id": 1,
        "positive_points": [[0.25, 0.5], [0.75, 0.5]],  # 2 positive
        "negative_points": [[0.5, 0.25]],                # 1 negative
    }]}
    out = _run_image_mode_segmentation(sam3, frame, pos_dict, None, None, None)
    assert out is not None
    call = sam3.processor.model.predict_inst_calls[-1]
    assert call["point_labels"].tolist() == [1, 1, 0], (
        "multi-object negatives must be extracted alongside positives"
    )
    # Pixel coords: 0.25*8=2, 0.75*8=6, 0.5*8=4; H=8 so y matches
    coords = call["point_coords"].tolist()
    assert coords[0] == pytest.approx([2.0, 4.0])
    assert coords[1] == pytest.approx([6.0, 4.0])
    assert coords[2] == pytest.approx([4.0, 2.0])


def test_negative_box_encoded_as_negative_point_at_centroid():
    """R1 review fold-in (Gemini): pre-fix dropped negative_boxes; post-fix
    encodes each negative box's centroid as a negative point so predict_inst
    incorporates it into the corrective mask.
    """
    sam3 = _FakeSAM3Model()
    frame = torch.zeros(1, 8, 8, 3)
    pos_pts = {"points": [[0.5, 0.5]], "labels": [1]}        # 1 positive anchor
    neg_box = {"boxes": [[0.25, 0.75, 0.1, 0.1]], "labels": [False]}  # cx, cy, w, h
    out = _run_image_mode_segmentation(sam3, frame, pos_pts, None, None, neg_box)
    assert out is not None
    call = sam3.processor.model.predict_inst_calls[-1]
    assert call["point_labels"].tolist() == [1, 0]
    # Negative centroid in pixel coords = (0.25 * 8, 0.75 * 8) = (2.0, 6.0)
    coords = call["point_coords"].tolist()
    assert coords[1] == pytest.approx([2.0, 6.0])


def test_no_positive_anchor_short_circuits_to_none():
    """R1 review fold-in (Codex): predict_inst REQUIRES a positive signal.
    Negative-only prompts must short-circuit to None so the caller falls
    through to base mask — never reaching predict_inst.
    """
    sam3 = _FakeSAM3Model()
    frame = torch.zeros(1, 8, 8, 3)
    neg_pts = {"points": [[0.5, 0.5]], "labels": [0]}
    out = _run_image_mode_segmentation(sam3, frame, None, neg_pts, None, None)
    assert out is None
    assert sam3.processor.model.predict_inst_calls == [], (
        "predict_inst must not be called when only negative prompts exist"
    )


def test_no_prompts_short_circuits_to_none():
    """No prompts at all → return None immediately, no model call."""
    sam3 = _FakeSAM3Model()
    frame = torch.zeros(1, 8, 8, 3)
    out = _run_image_mode_segmentation(sam3, frame, None, None, None, None)
    assert out is None
    assert sam3.processor.model.predict_inst_calls == []


def test_predict_inst_exception_propagates_with_cleanup():
    """R1 review fold-in (Codex): predict_inst wrapped in try/finally so
    state cleanup runs even on exception. The exception itself propagates.
    """
    sam3 = _FakeSAM3Model(raise_on_predict=True)
    frame = torch.zeros(1, 8, 8, 3)
    pos_pts = {"points": [[0.5, 0.5]], "labels": [1]}
    with pytest.raises(RuntimeError, match="simulated predict_inst failure"):
        _run_image_mode_segmentation(sam3, frame, pos_pts, None, None, None)
    # If the finally cleanup didn't run we'd get cascading issues; the test
    # verifies the exception did propagate cleanly.


def test_no_predictor_available_returns_none():
    """Defensive: if model.inst_interactive_predictor is None (model loaded
    without enable_inst_interactivity), return None and log — don't crash.
    """
    sam3 = _FakeSAM3Model()
    sam3.processor.model.inst_interactive_predictor = None
    frame = torch.zeros(1, 8, 8, 3)
    pos_pts = {"points": [[0.5, 0.5]], "labels": [1]}
    out = _run_image_mode_segmentation(sam3, frame, pos_pts, None, None, None)
    assert out is None


# ============================================================================
# refine() Step 3 suppression — R2 review fold-in (Codex Medium)
#
# Verifies that when the user requested compose mode but image-mode returned
# None (negative-only, no predictor, empty result), the direct corrective
# prompts are STILL suppressed in Step 3 — otherwise add_new_points_or_box
# would trigger SAM3's tracker-level mask clear on the base mask we just
# added at the corrective frame.
#
# We exercise this by monkeypatching `_run_image_mode_segmentation` on the
# loaded module to force a None return, then driving refine() far enough to
# see whether positive_points/etc. were nulled out before Step 3's add loop.
# A direct end-to-end run isn't possible without the full SAM3 model, but we
# can inspect the suppression branch by reading the post-Step-2 state.
# ============================================================================

def test_compose_mode_suppresses_direct_prompts_even_when_image_mode_falls_through(monkeypatch):
    """R3 regression for Codex R2 finding: compose_active=True + image-mode
    returns None must STILL suppress direct corrective prompts in Step 3.
    """
    # Force _run_image_mode_segmentation to return None to simulate the
    # negative-only / no-predictor / empty-result fall-through path.
    monkeypatch.setattr(
        _target_mod,
        "_run_image_mode_segmentation",
        lambda *args, **kwargs: None,
    )

    # Spy on VideoPrompt.create_point and VideoPrompt.create_box to detect any
    # direct corrective prompt being added after the mask keyframe. If
    # suppression works, neither should be called for the corrective frame.
    VideoPrompt = _video_state_mod.VideoPrompt
    point_calls = []
    box_calls = []
    original_create_point = VideoPrompt.create_point
    original_create_box = VideoPrompt.create_box

    def _spy_point(frame_idx, obj_id, points, labels):
        point_calls.append((frame_idx, obj_id, list(points), list(labels)))
        return original_create_point(frame_idx, obj_id, points, labels)

    def _spy_box(frame_idx, obj_id, box, is_positive=True):
        box_calls.append((frame_idx, obj_id, list(box), is_positive))
        return original_create_box(frame_idx, obj_id, box, is_positive)

    monkeypatch.setattr(VideoPrompt, "create_point", staticmethod(_spy_point))
    monkeypatch.setattr(VideoPrompt, "create_box", staticmethod(_spy_box))

    node = SAM3MaskRefine()
    fake_model = MagicMock()
    # 1 frame, non-empty mask at frame 0 (so it doesn't get skipped by the
    # 100-pixel threshold).
    frames = torch.zeros(1, 8, 8, 3)
    masks = torch.ones(1, 8, 8)  # 64 px — wait, threshold is 100; bump up.

    # Make mask large enough to pass the empty-skip threshold (>= 100 pixels).
    frames = torch.zeros(1, 16, 16, 3)
    masks = torch.ones(1, 16, 16)  # 256 px

    # Wire compose-active inputs: negative-only points trigger compose_active
    # but image-mode returns None per monkeypatch above → fall-through path.
    neg_pts = {"points": [[0.5, 0.5]], "labels": [0]}
    pos_box = {"boxes": [[0.5, 0.5, 0.3, 0.3]], "labels": [True]}

    # Drive refine() to completion. It will fail downstream when it tries to
    # propagate without a real SAM3 model, but Step 2 + Step 3 run first and
    # that's what we're inspecting.
    try:
        node.refine(
            sam3_model=fake_model,
            video_frames=frames,
            input_masks=masks,
            keyframe_interval=10,
            positive_points=None,
            negative_points=neg_pts,
            positive_boxes=pos_box,
            negative_boxes=None,
            frame_idx=0,
            obj_id=1,
            corrective_compose_mode="auto",
        )
    except Exception:
        # Expected — fails downstream when reaching real SAM3 propagation.
        pass

    # Step 3 should NOT have added any direct point or box at the corrective
    # frame (frame 0). pos_box was wired — pre-R3 fix it would have been added.
    corrective_point_adds = [c for c in point_calls if c[0] == 0]
    corrective_box_adds = [c for c in box_calls if c[0] == 0]
    assert corrective_point_adds == [], (
        f"compose_active should suppress direct point prompts at corrective "
        f"frame even when image-mode returned None, but got: {corrective_point_adds}"
    )
    assert corrective_box_adds == [], (
        f"compose_active should suppress direct box prompts at corrective "
        f"frame even when image-mode returned None, but got: {corrective_box_adds}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
