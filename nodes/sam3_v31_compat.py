"""
Node-layer compatibility shims for the SAM 3.1 multiplex predictor.

The fork's node layer (nodes/inference_reconstructor.py + sam3_video_nodes.py)
was written against the fork's legacy Sam3VideoPredictor wrapper. Meta's upstream
Sam3MultiplexVideoPredictor (vendored in sam3_lib_v31) is API-compatible in shape
but differs in two small ways that crash at propagate time. This module patches a
freshly-built multiplex predictor IN PLACE so the existing node layer drives it
unchanged. All shims are additive and defensive; the upstream vendored code is
left pristine.

Call apply_node_layer_compat(predictor) once, right after
build_sam3_multiplex_video_predictor(...).

Shims:
  1. SESSION REGISTRY ALIAS — the reconstructor reads the live session dict as
     `_ALL_INFERENCE_STATES` (upper, the legacy name); upstream stores the SAME
     dict as the instance attr `_all_inference_states` (lower) and only mutates it
     in place. Alias both names to one dict object.
  2. start_session SIGNATURE FILTER — upstream Sam3BasePredictor.start_session
     forwards `offload_state_to_cpu` (and possibly `video_loader_type`) to
     `self.model.init_state()`, but the multiplex init_state signature is
     (resource_path, offload_video_to_cpu, async_loading_frames, use_torchcodec,
     use_cv2, input_is_mp4) — it rejects those kwargs with a TypeError. Replace
     start_session with a version that filters init_state kwargs to what the
     model actually accepts (mirrors upstream's own add_prompt/propagate
     filtering, and the fork-3.0 behavior of stashing offload_state_to_cpu on the
     state instead of passing it down).

Known still-deferred (not on the text/point smoke path; fix when hit):
  - add_new_mask (mask-prompt chunked continuation) is absent on the base
    predictor — needs a multiplex mask-add path.
"""

import inspect
import time
import types
import uuid

_PREFIX = "[SAM31-compat]"


def apply_node_layer_compat(predictor):
    """Patch a built multiplex predictor in place. Returns it for chaining."""
    notes = []
    if _alias_session_registry(predictor):
        notes.append("session-registry alias")
    if _patch_start_session(predictor):
        notes.append("start_session kwarg-filter")
    if _patch_add_new_mask_guard(predictor):
        notes.append("add_new_mask guard")
    print(f"{_PREFIX} applied: {', '.join(notes) if notes else '(nothing - already compatible?)'}")
    return predictor


def _alias_session_registry(predictor) -> bool:
    if hasattr(predictor, "_all_inference_states") and not hasattr(predictor, "_ALL_INFERENCE_STATES"):
        predictor._ALL_INFERENCE_STATES = predictor._all_inference_states  # same dict object
        return True
    return False


def _patch_start_session(predictor) -> bool:
    if not hasattr(predictor, "model") or not hasattr(predictor.model, "init_state"):
        print(f"{_PREFIX} WARN: predictor.model.init_state missing - cannot patch start_session.")
        return False

    def start_session(self, resource_path, session_id=None,
                      offload_video_to_cpu=False, offload_state_to_cpu=False):
        # Build the superset of kwargs the legacy node layer expects to pass down...
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type

        # ...then keep only what THIS model's init_state actually accepts
        # (multiplex init_state rejects offload_state_to_cpu / video_loader_type).
        try:
            sig = inspect.signature(self.model.init_state)
            has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kw:
                init_kwargs = {k: v for k, v in init_kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            pass  # builtin/uninspectable — pass through unfiltered

        inference_state = self.model.init_state(**init_kwargs)

        # Preserve offload_state_to_cpu on the state for any downstream tracker
        # use (mirrors fork-3.0 Sam3VideoPredictor.start_session). Harmless if
        # the multiplex tracker never reads it.
        try:
            inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        except Exception:  # noqa: BLE001 — state may not be subscriptable
            pass

        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = types.MethodType(start_session, predictor)
    return True


def _patch_add_new_mask_guard(predictor) -> bool:
    """Install a legible-error add_new_mask on predictors that lack it.

    inference_reconstructor._apply_prompt calls model.add_new_mask(...) for
    MASK-type prompts (chunked mask-guided continuation). Upstream
    Sam3BasePredictor has no add_new_mask, so this would otherwise raise a bare
    AttributeError deep in the stack. The multiplex model DOES expose a real
    mask-add path (Sam3Multiplex*.add_new_masks / add_new_masks_to_existing_state)
    that a future shim can wire to; until then, fail with a clear pointer instead
    of a confusing crash. Not on the text/point smoke path.
    """
    if hasattr(predictor, "add_new_mask"):
        return False  # something already provides it — leave alone

    def add_new_mask(self, session_id, frame_idx, obj_id, mask, *args, **kwargs):
        raise NotImplementedError(
            "[SAM31] add_new_mask is not yet wired for the SAM 3.1 multiplex "
            "predictor. This path is only used by chunked / mask-guided "
            "continuation (SAM3 chunked video). The upstream multiplex mask-add "
            "method exists (sam3_lib_v31 video_tracking_multiplex.add_new_masks_"
            "to_existing_state / video_tracking_multiplex_demo.add_new_masks) and "
            "needs a thin adapter in sam3_v31_compat.py. Text/point/box prompts "
            "work without this. (session={}, frame={}, obj={})".format(
                session_id, frame_idx, obj_id)
        )

    predictor.add_new_mask = types.MethodType(add_new_mask, predictor)
    return True
