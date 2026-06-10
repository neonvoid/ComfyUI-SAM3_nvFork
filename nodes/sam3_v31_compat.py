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

  3. add_new_mask ADAPTER — the reconstructor calls singular
     model.add_new_mask(session_id, frame_idx, obj_id, mask) for MASK-type
     prompts (chunked continuation, corrective re-anchoring, mask-seeded flows).
     The multiplex model exposes only a BATCHED add_new_masks(...obj_ids, masks
     [N,H,W]); adapt the singular call to it (obj_ids=[obj_id], masks=mask[1,H,W])
     under bf16 autocast. Falls back to a legible error if a future snapshot
     lacks add_new_masks.
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
    note = _patch_add_new_mask(predictor)
    if note:
        notes.append(note)
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


def _patch_add_new_mask(predictor) -> str:
    """Wire singular add_new_mask onto the multiplex batched add_new_masks.

    inference_reconstructor._apply_prompt calls
    `model.add_new_mask(session_id, frame_idx, obj_id, mask)` for MASK-type
    prompts (chunked mask-guided continuation, corrective re-anchoring,
    mask-seeded workflows). The fork's 3.0 wrapper routed this to the model's
    singular `add_tracker_new_mask`. The multiplex model instead exposes a
    BATCHED entry point:

        model.add_new_masks(inference_state, frame_idx, obj_ids[list], masks[N,H,W],
                            add_mask_to_memory=False, reconditioning=False)

    which resizes/binarizes/device-moves the masks internally and creates new
    objects when reconditioning=False. We adapt the singular call to the batched
    one (obj_ids=[obj_id], masks=mask[1,H,W]) under bf16 autocast (matching the
    predictor's add_prompt path). NOTE: per the SAM3 mutual-exclusion contract,
    add_new_masks pops point inputs for that (frame, obj) — mask and point
    prompts on the same object/frame are alternatives, not additive.

    Falls back to a legible NotImplementedError if a future vendor snapshot
    renames/removes add_new_masks. Returns a note string for the apply log.
    """
    if hasattr(predictor, "add_new_mask"):
        return ""  # already provided — leave alone

    model = getattr(predictor, "model", None)
    has_batched = model is not None and hasattr(model, "add_new_masks")

    if has_batched:
        def add_new_mask(self, session_id, frame_idx, obj_id, mask, *args, **kwargs):
            import torch
            session = self._get_session(session_id)
            inference_state = session["state"]
            if hasattr(self, "_extend_expiration_time"):
                try:
                    self._extend_expiration_time(session)
                except Exception:  # noqa: BLE001
                    pass

            # Normalize mask -> [1, H, W] (mirror fork-3.0 add_new_mask shaping).
            if not isinstance(mask, torch.Tensor):
                import numpy as np
                mask = torch.as_tensor(np.asarray(mask))
            m = mask.squeeze() if mask.dim() > 2 else mask
            if m.dim() != 2:
                raise ValueError(
                    f"[SAM31] add_new_mask expects a 2D mask [H,W] "
                    f"(or squeezable to it), got shape {tuple(mask.shape)}"
                )
            masks = m.unsqueeze(0)  # [1, H, W]

            def _call():
                return self.model.add_new_masks(
                    inference_state=inference_state,
                    frame_idx=int(frame_idx),
                    obj_ids=[int(obj_id)],
                    masks=masks,
                )
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = _call()
            except TypeError:
                # positional fallback if upstream kwarg names ever drift
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = self.model.add_new_masks(
                        inference_state, int(frame_idx), [int(obj_id)], masks
                    )

            # Normalize return to the fork's {frame_index, outputs} shape. The
            # reconstructor ignores it, but stay consistent for any other caller.
            if isinstance(result, tuple) and len(result) == 2:
                fidx, outputs = result
            else:
                fidx, outputs = frame_idx, result
            return {"frame_index": fidx, "outputs": outputs}

        predictor.add_new_mask = types.MethodType(add_new_mask, predictor)
        return "add_new_mask (multiplex batched adapter)"

    # Fallback: legible error instead of a bare AttributeError deep in the stack.
    def add_new_mask(self, session_id, frame_idx, obj_id, mask, *args, **kwargs):
        raise NotImplementedError(
            "[SAM31] add_new_mask: this multiplex model exposes no add_new_masks "
            "method to adapt to (vendor snapshot drift?). Text/point/box prompts "
            "work without this; mask-seeded / chunked continuation does not. "
            "(session={}, frame={}, obj={})".format(session_id, frame_idx, obj_id)
        )

    predictor.add_new_mask = types.MethodType(add_new_mask, predictor)
    return "add_new_mask (guard - no multiplex add_new_masks found)"
