"""
NV_SAM3Diag — copy-paste-ready SAM3 diagnostics for debugging across machines.

Wire a SAM3_MODEL in (optionally the propagate masks too); get back ONE compact
markdown report on a STRING output. Drop a ShowText on it, copy the box, paste it
to your assistant. The report is also printed to console between clear delimiters.

Version-agnostic: works on the legacy SAM3 (3.0) loader AND the new SAM 3.1
multiplex loader, so it doubles as an A/B introspector. Pure introspection — no
inference, no GPU work, never raises (every probe is defensive).

What it reports:
  - which model/predictor classes are live (multiplex vs legacy)
  - which session-management methods the predictor exposes (the exact surface the
    node layer drives — start_session/add_prompt/add_new_mask/propagate/...)
  - the session registry name + active-session count (the thing that crashed at
    bring-up: _ALL_INFERENCE_STATES vs _all_inference_states)
  - which _apply_config tuning knobs actually exist on this model (3.1 renamed a
    few — they become silent no-ops)
  - if masks are wired: frame count, object ids seen, per-object coverage,
    all-zero frames, sample shape/dtype
"""

import json

_PREFIX = "[NV_SAM3Diag]"

# The session-management surface nodes/inference_reconstructor.py + sam3_video_nodes.py
# drive the predictor through. Presence here = node-layer compatibility.
_PREDICTOR_API = [
    "start_session", "add_prompt", "add_new_mask", "remove_object",
    "reset_session", "close_session", "propagate_in_video",
    "handle_request", "handle_stream_request",
]
# Config knobs inference_reconstructor._apply_config sets on model.model.
_CONFIG_KNOBS = [
    "score_threshold_detection", "new_det_thresh", "fill_hole_area",
    "assoc_iou_thresh", "det_nms_thresh", "hotstart_unmatch_thresh",
    "hotstart_dup_thresh", "init_trk_keep_alive", "hotstart_delay",
    "decrease_trk_keep_alive_for_empty_masklets",
    "suppress_unmatched_only_within_hotstart",
]


def _safe(fn, default="?"):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — diagnostics must never raise
        return f"<err: {type(e).__name__}: {e}>"


def _summarize_masks(masks):
    """Best-effort one-block summary of the propagate masks output (structure varies)."""
    lines = []
    try:
        if isinstance(masks, dict):
            frame_keys = sorted(k for k in masks.keys() if isinstance(k, int))
            lines.append(f"type=dict | frames={len(masks)} | "
                         f"range=[{frame_keys[0] if frame_keys else '?'}..{frame_keys[-1] if frame_keys else '?'}]")
            # union of obj_ids + per-object coverage + all-zero frame count
            coverage, zero_frames, sample_shape, sample_dtype = {}, 0, None, None
            for fk in frame_keys:
                entry = masks[fk]
                ids = entry.get("obj_ids") if isinstance(entry, dict) else None
                m = entry.get("mask") if isinstance(entry, dict) else entry
                if sample_shape is None and hasattr(m, "shape"):
                    sample_shape, sample_dtype = tuple(m.shape), str(getattr(m, "dtype", "?"))
                if ids is not None:
                    for oid in (ids.tolist() if hasattr(ids, "tolist") else ids):
                        coverage[int(oid)] = coverage.get(int(oid), 0) + 1
                if hasattr(m, "sum"):
                    try:
                        if float(m.sum()) == 0.0:
                            zero_frames += 1
                    except Exception:
                        pass
            ids_sorted = sorted(coverage.keys())
            lines.append(f"object_ids_seen={ids_sorted[:30]}{' ...' if len(ids_sorted) > 30 else ''} "
                         f"(n={len(ids_sorted)})")
            if coverage:
                cov_str = ", ".join(f"obj{o}={coverage[o]}/{len(frame_keys)}" for o in ids_sorted[:12])
                lines.append(f"coverage: {cov_str}{' ...' if len(ids_sorted) > 12 else ''}")
            lines.append(f"all_zero_frames={zero_frames}/{len(frame_keys)}")
            lines.append(f"sample_mask shape={sample_shape} dtype={sample_dtype}")
        elif hasattr(masks, "shape"):
            lines.append(f"type=tensor shape={tuple(masks.shape)} dtype={getattr(masks,'dtype','?')}")
        else:
            lines.append(f"type={type(masks).__name__} (unrecognized - wire the SAM3Propagate 'masks' output)")
    except Exception as e:  # noqa: BLE001
        lines.append(f"<masks summary error: {type(e).__name__}: {e}>")
    return lines


class NV_SAM3Diag:
    """Compact, copy-paste-ready SAM3 / SAM3.1 state report for cross-machine debugging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {"tooltip": "From LoadSAM3Model (3.0) or LoadSAM31Model (3.1)."}),
            },
            "optional": {
                "masks": ("*", {"tooltip": "Optional: SAM3Propagate `masks` output to summarize frames/objects/coverage."}),
                "label": ("STRING", {"default": "", "tooltip": "Free tag for the report header (e.g. 'cycle2-3.1pt')."}),
            },
        }

    RETURN_TYPES = ("STRING", "SAM3_MODEL")
    RETURN_NAMES = ("report", "sam3_model")
    FUNCTION = "diagnose"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True
    DESCRIPTION = "Compact copy-paste SAM3/3.1 diagnostics: model/predictor classes, predictor API surface, session registry, config-knob presence, and (optional) masks summary."

    def diagnose(self, sam3_model, masks=None, label=""):
        m = sam3_model
        pred = _safe(lambda: getattr(m, "_video_predictor", None), None)
        demo = _safe(lambda: m.model, None)

        L = []
        L.append(f"### NV_SAM3Diag{(' - ' + label) if label else ''}")

        # --- model / predictor identity ---
        pred_cls = _safe(lambda: type(pred).__name__) if pred is not None else "None"
        demo_cls = _safe(lambda: type(demo).__name__) if demo is not None else "None"
        is_mux = ("Multiplex" in str(pred_cls)) or ("Multiplex" in str(demo_cls))
        L.append(f"**version:** {'SAM 3.1 (multiplex)' if is_mux else 'SAM 3.0 (legacy)'}")
        L.append(f"**predictor:** {pred_cls}  |  **model:** {demo_cls}")
        L.append(f"**wrapper:** {_safe(lambda: type(m).__name__)}  |  "
                 f"**processor:** {'present' if _safe(lambda: getattr(m,'_processor',None)) is not None else 'None'}")

        # --- predictor API surface (node-layer compat) ---
        if pred is not None:
            present = [a for a in _PREDICTOR_API if _safe(lambda a=a: hasattr(pred, a), False) is True]
            missing = [a for a in _PREDICTOR_API if a not in present]
            L.append(f"**predictor API present:** {' '.join(present) or '(none)'}")
            if missing:
                L.append(f"**predictor API MISSING:** {' '.join(missing)}")
        else:
            L.append("**predictor:** <not accessible - `_video_predictor` missing>")

        # --- session registry (the bring-up crash point) ---
        has_upper = _safe(lambda: hasattr(pred, "_ALL_INFERENCE_STATES"), False) is True
        has_lower = _safe(lambda: hasattr(pred, "_all_inference_states"), False) is True
        reg = None
        if has_upper:
            reg = _safe(lambda: pred._ALL_INFERENCE_STATES)
        elif has_lower:
            reg = _safe(lambda: pred._all_inference_states)
        n_sessions = _safe(lambda: len(reg)) if isinstance(reg, dict) else "?"
        aliased = "yes" if (has_upper and has_lower and _safe(lambda: pred._ALL_INFERENCE_STATES is pred._all_inference_states, False) is True) else "no"
        L.append(f"**session registry:** _ALL_INFERENCE_STATES={has_upper}, _all_inference_states={has_lower}, "
                 f"aliased={aliased}, active_sessions={n_sessions}")

        # --- config knobs present on this model (3.1 renamed some -> silent no-ops) ---
        if demo is not None:
            present_k = [k for k in _CONFIG_KNOBS if _safe(lambda k=k: hasattr(demo, k), False) is True]
            missing_k = [k for k in _CONFIG_KNOBS if k not in present_k]
            L.append(f"**config knobs:** {len(present_k)}/{len(_CONFIG_KNOBS)} present")
            if missing_k:
                L.append(f"**config knobs NO-OP (absent on this model):** {', '.join(missing_k)}")

        # --- optional masks summary ---
        if masks is not None:
            L.append("**masks:**")
            for ln in _summarize_masks(masks):
                L.append(f"  - {ln}")

        report = "\n".join(L)
        banner = f"\n{'='*8} NV_SAM3Diag report (copy below) {'='*8}\n{report}\n{'='*54}\n"
        try:
            print(banner)
        except Exception:  # noqa: BLE001 — console encoding must never break the node
            print(banner.encode("ascii", "replace").decode("ascii"))
        return {"ui": {"text": [report]}, "result": (report, sam3_model)}


NODE_CLASS_MAPPINGS = {"NV_SAM3Diag": NV_SAM3Diag}
NODE_DISPLAY_NAME_MAPPINGS = {"NV_SAM3Diag": "NV SAM3 Diagnostics"}
