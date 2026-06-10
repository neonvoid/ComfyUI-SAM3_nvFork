"""
SAM3 Mask Tracks - ID-consistent multi-object mask export

This module provides nodes for exporting and selecting individual object masks
from SAM3's multi-object tracking output. Replaces the chunking approach with
direct object selection.

Key features:
- Export all tracked objects as separate mask channels [N_frames, N_objects, H, W]
- Per-object metadata (first/last frame, visibility count)
- Fast object selection (tensor slicing, no re-inference)
"""
import json
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def _parse_subject_map(raw, global_obj_ids):
    """Parse a subject_map payload (from NV_SAM3SeedBuilder) into the
    name<->obj_id tables embedded in track_info v2.

    The downstream resolver needs BOTH name->obj_id (declared upstream) AND
    obj_id->channel (only MaskTracks knows). This embeds the former into
    track_info so NV_SAM3SelectMask / NV_SAM3MaskRouter can select by NAME
    without the operator ever inspecting an integer obj_id.

    Accepts, defensively (a malformed map must NOT crash a render):
      - "" / None                                   -> no map
      - {"subjects": [{"obj_id": 1, "name": "head"}, ...]}  (SeedBuilder form)
      - [{"obj_id": 1, "name": "head"}, ...]               (bare subjects list)
      - {"1": "head", "2": "body"}                         (plain obj_id->name)

    `global_obj_ids` is the list of SAM3 obj_ids actually tracked this run;
    used to split declared subjects into present/missing for diagnostics.

    Returns dict {obj_id_to_name, name_to_obj_id, present, missing, warnings}
    or None when no usable map is supplied.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "obj_id_to_name": {}, "name_to_obj_id": {},
                "present": [], "missing": [],
                "warnings": [f"subject_map not valid JSON, ignored: {e}"],
            }

    # Normalize to an iterable of (obj_id, name) candidate pairs.
    pairs = []
    if isinstance(raw, dict) and "subjects" in raw:
        subjects = raw.get("subjects")
        if isinstance(subjects, list):
            for item in subjects:
                if isinstance(item, dict):
                    pairs.append((item.get("obj_id"), item.get("name")))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                pairs.append((item.get("obj_id"), item.get("name")))
    elif isinstance(raw, dict):
        # plain {obj_id: name}
        for k, v in raw.items():
            pairs.append((k, v))
    else:
        return {
            "obj_id_to_name": {}, "name_to_obj_id": {},
            "present": [], "missing": [],
            "warnings": [f"subject_map unrecognized shape {type(raw).__name__}, ignored"],
        }

    obj_id_to_name: Dict[int, str] = {}
    name_to_obj_id: Dict[str, int] = {}
    warnings: List[str] = []
    for rid, rname in pairs:
        # obj_id must be a non-bool integer-coercible value.
        if isinstance(rid, bool) or isinstance(rname, bool):
            warnings.append(f"subject entry with bool field skipped: ({rid!r}, {rname!r})")
            continue
        try:
            oid = int(rid)
        except (TypeError, ValueError):
            warnings.append(f"subject obj_id not integer-coercible, skipped: {rid!r}")
            continue
        name = "" if rname is None else str(rname).strip()
        if not name:
            warnings.append(f"subject obj_id={oid} has blank name, skipped")
            continue
        if oid in obj_id_to_name:
            warnings.append(
                f"duplicate obj_id={oid} in subject_map "
                f"({obj_id_to_name[oid]!r} kept, {name!r} dropped)"
            )
            continue
        if name in name_to_obj_id:
            warnings.append(
                f"duplicate name={name!r} in subject_map "
                f"(obj_id={name_to_obj_id[name]} kept, obj_id={oid} dropped)"
            )
            continue
        obj_id_to_name[oid] = name
        name_to_obj_id[name] = oid

    tracked = set(int(o) for o in global_obj_ids)
    present = [obj_id_to_name[o] for o in obj_id_to_name if o in tracked]
    missing = [obj_id_to_name[o] for o in obj_id_to_name if o not in tracked]
    return {
        "obj_id_to_name": obj_id_to_name,
        "name_to_obj_id": name_to_obj_id,
        "present": present,
        "missing": missing,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Shared name->obj_id->channel resolver (track_info v2)
# Used by NV_SAM3SelectMask (name mode) + NV_SAM3MaskRouter. Single source of
# truth so the two nodes can never drift on resolution semantics.
# ---------------------------------------------------------------------------

def _load_track_info(track_info):
    """Accept track_info as a dict or JSON string; return a dict ({} on failure)."""
    if isinstance(track_info, dict):
        return track_info
    if isinstance(track_info, str) and track_info.strip():
        try:
            obj = json.loads(track_info)
            return obj if isinstance(obj, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _normalize_subject_name(s):
    """casefold + collapse internal whitespace, for the unique-normalized fallback."""
    return " ".join(str(s).strip().casefold().split())


def _subject_to_obj_id_table(info):
    """name->obj_id table from track_info; derive from subject_map if the
    explicit subject_to_obj_id table is absent."""
    table = info.get("subject_to_obj_id")
    if isinstance(table, dict) and table:
        out = {}
        for k, v in table.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out
    sm = info.get("subject_map")  # obj_id -> name
    if isinstance(sm, dict) and sm:
        out = {}
        for oid, name in sm.items():
            try:
                out[str(name)] = int(oid)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def resolve_subject_name(track_info, name):
    """Resolve a subject NAME -> (obj_id, channel) from track_info v2 tables.

    Policy (mask-router R1 design, 20260610-004124):
      1. exact match on the name->obj_id table
      2. else unique casefold+whitespace-normalized match
      3. ambiguous normalized match -> ValueError (use the exact name)
      4. resolved obj_id has no channel (declared upstream but SAM3 produced no
         track) -> ValueError listing the AVAILABLE subjects, so the operator
         fixes a typo / upstream miss WITHOUT ever inspecting an integer id.

    `track_info` may be a dict or JSON string. Returns (obj_id:int, channel:int).
    """
    info = _load_track_info(track_info)
    name_clean = str(name).strip()
    if not name_clean:
        raise ValueError("resolve_subject_name: empty subject name")

    name_to_obj = _subject_to_obj_id_table(info)
    if not name_to_obj:
        raise ValueError(
            "track_info carries no subject_map. Wire NV_SAM3SeedBuilder.subject_map "
            "into NV_SAM3MaskTracks.subject_map so masks can be selected by name."
        )

    obj_id = name_to_obj.get(name_clean)
    if obj_id is None:
        target = _normalize_subject_name(name_clean)
        hits = [(nm, oid) for nm, oid in name_to_obj.items()
                if _normalize_subject_name(nm) == target]
        if len(hits) == 1:
            obj_id = hits[0][1]
        elif len(hits) > 1:
            raise ValueError(
                f"Subject '{name_clean}' is ambiguous under case/space "
                f"normalization: matches {[h[0] for h in hits]}. Use the exact name."
            )
        else:
            available = sorted(name_to_obj.keys())
            raise ValueError(
                f"Subject '{name_clean}' was not found. Available subjects: "
                f"{', '.join(available) if available else '(none)'}."
            )

    obj_id_to_channel = info.get("obj_id_to_channel") or {}
    channel = obj_id_to_channel.get(str(obj_id))
    if channel is None:
        present = info.get("subjects_present")
        avail = (sorted(present) if isinstance(present, list) and present
                 else sorted(nm for nm, oid in name_to_obj.items()
                             if str(oid) in obj_id_to_channel))
        raise ValueError(
            f"Subject '{name_clean}' resolved to obj_id {obj_id}, but that obj_id "
            f"produced no mask track this run (SAM3 found nothing for it). "
            f"Tracked subjects: {', '.join(avail) if avail else '(none)'}."
        )
    return int(obj_id), int(channel)


def _parse_names_list(raw):
    """Split a subject_names value (newline and/or comma separated) into a clean
    ordered list with blanks dropped and duplicates removed (order preserved)."""
    if not raw:
        return []
    seen = set()
    out = []
    for line in str(raw).replace(",", "\n").split("\n"):
        s = line.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


class SAM3MaskTracks:
    """
    Export all object masks as separate, ID-consistent tracks.

    Each tracked object maintains its identity across all frames.
    No chunking or stitching required - just select which objects you want downstream.

    Output tensor shape: [N_frames, N_objects, H, W]
    - Frame 0, Object 0: all_masks[0, 0, :, :]
    - Frame 100, Object 2: all_masks[100, 2, :, :]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Mask dictionary from SAM3Propagate"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state from SAM3 video nodes"
                }),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES", {
                    "tooltip": "Optional scores dictionary from SAM3Propagate"
                }),
                "min_visible_pixels": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 10000,
                    "step": 10,
                    "tooltip": "Minimum mask pixels to count frame as 'visible' for metadata"
                }),
                "output_bboxes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output filled bounding box masks instead of silhouettes. Also adds bbox coordinates to track_info JSON."
                }),
                "bbox_padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Padding (in pixels) to add around bounding boxes"
                }),
                "subject_map": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional subject_map JSON (wire from NV_SAM3SeedBuilder, or "
                        'paste for testing), e.g. {"subjects":[{"obj_id":1,"name":"head"},'
                        '{"obj_id":2,"name":"body"}]}. Embeds name<->obj_id tables into '
                        "track_info (v2) so downstream NV_SAM3SelectMask / NV_SAM3MaskRouter "
                        "can select masks by NAME instead of integer obj_id. Leave blank "
                        "for legacy integer use."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING", "INT")
    RETURN_NAMES = ("all_masks", "bbox_masks", "track_info", "num_objects")
    FUNCTION = "extract_tracks"
    CATEGORY = "SAM3/video"

    def extract_tracks(self, masks, video_state, scores=None, min_visible_pixels=100, output_bboxes=False, bbox_padding=0, subject_map=None):
        """
        Extract all object masks into a single [N_frames, N_objects, H, W] tensor.

        Also computes per-object visibility metadata.

        If output_bboxes=True, also computes:
        - bbox_masks: filled rectangular masks from bounding boxes
        - bbox coordinates in track_info JSON
        """
        num_frames = video_state.num_frames
        h, w = video_state.height, video_state.width

        print(f"[SAM3 MaskTracks] Extracting tracks from {num_frames} frames")
        if output_bboxes:
            print(f"[SAM3 MaskTracks] Bounding box output enabled (padding={bbox_padding}px)")

        # Helper to extract mask tensor + obj_ids from mask data.
        # Returns (tensor, obj_ids_or_None). When obj_ids is None, caller must
        # fall back to legacy channel-index semantics (raw-tensor or empty-list
        # cases). When obj_ids is a list, caller MUST use it to map local
        # channels to stable SAM3 object identities — assuming channel index ==
        # obj_id is the 2026-05-20 audit bug that silently swaps identities on
        # any clip where SAM3 returns sparse or reordered ids.
        def get_mask_data(mask_data):
            if isinstance(mask_data, dict):
                # Canonical format from SAM3Propagate:
                # {"mask": tensor[local_N, H, W], "obj_ids": [sam3_id_per_channel]}
                tensor = mask_data.get("mask")
                ids = mask_data.get("obj_ids")
                # Empty/None ids list → fall through to legacy channel-index mode.
                if not ids:
                    return tensor, None
                return tensor, list(ids)
            # Raw-tensor legacy mode — no per-frame id metadata available.
            return mask_data, None

        # First pass: discover the global set of SAM3 object IDs ever seen.
        # When dict-with-obj_ids is provided, this is the union of all per-frame
        # obj_ids sorted ascending. When falling back to raw-tensor mode (or any
        # frame's dict has empty obj_ids), we use channel-index as a synthetic
        # obj_id so existing workflows that wire raw tensors keep working.
        global_obj_id_set = set()
        max_legacy_channels = 0
        any_real_ids = False
        for frame_idx in range(num_frames):
            if frame_idx not in masks:
                continue
            frame_mask, frame_obj_ids = get_mask_data(masks[frame_idx])
            if frame_mask is None:
                continue
            if isinstance(frame_mask, np.ndarray):
                tensor_for_shape = torch.from_numpy(frame_mask)
            else:
                tensor_for_shape = frame_mask
            # Determine local channel count for legacy path.
            if tensor_for_shape.dim() == 4:
                local_n = tensor_for_shape.shape[1]
            elif tensor_for_shape.dim() == 3:
                local_n = tensor_for_shape.shape[0]
            elif tensor_for_shape.dim() == 2:
                local_n = 1
            else:
                continue
            if frame_obj_ids is not None:
                any_real_ids = True
                # Take only as many ids as channels available — guard against
                # producer/tensor length mismatch.
                for sam3_id in frame_obj_ids[:local_n]:
                    global_obj_id_set.add(int(sam3_id))
            else:
                max_legacy_channels = max(max_legacy_channels, local_n)

        # Resolve global ordering.
        # SAM3-real-ids path: sorted union of all seen SAM3 obj_ids. Stable
        # across frames. Channel position = position in this sorted list.
        # Legacy path: 0..N-1 channel indices (current behavior preserved when
        # no obj_ids metadata is supplied anywhere).
        if any_real_ids:
            global_obj_ids = sorted(global_obj_id_set)
        else:
            global_obj_ids = list(range(max_legacy_channels))
        obj_id_to_channel = {oid: idx for idx, oid in enumerate(global_obj_ids)}
        num_objects = len(global_obj_ids)

        # --- track_info v2: name<->obj_id + obj_id<->channel tables ---
        # Both halves of the name-addressed pipeline (NV_SAM3SelectMask name
        # mode + NV_SAM3MaskRouter) resolve name -> obj_id -> channel from
        # track_info alone. obj_id<->channel is known only here; name<->obj_id
        # comes from the optional subject_map (NV_SAM3SeedBuilder). All keys are
        # additive — existing consumers of `objects`/`obj_id_mapping` unaffected.
        v2_block = {
            "version": "nv_sam3_mask_tracks.v2",
            "obj_id_to_channel": {str(global_obj_ids[ch]): ch for ch in range(num_objects)},
            "channel_to_obj_id": {str(ch): global_obj_ids[ch] for ch in range(num_objects)},
        }
        _sm = _parse_subject_map(subject_map, global_obj_ids)
        v2_block["subject_map_source"] = "input" if _sm is not None else "none"
        if _sm is not None:
            v2_block["subject_map"] = {str(k): v for k, v in _sm["obj_id_to_name"].items()}
            v2_block["subject_to_obj_id"] = dict(_sm["name_to_obj_id"])
            v2_block["subjects_present"] = _sm["present"]
            v2_block["subjects_missing"] = _sm["missing"]
            if _sm["warnings"]:
                v2_block["subject_map_warnings"] = _sm["warnings"]
                for _w in _sm["warnings"]:
                    print(f"[SAM3 MaskTracks] WARN subject_map: {_w}")
            if _sm["missing"]:
                print(
                    f"[SAM3 MaskTracks] WARN: declared subjects with no tracked "
                    f"obj_id this run: {_sm['missing']} -- selecting them by name "
                    f"downstream will raise."
                )

        if num_objects == 0:
            print("[SAM3 MaskTracks] No objects found in masks")
            empty_masks = torch.zeros(num_frames, 1, h, w)
            track_info = json.dumps({
                "objects": [],
                "total_frames": num_frames,
                "total_objects": 0,
                **v2_block,
            }, indent=2)
            return (empty_masks, empty_masks, track_info, 0)

        if any_real_ids:
            print(
                f"[SAM3 MaskTracks] Found {num_objects} objects "
                f"(SAM3 obj_ids: {global_obj_ids})"
            )
        else:
            print(
                f"[SAM3 MaskTracks] Found {num_objects} objects "
                f"(legacy channel-index mode — no obj_ids metadata in input)"
            )

        # Build [N_frames, N_objects, H, W] tensor — second dim indexed by
        # STABLE global channel (SAM3 obj_id position when available, else
        # legacy channel index).
        all_masks = torch.zeros(num_frames, num_objects, h, w)

        # Track per-object visibility info keyed by global channel.
        object_info = {
            ch: {
                "sam3_obj_id": global_obj_ids[ch],
                "first_frame": None,
                "last_frame": None,
                "visible_frames": 0,
                "total_pixels": 0,
                "scores": [],
                "bboxes": {}  # frame_idx -> [x1, y1, x2, y2]
            }
            for ch in range(num_objects)
        }

        # Also build bbox_masks if requested
        bbox_masks = torch.zeros(num_frames, num_objects, h, w) if output_bboxes else None

        # Counter for legacy/raw frames encountered while in real-id mode —
        # those frames cannot be safely positional-mapped into real-id channels
        # (Codex R2 finding 2026-05-20). Surfaces in track_info as a diagnostic.
        inconsistent_metadata_frames = 0

        for frame_idx in range(num_frames):
            if frame_idx in masks:
                frame_mask, frame_obj_ids = get_mask_data(masks[frame_idx])
                if frame_mask is None:
                    continue

                # Convert numpy to torch if needed
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)

                # Handle different mask shapes.
                # 4D guard (Codex R2 finding): pre-fix code did frame_mask.squeeze(0)
                # blindly; for B>1 the result is still 4D and neither the 3D nor
                # 2D branch handles it. Raise loud — the SAM3 propagate contract
                # is single-batch per frame; B>1 is a wiring bug, not a valid case.
                if frame_mask.dim() == 4:
                    if frame_mask.shape[0] != 1:
                        raise ValueError(
                            f"[SAM3 MaskTracks] Frame {frame_idx} mask has 4D "
                            f"shape {tuple(frame_mask.shape)} with batch>1. "
                            f"SAM3 propagate is contracted to emit one batch "
                            f"per frame. This indicates a wiring bug upstream."
                        )
                    frame_mask = frame_mask.squeeze(0)  # Remove single batch dim

                # Resolve per-frame local-channel → global-channel mapping.
                # In SAM3-real-ids mode, each local channel's obj_id tells us
                # which stable global slot to write to. In legacy mode, local
                # channel position == global channel position.
                #
                # Codex R2 finding 2026-05-20: when any_real_ids=True (we
                # committed to real-id channel ordering globally) but THIS
                # frame has no obj_ids metadata (raw tensor or empty list),
                # the pre-R2 code silently fell back to positional mapping —
                # which placed an UNKNOWN identity into the sorted real-id
                # channel position. That's the exact identity-corruption class
                # the patch was meant to eliminate. R2 fix: skip the frame
                # with a warning + counter. Empty masks for one frame are
                # always safer than wrong identity.
                if frame_obj_ids is not None:
                    local_to_global = []
                    for local_idx, sam3_id in enumerate(frame_obj_ids):
                        sam3_id = int(sam3_id)
                        global_ch = obj_id_to_channel.get(sam3_id)
                        if global_ch is None:
                            # Should not happen — first pass collected all ids.
                            continue
                        local_to_global.append((local_idx, global_ch))
                elif any_real_ids:
                    # Mixed-mode frame in real-id mode — refuse to guess identity.
                    inconsistent_metadata_frames += 1
                    print(
                        f"[SAM3 MaskTracks] WARN: Frame {frame_idx} has no "
                        f"obj_ids metadata but other frames do — skipping "
                        f"this frame to avoid silent identity corruption. "
                        f"Producer should emit consistent metadata across all "
                        f"frames. (inconsistent so far: "
                        f"{inconsistent_metadata_frames})"
                    )
                    continue
                else:
                    # Pure legacy mode (no frame has real ids) — positional map.
                    local_to_global = [(i, i) for i in range(num_objects)]

                if frame_mask.dim() == 3:
                    # Multi-object mask [N_local, H, W]
                    n_local = frame_mask.shape[0]
                    for local_idx, global_ch in local_to_global:
                        if local_idx >= n_local:
                            continue
                        obj_mask = frame_mask[local_idx].float()

                        # Normalize to [0, 1] if needed
                        if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                            obj_mask = obj_mask / 255.0

                        # Resize if dimensions don't match
                        if obj_mask.shape[0] != h or obj_mask.shape[1] != w:
                            obj_mask = torch.nn.functional.interpolate(
                                obj_mask.unsqueeze(0).unsqueeze(0),
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)

                        all_masks[frame_idx, global_ch] = obj_mask

                        # Track visibility (keyed by stable global channel)
                        pixel_count = (obj_mask > 0.5).sum().item()
                        if pixel_count >= min_visible_pixels:
                            if object_info[global_ch]["first_frame"] is None:
                                object_info[global_ch]["first_frame"] = frame_idx
                            object_info[global_ch]["last_frame"] = frame_idx
                            object_info[global_ch]["visible_frames"] += 1
                            object_info[global_ch]["total_pixels"] += pixel_count

                            # Compute bounding box if enabled
                            if output_bboxes:
                                ys, xs = torch.where(obj_mask > 0.5)
                                if len(xs) > 0:
                                    x1 = max(0, xs.min().item() - bbox_padding)
                                    y1 = max(0, ys.min().item() - bbox_padding)
                                    x2 = min(w, xs.max().item() + 1 + bbox_padding)
                                    y2 = min(h, ys.max().item() + 1 + bbox_padding)
                                    object_info[global_ch]["bboxes"][frame_idx] = [x1, y1, x2, y2]
                                    # Fill bbox mask
                                    bbox_masks[frame_idx, global_ch, y1:y2, x1:x2] = 1.0

                elif frame_mask.dim() == 2:
                    # Single mask [H, W]. Target channel resolved below from
                    # local_to_global[0] when obj_ids carried in input;
                    # otherwise channel 0 (legacy mode).
                    obj_mask = frame_mask.float()
                    if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                        obj_mask = obj_mask / 255.0

                    if obj_mask.shape[0] != h or obj_mask.shape[1] != w:
                        obj_mask = torch.nn.functional.interpolate(
                            obj_mask.unsqueeze(0).unsqueeze(0),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)

                    # Resolve target channel — if frame_obj_ids has exactly one
                    # id, use its global mapping; otherwise channel 0.
                    if local_to_global:
                        target_ch = local_to_global[0][1]
                    else:
                        target_ch = 0
                    all_masks[frame_idx, target_ch] = obj_mask

                    pixel_count = (obj_mask > 0.5).sum().item()
                    if pixel_count >= min_visible_pixels:
                        if object_info[target_ch]["first_frame"] is None:
                            object_info[target_ch]["first_frame"] = frame_idx
                        object_info[target_ch]["last_frame"] = frame_idx
                        object_info[target_ch]["visible_frames"] += 1
                        object_info[target_ch]["total_pixels"] += pixel_count

                        # Compute bounding box if enabled
                        if output_bboxes:
                            ys, xs = torch.where(obj_mask > 0.5)
                            if len(xs) > 0:
                                x1 = max(0, xs.min().item() - bbox_padding)
                                y1 = max(0, ys.min().item() - bbox_padding)
                                x2 = min(w, xs.max().item() + 1 + bbox_padding)
                                y2 = min(h, ys.max().item() + 1 + bbox_padding)
                                object_info[target_ch]["bboxes"][frame_idx] = [x1, y1, x2, y2]
                                # Fill bbox mask
                                bbox_masks[frame_idx, target_ch, y1:y2, x1:x2] = 1.0

            # Collect scores if available — map score index using same
            # per-frame obj_ids when present, else legacy positional alignment.
            if scores is not None and frame_idx in scores:
                frame_scores = scores[frame_idx]
                if hasattr(frame_scores, 'tolist'):
                    score_list = frame_scores.tolist()
                    if score_list and isinstance(score_list[0], list):
                        score_list = score_list[0]
                    # Re-fetch obj_ids for score alignment (mask path above may
                    # not have run if frame_idx not in masks but is in scores).
                    _, score_obj_ids = (
                        get_mask_data(masks[frame_idx])
                        if frame_idx in masks else (None, None)
                    )
                    if score_obj_ids is not None:
                        for local_idx, score in enumerate(score_list):
                            if local_idx >= len(score_obj_ids):
                                break
                            sam3_id = int(score_obj_ids[local_idx])
                            global_ch = obj_id_to_channel.get(sam3_id)
                            if global_ch is not None:
                                object_info[global_ch]["scores"].append(score)
                    elif not any_real_ids:
                        # Pure legacy mode — positional alignment is safe.
                        for ch, score in enumerate(score_list):
                            if ch < num_objects:
                                object_info[ch]["scores"].append(score)
                    # else: real-id mode but this frame's scores have no obj_ids
                    # metadata — skip rather than positional-map. Mirrors the
                    # mixed-mode mask safety contract. Drops the scores but does
                    # not corrupt avg_score by mis-attributing scores to wrong
                    # channels. (R3 follow-up flagged by both Codex + Gemini.)

        # Compute average scores
        for ch in range(num_objects):
            if object_info[ch]["scores"]:
                object_info[ch]["avg_score"] = sum(object_info[ch]["scores"]) / len(object_info[ch]["scores"])
            else:
                object_info[ch]["avg_score"] = None

        # Build track_info JSON.
        # `id` stays as channel index for back-compat with existing
        # NV_SAM3SelectMask object_ids string ("0,1,2"). NEW: `sam3_obj_id`
        # exposes the real SAM3 object id so callers can introspect the
        # channel-to-identity mapping (and a future SelectMask widget can
        # accept obj_id semantics without breaking saved workflows).
        objects_list = []
        for ch, info in object_info.items():
            if info["first_frame"] is not None:
                obj_entry = {
                    "id": ch,
                    "sam3_obj_id": info["sam3_obj_id"],
                    "first_frame": info["first_frame"],
                    "last_frame": info["last_frame"],
                    "visible_frames": info["visible_frames"],
                    "avg_score": round(info["avg_score"], 4) if info["avg_score"] is not None else None,
                }
                # Include bbox data if enabled
                if output_bboxes and info["bboxes"]:
                    obj_entry["bboxes"] = info["bboxes"]
                objects_list.append(obj_entry)

        track_info = {
            "objects": objects_list,
            "total_frames": num_frames,
            "total_objects": num_objects,
            "dimensions": {"height": h, "width": w},
            "bbox_output_enabled": output_bboxes,
            # NEW: explicit channel-to-obj_id table — helps debug downstream
            # SelectMask usage and confirms the audit fix is in effect.
            "obj_id_mapping": {
                str(ch): global_obj_ids[ch] for ch in range(num_objects)
            },
            "id_source": "sam3_obj_ids" if any_real_ids else "legacy_channel_index",
            # R2 diagnostic: count of frames skipped due to missing obj_ids
            # metadata while running in real-id mode. Non-zero means the
            # producer is emitting inconsistent metadata across frames.
            "inconsistent_metadata_frames": inconsistent_metadata_frames,
            # track_info v2: obj_id<->channel + (optional) name<->obj_id tables.
            **v2_block,
        }

        # Print summary — surface the obj_id mapping so users can confirm.
        for obj in track_info["objects"]:
            score_str = f", avg_score={obj['avg_score']:.3f}" if obj['avg_score'] else ""
            bbox_str = f", {len(obj.get('bboxes', {}))} bboxes" if output_bboxes else ""
            print(
                f"[SAM3 MaskTracks]   Channel {obj['id']} (SAM3 obj_id={obj['sam3_obj_id']}): "
                f"frames {obj['first_frame']}-{obj['last_frame']} "
                f"({obj['visible_frames']} visible){score_str}{bbox_str}"
            )

        # If bbox output not enabled, return all_masks as bbox_masks (identity)
        if bbox_masks is None:
            bbox_masks = all_masks

        return (all_masks, bbox_masks, json.dumps(track_info, indent=2), num_objects)


class SAM3SelectMask:
    """
    Select specific object(s) from multi-track mask output.

    Fast operation - no re-inference needed, just tensor slicing.

    Examples:
    - object_ids="0" → Just object 0's mask [N_frames, H, W]
    - object_ids="0,2" + combine_mode="union" → Combined mask of objects 0 and 2
    - object_ids="all" + combine_mode="separate" → All masks [N_frames, N_obj, H, W]

    Optionally provide video_frames to get a colored visualization overlay.
    """

    # Color palette for visualization (RGB, 0-1 range)
    COLORS = [
        [0.0, 0.5, 1.0],   # Blue
        [1.0, 0.3, 0.3],   # Red
        [0.3, 1.0, 0.3],   # Green
        [1.0, 1.0, 0.0],   # Yellow
        [1.0, 0.0, 1.0],   # Magenta
        [0.0, 1.0, 1.0],   # Cyan
        [1.0, 0.5, 0.0],   # Orange
        [0.5, 0.0, 1.0],   # Purple
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "all_masks": ("MASK", {
                    "tooltip": "Multi-object mask tensor [N_frames, N_objects, H, W] from SAM3MaskTracks"
                }),
                "object_ids": ("STRING", {
                    "default": "all",
                    "tooltip": "Object IDs to select: 'all', single '0', or comma-separated '0,2,3'"
                }),
            },
            "optional": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Optional video frames for visualization overlay [N, H, W, C]"
                }),
                "combine_mode": (["union", "separate", "first"], {
                    "default": "union",
                    "tooltip": "How to combine multiple objects: union (OR), separate (keep channels), first (just first selected)"
                }),
                "viz_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask overlay transparency for visualization (0=invisible, 1=opaque)"
                }),
                "mask_colors": ("STRING", {
                    "default": "",
                    "tooltip": "Custom mask colors (comma-separated). Names: red, blue, green, yellow, magenta, cyan, orange, purple, pink, lime, teal, coral, gold, navy. Or hex: #FF0000. Order matches object IDs. Empty = default colors."
                }),
                # --- name-addressed selection (appended for back-compat; never
                # reorder the widgets above — widgets_values is positional) ---
                "select_by": (["id", "name"], {
                    "default": "id",
                    "tooltip": (
                        "id = select by integer channel (legacy, unchanged). "
                        "name = select by subject name from upstream "
                        "NV_SAM3SeedBuilder; requires track_info wired. In name "
                        "mode, object_ids is ignored."
                    ),
                }),
                "subject_names": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Subject name(s) to select, one per line or comma-"
                        "separated (e.g. 'head' or 'head, hair'). Only used when "
                        "select_by=name. Names come from the subjects list "
                        "declared far upstream -- no integer ids to look up."
                    ),
                }),
                "missing_name_policy": (["error", "empty_warn"], {
                    "default": "error",
                    "tooltip": (
                        "name mode: error = raise if a requested name has no "
                        "track (lists available subjects). empty_warn = skip it "
                        "with a warning (empty mask if none resolve)."
                    ),
                }),
                "track_info": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "track_info JSON from NV_SAM3MaskTracks. Required for "
                        "select_by=name (carries the name<->obj_id<->channel tables)."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("selected_masks", "selected_ids", "visualization")
    FUNCTION = "select"
    CATEGORY = "SAM3/video"

    def _create_visualization(self, frames, masks, valid_ids, alpha=0.5, mask_colors=""):
        """Create colored mask overlay on video frames."""
        from .utils import get_color_palette

        num_frames = frames.shape[0]
        h, w = frames.shape[1], frames.shape[2]
        vis_list = []

        # Get color palette based on number of objects
        num_objects = masks.shape[1] if masks.dim() == 4 else 1
        colors = get_color_palette(mask_colors, max(num_objects, len(valid_ids)))

        for frame_idx in range(num_frames):
            vis_frame = frames[frame_idx].clone()

            # Handle different mask shapes
            if masks.dim() == 3:
                # Single combined mask [N_frames, H, W]
                frame_mask = masks[frame_idx].float()
                if frame_mask.max() > 1.0:
                    frame_mask = frame_mask / 255.0

                # Use first color for combined mask
                color = torch.tensor(colors[0])
                mask_rgb = frame_mask.unsqueeze(-1) * color.view(1, 1, 3)
                vis_frame = vis_frame * (1 - alpha * frame_mask.unsqueeze(-1)) + alpha * mask_rgb

            elif masks.dim() == 4:
                # Multi-object masks [N_frames, N_objects, H, W]
                for idx, oid in enumerate(valid_ids):
                    if oid < masks.shape[1]:
                        obj_mask = masks[frame_idx, oid].float()
                        if obj_mask.max() > 1.0:
                            obj_mask = obj_mask / 255.0

                        color = torch.tensor(colors[idx % len(colors)])
                        mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - alpha * obj_mask.unsqueeze(-1)) + alpha * mask_rgb

            vis_list.append(vis_frame.clamp(0, 1))

        return torch.stack(vis_list, dim=0)

    def select(self, all_masks, object_ids="all", video_frames=None, combine_mode="union", viz_alpha=0.5, mask_colors="",
               select_by="id", subject_names="", missing_name_policy="error", track_info=""):
        """
        Select and optionally combine object masks.

        select_by="id" (default): legacy integer-channel selection via object_ids.
        select_by="name": resolve subject_names -> obj_id -> channel from
        track_info (v2). object_ids is ignored in name mode.
        """
        # Handle different input shapes
        if all_masks.dim() == 3:
            # Already [N_frames, H, W] - single object, just return
            print(f"[SAM3 SelectMask] Input is single-object mask, returning as-is")
            num_frames, h, w = all_masks.shape
            # Create visualization if frames provided
            if video_frames is not None:
                visualization = self._create_visualization(video_frames, all_masks, [0], viz_alpha, mask_colors)
            else:
                visualization = torch.zeros(num_frames, h, w, 3)
            return (all_masks, "0", visualization)

        if all_masks.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N_frames, N_objects, H, W], got shape {all_masks.shape}")

        num_frames, num_objects, h, w = all_masks.shape
        print(f"[SAM3 SelectMask] Input: {num_frames} frames, {num_objects} objects")

        if select_by == "name":
            # Resolve subject names -> channels via track_info v2 tables.
            # object_ids is intentionally ignored here.
            names = _parse_names_list(subject_names)
            if not names:
                raise ValueError(
                    "[SAM3 SelectMask] select_by=name but subject_names is empty. "
                    "Enter one subject name per line (e.g. 'head')."
                )
            info = _load_track_info(track_info)
            if not info:
                raise ValueError(
                    "[SAM3 SelectMask] select_by=name requires track_info from "
                    "NV_SAM3MaskTracks. Wire its track_info output into this node."
                )
            valid_ids = []
            for nm in names:
                try:
                    _oid, ch = resolve_subject_name(info, nm)
                except ValueError as e:
                    if missing_name_policy == "empty_warn":
                        print(f"[SAM3 SelectMask] WARN (empty_warn): {e}")
                        continue
                    raise
                # Stale-mismatch guard: track_info channel must index this tensor.
                if ch >= num_objects:
                    raise ValueError(
                        f"[SAM3 SelectMask] Subject '{nm}' maps to channel {ch} but "
                        f"all_masks has only {num_objects} channels. track_info and "
                        f"all_masks are from different runs -- re-wire both from the "
                        f"SAME NV_SAM3MaskTracks."
                    )
                if ch not in valid_ids:
                    valid_ids.append(ch)
            print(f"[SAM3 SelectMask] name mode: {names} -> channels {valid_ids}")
        else:
            # Parse object_ids (legacy integer-channel path, unchanged)
            object_ids_str = object_ids.strip().lower()
            if object_ids_str == "all":
                ids = list(range(num_objects))
            else:
                try:
                    ids = [int(x.strip()) for x in object_ids.split(",") if x.strip()]
                except ValueError:
                    print(f"[SAM3 SelectMask] Warning: Could not parse '{object_ids}', using all objects")
                    ids = list(range(num_objects))

            # Filter to valid IDs
            valid_ids = [i for i in ids if 0 <= i < num_objects]
            if not valid_ids:
                print(f"[SAM3 SelectMask] Warning: No valid object IDs found, using object 0")
                valid_ids = [0] if num_objects > 0 else []

        print(f"[SAM3 SelectMask] Selecting objects: {valid_ids}, mode: {combine_mode}")

        if not valid_ids:
            # No objects - return empty mask and visualization
            empty_mask = torch.zeros(num_frames, h, w)
            empty_vis = torch.zeros(num_frames, h, w, 3)
            if video_frames is not None:
                empty_vis = video_frames.clone()
            return (empty_mask, "", empty_vis)

        # Select and combine based on mode
        if combine_mode == "union":
            # Combine all selected objects with max (OR operation)
            selected = torch.zeros(num_frames, h, w, device=all_masks.device)
            for oid in valid_ids:
                selected = torch.max(selected, all_masks[:, oid])
            ids_str = ",".join(str(i) for i in valid_ids)

        elif combine_mode == "separate":
            # Keep as separate channels
            selected = all_masks[:, valid_ids]
            # If only one object, squeeze to [N_frames, H, W]
            if len(valid_ids) == 1:
                selected = selected.squeeze(1)
            ids_str = ",".join(str(i) for i in valid_ids)

        elif combine_mode == "first":
            # Just return the first selected object
            selected = all_masks[:, valid_ids[0]]
            ids_str = str(valid_ids[0])

        else:
            # Default to union
            selected = torch.zeros(num_frames, h, w, device=all_masks.device)
            for oid in valid_ids:
                selected = torch.max(selected, all_masks[:, oid])
            ids_str = ",".join(str(i) for i in valid_ids)

        # Create visualization if frames provided
        if video_frames is not None:
            # For visualization, use the selected masks (may be multi-channel if separate)
            if combine_mode == "separate" and selected.dim() == 4:
                visualization = self._create_visualization(video_frames, selected, list(range(len(valid_ids))), viz_alpha, mask_colors)
            else:
                visualization = self._create_visualization(video_frames, selected, valid_ids, viz_alpha, mask_colors)
        else:
            # Return empty visualization with correct shape
            if selected.dim() == 3:
                visualization = torch.zeros(num_frames, h, w, 3)
            else:
                visualization = torch.zeros(num_frames, h, w, 3)

        return (selected, ids_str, visualization)


class SAM3BatchPlanner:
    """
    Plan batch processing for multi-object tracking output.

    Intelligently groups objects into batches for downstream workflows with
    object count limits (e.g., 4-actor limit). Includes noise filtering for
    robust handling of noisy footage.

    Features:
    - Filter out short-lived/noisy tracks by minimum visible frames
    - Filter flickering objects by visibility ratio
    - Sort by stability (visible_frames, avg_score, or first_frame)
    - Output per-batch frame ranges for optional trimming
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_info": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON track info from SAM3MaskTracks"
                }),
            },
            "optional": {
                "batch_mode": (["by_stability", "temporal", "overlap_optimized"], {
                    "default": "by_stability",
                    "tooltip": "by_stability: group by visibility/score. temporal: group by co-occurrence. overlap_optimized: group by maximum overlap efficiency (players visible together for longest time)"
                }),
                "max_objects_per_batch": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum number of objects per batch"
                }),
                "min_visible_frames": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Filter out tracks with fewer visible frames (noise filter)"
                }),
                "min_visibility_ratio": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum ratio of visible_frames / total_span (filter flickering)"
                }),
                "sort_by": (["visible_frames", "avg_score", "first_frame"], {
                    "default": "visible_frames",
                    "tooltip": "Sort priority (by_stability mode): most visible first, highest score first, or temporal order"
                }),
                "min_overlap_frames": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 300,
                    "step": 5,
                    "tooltip": "Minimum frames of overlap required to batch players together (overlap_optimized mode). ~0.5s at 30fps."
                }),
                "batch_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Which batch to output (0-indexed). Use with loops."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING", "INT")
    RETURN_NAMES = ("batch_object_ids", "batch_start_frame", "batch_end_frame", "num_batches", "batch_schedule", "filtered_count")
    FUNCTION = "plan_batches"
    CATEGORY = "SAM3/video"

    def _temporal_batching(self, objects, max_per_batch, total_frames):
        """
        Group objects by temporal co-occurrence.

        Creates batches where objects in each batch are visible together
        in the same time window. Each batch has specific frame ranges.
        """
        if not objects or total_frames <= 0:
            return []

        # Build object lookup by ID
        obj_by_id = {obj["id"]: obj for obj in objects}

        # Build frame-to-objects mapping
        frame_objects = defaultdict(set)
        for obj in objects:
            first_frame = obj["first_frame"]
            last_frame = obj["last_frame"]
            for f in range(first_frame, last_frame + 1):
                frame_objects[f].add(obj["id"])

        # Find transition points (where active object set changes)
        transitions = [0]
        prev_set = frame_objects.get(0, set())
        for f in range(1, total_frames):
            curr_set = frame_objects.get(f, set())
            if curr_set != prev_set:
                transitions.append(f)
                prev_set = curr_set
        transitions.append(total_frames)

        # Create batches for each time window
        batches = []
        for i in range(len(transitions) - 1):
            start_frame = transitions[i]
            end_frame = transitions[i + 1] - 1

            active_ids = list(frame_objects.get(start_frame, set()))
            if not active_ids:
                continue

            # Sort active objects by their first_frame (entrance order)
            active_ids.sort(key=lambda oid: obj_by_id[oid]["first_frame"])

            # If more than max_per_batch objects, split into multiple batches
            for j in range(0, len(active_ids), max_per_batch):
                batch_ids = active_ids[j:j + max_per_batch]
                batches.append({
                    "batch_index": len(batches),
                    "object_ids": batch_ids,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "object_count": len(batch_ids)
                })

        return batches

    def _stability_batching(self, objects, max_per_batch, sort_by):
        """
        Group objects by stability (visibility/score), processing all their frames.

        This is the original batching mode - groups objects regardless of
        when they appear, sorted by stability metrics.
        """
        # Sort by stability
        if sort_by == "visible_frames":
            objects.sort(key=lambda x: -x.get("visible_frames", 0))
        elif sort_by == "avg_score":
            objects.sort(key=lambda x: -(x.get("avg_score") or 0))
        elif sort_by == "first_frame":
            objects.sort(key=lambda x: x.get("first_frame", 0))

        # Group into batches
        batches = []
        for i in range(0, len(objects), max_per_batch):
            batch_objs = objects[i:i + max_per_batch]
            batch_ids = [obj["id"] for obj in batch_objs]
            batch_start = min(obj["first_frame"] for obj in batch_objs)
            batch_end = max(obj["last_frame"] for obj in batch_objs)

            batches.append({
                "batch_index": len(batches),
                "object_ids": batch_ids,
                "start_frame": batch_start,
                "end_frame": batch_end,
                "object_count": len(batch_ids)
            })

        return batches

    def _overlap_optimized_batching(self, objects, max_per_batch, min_overlap_frames):
        """
        Group objects by maximum overlap efficiency.

        Creates batches where all objects in each batch are visible together
        for the longest possible time. Each batch's frame range is the tight
        overlap window where ALL batch members are present.

        This is optimal for workflows where:
        - Different objects have very different screen times
        - You want to minimize wasted processing (only process frames where all batch objects exist)
        - Objects with long durations should be isolated rather than "wasting" their frames
        """
        from itertools import combinations

        if not objects:
            return []

        # Get player presence windows: (id, first_frame, last_frame)
        players = [(obj["id"], obj["first_frame"], obj["last_frame"]) for obj in objects]

        # Print duration analysis
        print(f"[SAM3 BatchPlanner] Player duration analysis:")
        for p in sorted(players, key=lambda x: x[2]):  # Sort by exit time
            duration = p[2] - p[1]
            print(f"[SAM3 BatchPlanner]   Player {p[0]}: frames {p[1]}-{p[2]} (duration: {duration} frames)")

        def get_overlap(player_subset):
            """Calculate overlap window for a set of players."""
            if not player_subset:
                return None, 0
            start = max(p[1] for p in player_subset)  # Latest entry
            end = min(p[2] for p in player_subset)    # Earliest exit
            if end <= start:
                return None, 0
            return (start, end), end - start

        # Greedy batch assignment
        unassigned = set(range(len(players)))
        batches = []

        while unassigned:
            best_batch = None
            best_score = 0
            best_window = None

            # Try all combinations up to max_per_batch
            # Start with larger groups (prefer grouping when possible)
            for size in range(min(len(unassigned), max_per_batch), 0, -1):
                for combo in combinations(unassigned, size):
                    subset = [players[i] for i in combo]
                    window, overlap = get_overlap(subset)

                    if overlap >= min_overlap_frames:
                        # Score: overlap duration * small bonus for larger groups
                        # This prefers grouping players together when overlap is similar
                        score = overlap * (1 + 0.1 * len(combo))
                        if score > best_score:
                            best_score = score
                            best_batch = combo
                            best_window = window

            if best_batch:
                # Create batch with calculated frame range
                batch_ids = [players[i][0] for i in best_batch]
                batches.append({
                    "batch_index": len(batches),
                    "object_ids": batch_ids,
                    "start_frame": best_window[0],
                    "end_frame": best_window[1],
                    "object_count": len(batch_ids),
                    "overlap_frames": best_window[1] - best_window[0]
                })
                unassigned -= set(best_batch)
            else:
                # No valid overlap found - assign remaining players individually
                print(f"[SAM3 BatchPlanner] No valid overlaps found for remaining {len(unassigned)} players, creating individual batches")
                for i in sorted(unassigned):
                    p = players[i]
                    batches.append({
                        "batch_index": len(batches),
                        "object_ids": [p[0]],
                        "start_frame": p[1],
                        "end_frame": p[2],
                        "object_count": 1,
                        "overlap_frames": p[2] - p[1]
                    })
                break

        return batches

    def plan_batches(
        self,
        track_info,
        batch_mode="by_stability",
        max_objects_per_batch=4,
        min_visible_frames=30,
        min_visibility_ratio=0.1,
        sort_by="visible_frames",
        min_overlap_frames=15,
        batch_index=0
    ):
        """
        Parse track_info, filter noise, and generate batch schedule.

        Three modes available:
        - by_stability: Group by visibility/score, process all frames for each batch
        - temporal: Group by co-occurrence, each batch processes only its time window
        - overlap_optimized: Group by maximum overlap efficiency (tight frame ranges where all batch members present)
        """
        # Parse JSON
        try:
            info = json.loads(track_info)
        except json.JSONDecodeError as e:
            print(f"[SAM3 BatchPlanner] Error parsing track_info JSON: {e}")
            return ("", 0, 0, 0, "{}", 0)

        objects = info.get("objects", [])
        total_frames = info.get("total_frames", 0)

        print(f"[SAM3 BatchPlanner] Input: {len(objects)} objects, {total_frames} frames, mode={batch_mode}")

        # Filter noise
        filtered = []
        filtered_out = []
        for obj in objects:
            first_frame = obj.get("first_frame")
            last_frame = obj.get("last_frame")
            visible_frames = obj.get("visible_frames", 0)

            # Skip objects with no valid frame range
            if first_frame is None or last_frame is None:
                filtered_out.append(obj)
                continue

            # Calculate span and visibility ratio
            span = last_frame - first_frame + 1
            ratio = visible_frames / span if span > 0 else 0

            # Apply filters
            if visible_frames >= min_visible_frames and ratio >= min_visibility_ratio:
                filtered.append(obj)
            else:
                filtered_out.append(obj)

        filtered_count = len(filtered_out)
        print(f"[SAM3 BatchPlanner] Filtered out {filtered_count} noisy tracks, {len(filtered)} remaining")

        if filtered_out:
            for obj in filtered_out:
                print(f"[SAM3 BatchPlanner]   Filtered: Object {obj.get('id')} ({obj.get('visible_frames', 0)} visible frames)")

        # Handle empty case
        if not filtered:
            print("[SAM3 BatchPlanner] Warning: No objects remaining after filtering")
            empty_schedule = json.dumps({
                "batches": [],
                "total_batches": 0,
                "filtered_count": filtered_count,
                "settings": {
                    "batch_mode": batch_mode,
                    "max_objects_per_batch": max_objects_per_batch,
                    "min_visible_frames": min_visible_frames,
                    "min_visibility_ratio": min_visibility_ratio,
                    "sort_by": sort_by
                }
            }, indent=2)
            return ("", 0, 0, 0, empty_schedule, filtered_count)

        # Generate batches based on mode
        if batch_mode == "temporal":
            print(f"[SAM3 BatchPlanner] Using temporal co-occurrence batching")
            batches = self._temporal_batching(filtered, max_objects_per_batch, total_frames)
        elif batch_mode == "overlap_optimized":
            print(f"[SAM3 BatchPlanner] Using overlap-optimized batching (min_overlap={min_overlap_frames} frames)")
            batches = self._overlap_optimized_batching(filtered, max_objects_per_batch, min_overlap_frames)
        else:
            print(f"[SAM3 BatchPlanner] Using stability batching, sorted by {sort_by}")
            batches = self._stability_batching(filtered, max_objects_per_batch, sort_by)

        num_batches = len(batches)

        # Print batch summary
        print(f"[SAM3 BatchPlanner] Created {num_batches} batches:")
        for batch in batches:
            ids_str = ",".join(str(i) for i in batch["object_ids"])
            overlap_str = f" ({batch['overlap_frames']} overlap)" if "overlap_frames" in batch else ""
            print(f"[SAM3 BatchPlanner]   Batch {batch['batch_index']}: Objects [{ids_str}], frames {batch['start_frame']}-{batch['end_frame']}{overlap_str}")

        # Build full schedule JSON
        schedule = {
            "batches": batches,
            "total_batches": num_batches,
            "filtered_count": filtered_count,
            "settings": {
                "batch_mode": batch_mode,
                "max_objects_per_batch": max_objects_per_batch,
                "min_visible_frames": min_visible_frames,
                "min_visibility_ratio": min_visibility_ratio,
                "sort_by": sort_by
            }
        }

        # Get current batch info
        if batch_index >= num_batches:
            if num_batches > 0:
                print(f"[SAM3 BatchPlanner] Warning: batch_index {batch_index} >= num_batches {num_batches}, using last batch")
                batch_index = num_batches - 1
            else:
                return ("", 0, 0, 0, json.dumps(schedule, indent=2), filtered_count)

        current_batch = batches[batch_index]
        batch_object_ids = ",".join(str(i) for i in current_batch["object_ids"])
        batch_start_frame = current_batch["start_frame"]
        batch_end_frame = current_batch["end_frame"]

        print(f"[SAM3 BatchPlanner] Output batch {batch_index}: objects=[{batch_object_ids}], frames={batch_start_frame}-{batch_end_frame}")

        return (
            batch_object_ids,
            batch_start_frame,
            batch_end_frame,
            num_batches,
            json.dumps(schedule, indent=2),
            filtered_count
        )


class SAM3VideoSegmenter:
    """
    Segment video and masks based on batch schedule.

    Takes video frames, multi-object masks, and a batch schedule from BatchPlanner,
    then outputs the actual segmented video/masks for a specific batch.

    Key features:
    - Extracts only the frame range for the batch
    - Extracts only the object masks specified in the batch
    - Provides visualization with colored mask overlays
    - Outputs consistent mask channels (no jumping)
    """

    # Color palette for visualization (RGB, 0-1 range)
    COLORS = [
        [0.0, 0.5, 1.0],   # Blue
        [1.0, 0.3, 0.3],   # Red
        [0.3, 1.0, 0.3],   # Green
        [1.0, 1.0, 0.0],   # Yellow
        [1.0, 0.0, 1.0],   # Magenta
        [0.0, 1.0, 1.0],   # Cyan
        [1.0, 0.5, 0.0],   # Orange
        [0.5, 0.0, 1.0],   # Purple
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Original video frames [N, H, W, C]"
                }),
                "all_masks": ("MASK", {
                    "tooltip": "Multi-object masks from SAM3MaskTracks [N, num_objects, H, W]"
                }),
                "batch_schedule": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON schedule from SAM3BatchPlanner"
                }),
            },
            "optional": {
                "batch_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Which batch to output (0-indexed)"
                }),
                "viz_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask overlay transparency for visualization"
                }),
                "show_ids": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overlay object ID labels on visualization to verify ID consistency"
                }),
                "label_size": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 72,
                    "step": 2,
                    "tooltip": "Font size for ID labels"
                }),
                "margin_head": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 300,
                    "step": 1,
                    "tooltip": "Extra frames to include before mask presence starts (smoother cuts)"
                }),
                "margin_tail": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 300,
                    "step": 1,
                    "tooltip": "Extra frames to include after mask presence ends (smoother cuts)"
                }),
                "mask_colors": ("STRING", {
                    "default": "",
                    "tooltip": "Custom mask colors (comma-separated). Names: red, blue, green, yellow, magenta, cyan, orange, purple, pink, lime, teal, coral, gold, navy. Or hex: #FF0000. Order matches object IDs. Empty = default colors."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("segment_frames", "segment_masks", "segment_combined_mask", "visualization", "segment_info")
    FUNCTION = "segment_video"
    CATEGORY = "SAM3/video"

    def _draw_text(self, frame_np, text, x, y, color, size=24):
        """Draw text label with background on a numpy frame (H, W, C) uint8."""
        font_scale = size / 30.0
        thickness = max(1, int(size / 12))

        # Get text size for centering and background
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Center text on position, clamp to frame bounds
        x = max(0, min(int(x) - text_w // 2, frame_np.shape[1] - text_w - 4))
        y = max(text_h + 4, min(int(y) + text_h // 2, frame_np.shape[0] - 4))

        # Draw black background rectangle for readability
        cv2.rectangle(
            frame_np,
            (x - 2, y - text_h - 2),
            (x + text_w + 2, y + baseline + 2),
            (0, 0, 0),
            -1
        )

        # Draw text in the object's color
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.putText(
            frame_np, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness
        )

        return frame_np

    def _create_visualization(self, frames, masks, object_ids=None, alpha=0.5, show_ids=True, label_size=24, mask_colors=""):
        """
        Create colored mask overlay on video frames with optional ID labels.

        Args:
            frames: Video frames [N, H, W, C]
            masks: Masks [N, num_objects, H, W] or [N, H, W]
            object_ids: List of actual object IDs (for labeling)
            alpha: Mask overlay transparency
            show_ids: Whether to draw ID labels at mask centroids
            label_size: Font size for labels
            mask_colors: Custom color string for visualization
        """
        from .utils import get_color_palette

        num_frames = frames.shape[0]
        h, w = frames.shape[1], frames.shape[2]
        vis_list = []

        # Get color palette based on number of objects
        num_objects = masks.shape[1] if masks.dim() == 4 else 1
        colors = get_color_palette(mask_colors, num_objects)

        for frame_idx in range(num_frames):
            vis_frame = frames[frame_idx].clone()

            if masks.dim() == 4:
                # Multi-object masks [N_frames, N_objects, H, W]
                num_objects = masks.shape[1]

                # First pass: draw mask overlays
                for obj_idx in range(num_objects):
                    obj_mask = masks[frame_idx, obj_idx].float()
                    if obj_mask.max() > 1.0:
                        obj_mask = obj_mask / 255.0

                    color = torch.tensor(colors[obj_idx % len(colors)])
                    mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                    vis_frame = vis_frame * (1 - alpha * obj_mask.unsqueeze(-1)) + alpha * mask_rgb

                vis_frame = vis_frame.clamp(0, 1)

                # Second pass: draw ID labels (after all masks so labels are on top)
                if show_ids and object_ids is not None:
                    # Convert to numpy for cv2 text rendering
                    frame_np = (vis_frame.cpu().numpy() * 255).astype(np.uint8)

                    for obj_idx in range(num_objects):
                        if obj_idx < len(object_ids):
                            obj_id = object_ids[obj_idx]
                            obj_mask = masks[frame_idx, obj_idx].float()
                            if obj_mask.max() > 1.0:
                                obj_mask = obj_mask / 255.0

                            # Only draw label if mask has significant pixels
                            if obj_mask.max() > 0.1:
                                # Find mask centroid
                                y_coords, x_coords = torch.where(obj_mask > 0.5)
                                if len(y_coords) > 0:
                                    cy = float(y_coords.float().mean())
                                    cx = float(x_coords.float().mean())
                                    color = colors[obj_idx % len(colors)]
                                    frame_np = self._draw_text(
                                        frame_np, f"ID:{obj_id}",
                                        cx, cy, color, label_size
                                    )

                    # Convert back to tensor
                    vis_frame = torch.from_numpy(frame_np.astype(np.float32) / 255.0)

            elif masks.dim() == 3:
                # Single combined mask [N_frames, H, W]
                obj_mask = masks[frame_idx].float()
                if obj_mask.max() > 1.0:
                    obj_mask = obj_mask / 255.0

                color = torch.tensor(colors[0])
                mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                vis_frame = vis_frame * (1 - alpha * obj_mask.unsqueeze(-1)) + alpha * mask_rgb
                vis_frame = vis_frame.clamp(0, 1)

            vis_list.append(vis_frame)

        return torch.stack(vis_list, dim=0)

    def segment_video(
        self,
        video_frames,
        all_masks,
        batch_schedule,
        batch_index=0,
        viz_alpha=0.5,
        show_ids=True,
        label_size=24,
        margin_head=0,
        margin_tail=0,
        mask_colors=""
    ):
        """
        Extract video segment and masks for a specific batch.

        margin_head/margin_tail add extra frames before/after the batch's
        frame range to create smoother cuts when video is trimmed.
        """
        # Parse schedule JSON
        try:
            schedule = json.loads(batch_schedule)
        except json.JSONDecodeError as e:
            print(f"[SAM3 VideoSegmenter] Error parsing batch_schedule JSON: {e}")
            empty_info = json.dumps({"error": str(e)})
            return (video_frames[:1], all_masks[:1], all_masks[:1, 0] if all_masks.dim() == 4 else all_masks[:1], video_frames[:1], empty_info)

        batches = schedule.get("batches", [])
        if not batches:
            print("[SAM3 VideoSegmenter] Warning: No batches in schedule")
            empty_info = json.dumps({"error": "No batches in schedule"})
            return (video_frames[:1], all_masks[:1], all_masks[:1, 0] if all_masks.dim() == 4 else all_masks[:1], video_frames[:1], empty_info)

        # Clamp batch_index
        if batch_index >= len(batches):
            print(f"[SAM3 VideoSegmenter] Warning: batch_index {batch_index} >= num_batches {len(batches)}, using last batch")
            batch_index = len(batches) - 1

        batch = batches[batch_index]
        batch_start = batch["start_frame"]
        batch_end = batch["end_frame"]
        object_ids = batch["object_ids"]

        print(f"[SAM3 VideoSegmenter] Processing batch {batch_index}: objects={object_ids}, frames={batch_start}-{batch_end}")

        # Apply head/tail margins and validate frame range
        num_frames = video_frames.shape[0]
        start_frame = max(0, batch_start - margin_head)
        end_frame = min(num_frames - 1, batch_end + margin_tail)

        if margin_head > 0 or margin_tail > 0:
            print(f"[SAM3 VideoSegmenter] Applied margins: head={margin_head}, tail={margin_tail} -> frames={start_frame}-{end_frame}")

        # Extract frame range
        segment_frames = video_frames[start_frame:end_frame + 1]
        print(f"[SAM3 VideoSegmenter] Extracted {segment_frames.shape[0]} frames")

        # Extract masks for selected objects
        if all_masks.dim() == 4:
            # Multi-object masks [N_frames, N_objects, H, W]
            num_objects = all_masks.shape[1]

            # Validate object IDs
            valid_ids = [oid for oid in object_ids if 0 <= oid < num_objects]
            if not valid_ids:
                print(f"[SAM3 VideoSegmenter] Warning: No valid object IDs, using object 0")
                valid_ids = [0] if num_objects > 0 else []

            if valid_ids:
                # Extract frame range first, then select objects
                frame_masks = all_masks[start_frame:end_frame + 1]
                segment_masks = frame_masks[:, valid_ids, :, :]
                print(f"[SAM3 VideoSegmenter] Extracted masks for {len(valid_ids)} objects: {valid_ids}")

                # Create combined mask (union)
                segment_combined = segment_masks.max(dim=1)[0]
            else:
                h, w = all_masks.shape[2], all_masks.shape[3]
                segment_masks = torch.zeros(end_frame - start_frame + 1, 1, h, w)
                segment_combined = torch.zeros(end_frame - start_frame + 1, h, w)

        elif all_masks.dim() == 3:
            # Single mask [N_frames, H, W] - just slice frames
            segment_masks = all_masks[start_frame:end_frame + 1].unsqueeze(1)
            segment_combined = all_masks[start_frame:end_frame + 1]
            valid_ids = [0]
            print(f"[SAM3 VideoSegmenter] Single mask input, extracted {segment_masks.shape[0]} frames")

        else:
            print(f"[SAM3 VideoSegmenter] Unexpected mask shape: {all_masks.shape}")
            h, w = video_frames.shape[1], video_frames.shape[2]
            segment_masks = torch.zeros(end_frame - start_frame + 1, 1, h, w)
            segment_combined = torch.zeros(end_frame - start_frame + 1, h, w)
            valid_ids = []

        # Create visualization with ID labels
        visualization = self._create_visualization(
            segment_frames, segment_masks,
            object_ids=valid_ids,
            alpha=viz_alpha,
            show_ids=show_ids,
            label_size=label_size,
            mask_colors=mask_colors
        )

        # Squeeze single-object masks to [N, H, W] for ComfyUI compatibility
        # (KJNodes and other preview nodes expect 3D masks)
        if segment_masks.dim() == 4 and segment_masks.shape[1] == 1:
            segment_masks = segment_masks.squeeze(1)
            print(f"[SAM3 VideoSegmenter] Squeezed single-object mask to shape {segment_masks.shape}")

        # Build segment info
        segment_info = {
            "batch_index": batch_index,
            "object_ids": valid_ids,
            "original_object_ids": object_ids,
            "batch_start_frame": batch_start,
            "batch_end_frame": batch_end,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "margin_head": margin_head,
            "margin_tail": margin_tail,
            "num_frames": segment_frames.shape[0],
            "num_objects": len(valid_ids),
        }

        num_objs_out = segment_masks.shape[1] if segment_masks.dim() == 4 else 1
        print(f"[SAM3 VideoSegmenter] Output: {segment_frames.shape[0]} frames, {num_objs_out} object(s), mask shape {segment_masks.shape}")

        return (
            segment_frames,
            segment_masks,
            segment_combined,
            visualization,
            json.dumps(segment_info, indent=2)
        )


class SAM3MaskRouter:
    """Name-addressed fan-out: K fixed slots, each configured by a subject NAME.

    Solves the repetitive-graph-fan-out problem that NV_SAM3SelectMask doesn't:
    one node, fixed output lanes whose meaning is set by name widgets. The
    operator types subject names they declared far upstream (NV_SAM3SeedBuilder)
    -- never an integer obj_id, never an inspection of the segmentation output.

    Per-slot semantics (mask-router R1 design, 20260610-004124):
      - blank slot name      -> emit an all-zero mask, NOT included in
                                names / masks_batch, no warning (intentionally unused)
      - non-blank + resolves -> that subject's mask, included in names + batch
      - non-blank + missing  -> missing_name_policy: error (raise, listing
                                available subjects) | empty_warn (zero lane + warn,
                                excluded from names/batch)

    Each lane is [F,H,W] for all_frames, or [1,H,W] for the single-frame modes.
    masks_batch is the resolved lanes concatenated subject-major along dim 0;
    names is the newline-joined resolved subject names aligned to that batch.
    """

    NUM_SLOTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        slots = {}
        _defaults = ["head", "body", "hair", "", "", "", "", ""]
        for i in range(cls.NUM_SLOTS):
            slots[f"slot_{i+1}_name"] = ("STRING", {
                "default": _defaults[i],
                "tooltip": (
                    f"Subject name routed to output slot {i+1}. Blank = unused "
                    f"(emits an empty mask, excluded from names/masks_batch)."
                ),
            })
        return {
            "required": {
                "mask_tracks": ("MASK", {
                    "tooltip": "Multi-object mask tensor [F, N_obj, H, W] from NV_SAM3MaskTracks.",
                }),
                "track_info": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "track_info JSON from NV_SAM3MaskTracks (name<->obj_id<->channel tables).",
                }),
                **slots,
                "missing_name_policy": (["error", "empty_warn"], {
                    "default": "error",
                    "tooltip": (
                        "A non-blank slot whose name has no track: error = raise "
                        "(lists available subjects); empty_warn = emit zero lane + warn."
                    ),
                }),
                "frame_mode": (["all_frames", "first_frame", "last_frame", "frame_index"], {
                    "default": "all_frames",
                    "tooltip": "Emit all frames, or a single frame per lane.",
                }),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Frame to emit when frame_mode=frame_index.",
                }),
            },
        }

    RETURN_TYPES = tuple(["MASK"] * NUM_SLOTS + ["STRING", "MASK"])
    RETURN_NAMES = tuple([f"slot_{i+1}_mask" for i in range(NUM_SLOTS)] + ["names", "masks_batch"])
    FUNCTION = "route_masks"
    CATEGORY = "SAM3/video"

    def route_masks(self, mask_tracks, track_info="", missing_name_policy="error",
                    frame_mode="all_frames", frame_index=0, **slot_kwargs):
        if mask_tracks.dim() != 4:
            raise ValueError(
                f"[SAM3 MaskRouter] Expected 4D mask_tracks [F, N_obj, H, W], "
                f"got shape {tuple(mask_tracks.shape)}. Wire NV_SAM3MaskTracks.all_masks."
            )
        num_frames, num_objects, h, w = mask_tracks.shape
        info = _load_track_info(track_info)

        def _frame_slice(t):
            # t is [F, H, W]; reduce to the configured frame window.
            if frame_mode == "all_frames":
                return t
            if frame_mode == "first_frame":
                return t[0:1]
            if frame_mode == "last_frame":
                return t[num_frames - 1:num_frames]
            # frame_index
            idx = max(0, min(frame_index, num_frames - 1))
            return t[idx:idx + 1]

        zero_lane = _frame_slice(torch.zeros(num_frames, h, w, device=mask_tracks.device))

        lane_masks = []
        resolved_names = []
        resolved_lanes = []
        for i in range(self.NUM_SLOTS):
            nm = (slot_kwargs.get(f"slot_{i+1}_name") or "").strip()
            if not nm:
                lane_masks.append(zero_lane)
                continue
            if not info:
                raise ValueError(
                    f"[SAM3 MaskRouter] slot {i+1} requests '{nm}' but track_info "
                    f"is empty. Wire NV_SAM3MaskTracks.track_info into this node."
                )
            try:
                _oid, ch = resolve_subject_name(info, nm)
            except ValueError as e:
                if missing_name_policy == "empty_warn":
                    print(f"[SAM3 MaskRouter] WARN (empty_warn) slot {i+1}: {e}")
                    lane_masks.append(zero_lane)
                    continue
                raise ValueError(f"[SAM3 MaskRouter] slot {i+1}: {e}")
            if ch >= num_objects:
                raise ValueError(
                    f"[SAM3 MaskRouter] slot {i+1} '{nm}' maps to channel {ch} but "
                    f"mask_tracks has only {num_objects} channels. track_info and "
                    f"mask_tracks are from different runs -- re-wire both from the "
                    f"SAME NV_SAM3MaskTracks."
                )
            lane = _frame_slice(mask_tracks[:, ch])
            lane_masks.append(lane)
            resolved_names.append(nm)
            resolved_lanes.append(lane)

        if resolved_lanes:
            masks_batch = torch.cat(resolved_lanes, dim=0)
        else:
            masks_batch = zero_lane  # nothing resolved -> single zero frame-window
        names_str = "\n".join(resolved_names)
        print(
            f"[SAM3 MaskRouter] frame_mode={frame_mode} resolved {len(resolved_names)} "
            f"lane(s): {resolved_names}; masks_batch {tuple(masks_batch.shape)}"
        )
        return tuple(lane_masks + [names_str, masks_batch])


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SAM3MaskTracks": SAM3MaskTracks,
    "NV_SAM3SelectMask": SAM3SelectMask,
    "NV_SAM3MaskRouter": SAM3MaskRouter,
    "NV_SAM3BatchPlanner": SAM3BatchPlanner,
    "NV_SAM3VideoSegmenter": SAM3VideoSegmenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SAM3MaskTracks": "NV SAM3 Mask Tracks",
    "NV_SAM3SelectMask": "NV SAM3 Select Mask",
    "NV_SAM3MaskRouter": "NV SAM3 Mask Router",
    "NV_SAM3BatchPlanner": "NV SAM3 Batch Planner",
    "NV_SAM3VideoSegmenter": "NV SAM3 Video Segmenter",
}
