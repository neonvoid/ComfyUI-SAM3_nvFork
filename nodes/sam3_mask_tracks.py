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
from typing import Dict, List, Optional, Tuple


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
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("all_masks", "track_info", "num_objects")
    FUNCTION = "extract_tracks"
    CATEGORY = "SAM3/video"

    def extract_tracks(self, masks, video_state, scores=None, min_visible_pixels=100):
        """
        Extract all object masks into a single [N_frames, N_objects, H, W] tensor.

        Also computes per-object visibility metadata.
        """
        num_frames = video_state.num_frames
        h, w = video_state.height, video_state.width

        print(f"[SAM3 MaskTracks] Extracting tracks from {num_frames} frames")

        # Determine number of objects from mask data
        num_objects = 0
        for frame_idx in range(num_frames):
            if frame_idx in masks:
                frame_mask = masks[frame_idx]
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)
                if frame_mask.dim() == 3:
                    num_objects = max(num_objects, frame_mask.shape[0])
                elif frame_mask.dim() == 2:
                    num_objects = max(num_objects, 1)

        if num_objects == 0:
            print("[SAM3 MaskTracks] No objects found in masks")
            empty_masks = torch.zeros(num_frames, 1, h, w)
            track_info = json.dumps({
                "objects": [],
                "total_frames": num_frames,
                "total_objects": 0,
            }, indent=2)
            return (empty_masks, track_info, 0)

        print(f"[SAM3 MaskTracks] Found {num_objects} objects")

        # Build [N_frames, N_objects, H, W] tensor
        all_masks = torch.zeros(num_frames, num_objects, h, w)

        # Track per-object visibility info
        object_info = {
            i: {
                "first_frame": None,
                "last_frame": None,
                "visible_frames": 0,
                "total_pixels": 0,
                "scores": []
            }
            for i in range(num_objects)
        }

        for frame_idx in range(num_frames):
            if frame_idx in masks:
                frame_mask = masks[frame_idx]

                # Convert numpy to torch if needed
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)

                # Handle different mask shapes
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)  # Remove batch dim

                if frame_mask.dim() == 3:
                    # Multi-object mask [N_obj, H, W]
                    for oid in range(min(frame_mask.shape[0], num_objects)):
                        obj_mask = frame_mask[oid].float()

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

                        all_masks[frame_idx, oid] = obj_mask

                        # Track visibility
                        pixel_count = (obj_mask > 0.5).sum().item()
                        if pixel_count >= min_visible_pixels:
                            if object_info[oid]["first_frame"] is None:
                                object_info[oid]["first_frame"] = frame_idx
                            object_info[oid]["last_frame"] = frame_idx
                            object_info[oid]["visible_frames"] += 1
                            object_info[oid]["total_pixels"] += pixel_count

                elif frame_mask.dim() == 2:
                    # Single mask [H, W]
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

                    all_masks[frame_idx, 0] = obj_mask

                    pixel_count = (obj_mask > 0.5).sum().item()
                    if pixel_count >= min_visible_pixels:
                        if object_info[0]["first_frame"] is None:
                            object_info[0]["first_frame"] = frame_idx
                        object_info[0]["last_frame"] = frame_idx
                        object_info[0]["visible_frames"] += 1
                        object_info[0]["total_pixels"] += pixel_count

            # Collect scores if available
            if scores is not None and frame_idx in scores:
                frame_scores = scores[frame_idx]
                if hasattr(frame_scores, 'tolist'):
                    score_list = frame_scores.tolist()
                    if score_list and isinstance(score_list[0], list):
                        score_list = score_list[0]
                    for oid, score in enumerate(score_list):
                        if oid < num_objects:
                            object_info[oid]["scores"].append(score)

        # Compute average scores
        for oid in range(num_objects):
            if object_info[oid]["scores"]:
                object_info[oid]["avg_score"] = sum(object_info[oid]["scores"]) / len(object_info[oid]["scores"])
            else:
                object_info[oid]["avg_score"] = None

        # Build track_info JSON
        track_info = {
            "objects": [
                {
                    "id": oid,
                    "first_frame": info["first_frame"],
                    "last_frame": info["last_frame"],
                    "visible_frames": info["visible_frames"],
                    "avg_score": round(info["avg_score"], 4) if info["avg_score"] is not None else None,
                }
                for oid, info in object_info.items()
                if info["first_frame"] is not None
            ],
            "total_frames": num_frames,
            "total_objects": num_objects,
            "dimensions": {"height": h, "width": w},
        }

        # Print summary
        for obj in track_info["objects"]:
            score_str = f", avg_score={obj['avg_score']:.3f}" if obj['avg_score'] else ""
            print(f"[SAM3 MaskTracks]   Object {obj['id']}: frames {obj['first_frame']}-{obj['last_frame']} ({obj['visible_frames']} visible){score_str}")

        return (all_masks, json.dumps(track_info, indent=2), num_objects)


class SAM3SelectMask:
    """
    Select specific object(s) from multi-track mask output.

    Fast operation - no re-inference needed, just tensor slicing.

    Examples:
    - object_ids="0" → Just object 0's mask [N_frames, H, W]
    - object_ids="0,2" + combine_mode="union" → Combined mask of objects 0 and 2
    - object_ids="all" + combine_mode="separate" → All masks [N_frames, N_obj, H, W]
    """

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
                "combine_mode": (["union", "separate", "first"], {
                    "default": "union",
                    "tooltip": "How to combine multiple objects: union (OR), separate (keep channels), first (just first selected)"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("selected_masks", "selected_ids")
    FUNCTION = "select"
    CATEGORY = "SAM3/video"

    def select(self, all_masks, object_ids="all", combine_mode="union"):
        """
        Select and optionally combine object masks.
        """
        # Handle different input shapes
        if all_masks.dim() == 3:
            # Already [N_frames, H, W] - single object, just return
            print(f"[SAM3 SelectMask] Input is single-object mask, returning as-is")
            return (all_masks, "0")

        if all_masks.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N_frames, N_objects, H, W], got shape {all_masks.shape}")

        num_frames, num_objects, h, w = all_masks.shape
        print(f"[SAM3 SelectMask] Input: {num_frames} frames, {num_objects} objects")

        # Parse object_ids
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
            # No objects - return empty mask
            return (torch.zeros(num_frames, h, w), "")

        # Select and combine based on mode
        if combine_mode == "union":
            # Combine all selected objects with max (OR operation)
            selected = torch.zeros(num_frames, h, w, device=all_masks.device)
            for oid in valid_ids:
                selected = torch.max(selected, all_masks[:, oid])
            return (selected, ",".join(str(i) for i in valid_ids))

        elif combine_mode == "separate":
            # Keep as separate channels
            selected = all_masks[:, valid_ids]
            # If only one object, squeeze to [N_frames, H, W]
            if len(valid_ids) == 1:
                selected = selected.squeeze(1)
            return (selected, ",".join(str(i) for i in valid_ids))

        elif combine_mode == "first":
            # Just return the first selected object
            selected = all_masks[:, valid_ids[0]]
            return (selected, str(valid_ids[0]))

        else:
            # Default to union
            selected = torch.zeros(num_frames, h, w, device=all_masks.device)
            for oid in valid_ids:
                selected = torch.max(selected, all_masks[:, oid])
            return (selected, ",".join(str(i) for i in valid_ids))


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SAM3MaskTracks": SAM3MaskTracks,
    "NV_SAM3SelectMask": SAM3SelectMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SAM3MaskTracks": "NV SAM3 Mask Tracks",
    "NV_SAM3SelectMask": "NV SAM3 Select Mask",
}
