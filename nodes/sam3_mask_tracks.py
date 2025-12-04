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
from collections import defaultdict


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
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("selected_masks", "selected_ids", "visualization")
    FUNCTION = "select"
    CATEGORY = "SAM3/video"

    def _create_visualization(self, frames, masks, valid_ids, alpha=0.5):
        """Create colored mask overlay on video frames."""
        num_frames = frames.shape[0]
        h, w = frames.shape[1], frames.shape[2]
        vis_list = []

        for frame_idx in range(num_frames):
            vis_frame = frames[frame_idx].clone()

            # Handle different mask shapes
            if masks.dim() == 3:
                # Single combined mask [N_frames, H, W]
                frame_mask = masks[frame_idx].float()
                if frame_mask.max() > 1.0:
                    frame_mask = frame_mask / 255.0

                # Use first color for combined mask
                color = torch.tensor(self.COLORS[0])
                mask_rgb = frame_mask.unsqueeze(-1) * color.view(1, 1, 3)
                vis_frame = vis_frame * (1 - alpha * frame_mask.unsqueeze(-1)) + alpha * mask_rgb

            elif masks.dim() == 4:
                # Multi-object masks [N_frames, N_objects, H, W]
                for idx, oid in enumerate(valid_ids):
                    if oid < masks.shape[1]:
                        obj_mask = masks[frame_idx, oid].float()
                        if obj_mask.max() > 1.0:
                            obj_mask = obj_mask / 255.0

                        color = torch.tensor(self.COLORS[idx % len(self.COLORS)])
                        mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - alpha * obj_mask.unsqueeze(-1)) + alpha * mask_rgb

            vis_list.append(vis_frame.clamp(0, 1))

        return torch.stack(vis_list, dim=0)

    def select(self, all_masks, object_ids="all", video_frames=None, combine_mode="union", viz_alpha=0.5):
        """
        Select and optionally combine object masks.
        """
        # Handle different input shapes
        if all_masks.dim() == 3:
            # Already [N_frames, H, W] - single object, just return
            print(f"[SAM3 SelectMask] Input is single-object mask, returning as-is")
            num_frames, h, w = all_masks.shape
            # Create visualization if frames provided
            if video_frames is not None:
                visualization = self._create_visualization(video_frames, all_masks, [0], viz_alpha)
            else:
                visualization = torch.zeros(num_frames, h, w, 3)
            return (all_masks, "0", visualization)

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
                visualization = self._create_visualization(video_frames, selected, list(range(len(valid_ids))), viz_alpha)
            else:
                visualization = self._create_visualization(video_frames, selected, valid_ids, viz_alpha)
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
                "batch_mode": (["by_stability", "temporal"], {
                    "default": "by_stability",
                    "tooltip": "by_stability: group by visibility/score. temporal: group by co-occurrence (who's on screen together)"
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

    def plan_batches(
        self,
        track_info,
        batch_mode="by_stability",
        max_objects_per_batch=4,
        min_visible_frames=30,
        min_visibility_ratio=0.1,
        sort_by="visible_frames",
        batch_index=0
    ):
        """
        Parse track_info, filter noise, and generate batch schedule.

        Two modes available:
        - by_stability: Group by visibility/score, process all frames for each batch
        - temporal: Group by co-occurrence, each batch processes only its time window
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
        else:
            print(f"[SAM3 BatchPlanner] Using stability batching, sorted by {sort_by}")
            batches = self._stability_batching(filtered, max_objects_per_batch, sort_by)

        num_batches = len(batches)

        # Print batch summary
        print(f"[SAM3 BatchPlanner] Created {num_batches} batches:")
        for batch in batches:
            ids_str = ",".join(str(i) for i in batch["object_ids"])
            print(f"[SAM3 BatchPlanner]   Batch {batch['batch_index']}: Objects [{ids_str}], frames {batch['start_frame']}-{batch['end_frame']}")

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


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SAM3MaskTracks": SAM3MaskTracks,
    "NV_SAM3SelectMask": SAM3SelectMask,
    "NV_SAM3BatchPlanner": SAM3BatchPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SAM3MaskTracks": "NV SAM3 Mask Tracks",
    "NV_SAM3SelectMask": "NV SAM3 Select Mask",
    "NV_SAM3BatchPlanner": "NV SAM3 Batch Planner",
}
