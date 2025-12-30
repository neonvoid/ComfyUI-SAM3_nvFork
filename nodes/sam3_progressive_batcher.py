"""
SAM3 Progressive Batcher - Single-pass tracking with progressive batch output

This node tracks all detected objects, monitors exits in real-time during propagation,
and groups objects into batches based on exit time similarity. Designed for workflows
where objects have varying screen times (e.g., hockey players entering/exiting frame).

Key features:
- Single-pass propagation (no pre-processing required)
- Real-time exit detection during tracking
- Exit-time aligned batching (groups objects that leave around the same time)
- Memory efficient (doesn't require storing all frames before batching)
"""
import gc
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

import comfy.model_management

from .video_state import (
    SAM3VideoState,
    VideoPrompt,
    VideoConfig,
)
from .inference_reconstructor import (
    get_inference_state,
    invalidate_session,
)


def _get_autocast_context():
    """Get autocast context manager based on GPU capability."""
    if not torch.cuda.is_available():
        return torch.no_grad()
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif major >= 7:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.no_grad()


def print_vram(label: str):
    """Print current VRAM usage for debugging."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {label}: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")


class SAM3ProgressiveBatcher:
    """
    Single-pass tracking with progressive batch output.

    Tracks all detected objects, monitors exits, and outputs batches
    as groups of objects exit together. No need for full pre-processing.

    Key algorithm:
    1. Run propagation, tracking per-object presence frame-by-frame
    2. Detect when each object exits (consecutive empty frames)
    3. Group exited objects by similar exit times
    4. Output batch schedule with frame ranges for downstream processing

    This replaces the two-phase approach (track all → then batch) with a
    single-pass approach that detects exits during propagation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model (from LoadSAM3Model)"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state with prompts for all objects to track"
                }),
            },
            "optional": {
                "max_objects_per_batch": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Maximum number of objects per batch (e.g., 4 for downstream tool limit)"
                }),
                "exit_threshold_frames": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "tooltip": "Consecutive empty frames before object is considered 'exited'. ~0.5s at 30fps."
                }),
                "exit_grouping_frames": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 120,
                    "step": 5,
                    "tooltip": "Max gap between exit times to group objects in same batch. ~1s at 30fps."
                }),
                "min_visible_pixels": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Minimum mask pixels to consider object 'visible' in a frame"
                }),
                "min_visible_frames": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 300,
                    "step": 5,
                    "tooltip": "Minimum frames an object must be visible to be included (noise filter)"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Start frame for propagation"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "End frame for propagation (-1 for all frames)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SCORES", "STRING", "INT", "STRING")
    RETURN_NAMES = ("masks", "scores", "batch_schedule", "num_batches", "track_info")
    FUNCTION = "progressive_batch"
    CATEGORY = "SAM3/video"

    def _get_mask_for_object(self, mask_tensor, obj_idx: int) -> torch.Tensor:
        """Extract mask for a specific object from the mask tensor."""
        if mask_tensor is None:
            return torch.zeros(1)

        if hasattr(mask_tensor, 'cpu'):
            mask_tensor = mask_tensor.cpu()

        # Handle different mask shapes
        if mask_tensor.dim() == 4:
            # [1, num_objects, H, W] - batch dim present
            if obj_idx < mask_tensor.shape[1]:
                return mask_tensor[0, obj_idx]
            return torch.zeros(mask_tensor.shape[2:])
        elif mask_tensor.dim() == 3:
            # [num_objects, H, W]
            if obj_idx < mask_tensor.shape[0]:
                return mask_tensor[obj_idx]
            return torch.zeros(mask_tensor.shape[1:])
        elif mask_tensor.dim() == 2:
            # [H, W] - single object
            return mask_tensor if obj_idx == 0 else torch.zeros_like(mask_tensor)

        return torch.zeros(1)

    def _group_by_exit_time(
        self,
        object_exit_frames: Dict[int, int],
        object_entry_frames: Dict[int, int],
        object_visible_frames: Dict[int, int],
        max_objects_per_batch: int,
        exit_grouping_frames: int,
        min_visible_frames: int,
        total_frames: int
    ) -> List[dict]:
        """
        Group objects into batches by exit time similarity.

        Objects that exit within `exit_grouping_frames` of each other are
        grouped together, up to `max_objects_per_batch` per batch.
        """
        # Filter objects by minimum visible frames
        valid_objects = {
            obj_id: exit_frame
            for obj_id, exit_frame in object_exit_frames.items()
            if object_visible_frames.get(obj_id, 0) >= min_visible_frames
        }

        if not valid_objects:
            print(f"[SAM3 ProgressiveBatcher] No objects met minimum visibility threshold ({min_visible_frames} frames)")
            return []

        # Sort objects by exit time
        sorted_exits = sorted(valid_objects.items(), key=lambda x: x[1])

        print(f"[SAM3 ProgressiveBatcher] Object exit times:")
        for obj_id, exit_frame in sorted_exits:
            entry = object_entry_frames.get(obj_id, 0)
            visible = object_visible_frames.get(obj_id, 0)
            print(f"[SAM3 ProgressiveBatcher]   Object {obj_id}: entry={entry}, exit={exit_frame}, visible={visible} frames")

        # Group by exit time
        batches = []
        current_batch_ids = []
        current_batch_exit = None

        for obj_id, exit_frame in sorted_exits:
            # Start new batch if:
            # 1. Current batch is full, OR
            # 2. This object exits much later than current batch
            should_start_new_batch = False

            if len(current_batch_ids) >= max_objects_per_batch:
                should_start_new_batch = True
            elif current_batch_exit is not None:
                if exit_frame - current_batch_exit > exit_grouping_frames:
                    should_start_new_batch = True

            if should_start_new_batch and current_batch_ids:
                # Finalize current batch
                batch_start = min(object_entry_frames.get(oid, 0) for oid in current_batch_ids)
                batches.append({
                    "batch_index": len(batches),
                    "object_ids": list(current_batch_ids),
                    "start_frame": batch_start,
                    "end_frame": current_batch_exit,
                    "object_count": len(current_batch_ids)
                })
                current_batch_ids = []
                current_batch_exit = None

            current_batch_ids.append(obj_id)
            current_batch_exit = max(current_batch_exit or 0, exit_frame)

        # Don't forget remaining objects
        if current_batch_ids:
            batch_start = min(object_entry_frames.get(oid, 0) for oid in current_batch_ids)
            batches.append({
                "batch_index": len(batches),
                "object_ids": list(current_batch_ids),
                "start_frame": batch_start,
                "end_frame": current_batch_exit,
                "object_count": len(current_batch_ids)
            })

        return batches

    def progressive_batch(
        self,
        sam3_model,
        video_state,
        max_objects_per_batch: int = 4,
        exit_threshold_frames: int = 15,
        exit_grouping_frames: int = 30,
        min_visible_pixels: int = 100,
        min_visible_frames: int = 15,
        start_frame: int = 0,
        end_frame: int = -1
    ):
        """
        Run propagation with inline exit detection and batch grouping.

        Returns masks, scores, and batch schedule for downstream processing.
        """
        print(f"[SAM3 ProgressiveBatcher] Starting progressive batching")
        print(f"[SAM3 ProgressiveBatcher] Settings: max_per_batch={max_objects_per_batch}, "
              f"exit_threshold={exit_threshold_frames}, exit_grouping={exit_grouping_frames}")

        num_frames = video_state.num_frames
        if end_frame < 0 or end_frame >= num_frames:
            end_frame = num_frames - 1

        total_frames = end_frame - start_frame + 1
        print(f"[SAM3 ProgressiveBatcher] Processing frames {start_frame}-{end_frame} ({total_frames} frames)")

        # Tracking state
        object_entry_frames: Dict[int, int] = {}  # obj_id -> first frame seen
        object_exit_frames: Dict[int, int] = {}   # obj_id -> last valid frame (before exit threshold)
        object_visible_frames: Dict[int, int] = {}  # obj_id -> count of visible frames
        consecutive_empty: Dict[int, int] = {}    # obj_id -> consecutive empty frame count
        known_obj_ids: Set[int] = set()
        all_obj_ids_ever: Set[int] = set()

        # Storage for masks and scores
        all_masks: Dict[int, torch.Tensor] = {}
        all_scores: Dict[int, torch.Tensor] = {}

        # Run propagation
        print_vram("Before propagation")

        request = {
            "type": "propagate_in_video",
            "session_id": video_state.session_uuid,
            "propagation_direction": "forward",
            "start_frame_index": start_frame,
            "max_frame_num_to_track": total_frames,
        }

        autocast_context = _get_autocast_context()

        try:
            with autocast_context:
                inference_state = get_inference_state(sam3_model, video_state)
                print_vram("After inference state reconstruction")

                for response in sam3_model.handle_stream_request(request):
                    comfy.model_management.throw_exception_if_processing_interrupted()

                    frame_idx = response.get("frame_index", response.get("frame_idx"))
                    if frame_idx is None:
                        continue

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        continue

                    # Extract mask
                    mask_tensor = None
                    for key in ["out_binary_masks", "video_res_masks", "masks"]:
                        if key in outputs and outputs[key] is not None:
                            mask_tensor = outputs[key]
                            if hasattr(mask_tensor, 'cpu'):
                                mask_tensor = mask_tensor.cpu()
                            break

                    # Extract object IDs from response
                    current_obj_ids = set()
                    if "obj_ids" in outputs:
                        current_obj_ids = set(outputs["obj_ids"])
                    elif mask_tensor is not None:
                        # Infer from mask shape
                        if mask_tensor.dim() == 4:
                            current_obj_ids = set(range(mask_tensor.shape[1]))
                        elif mask_tensor.dim() == 3:
                            current_obj_ids = set(range(mask_tensor.shape[0]))
                        elif mask_tensor.dim() == 2:
                            current_obj_ids = {0}

                    # Track entries (new objects appearing)
                    for obj_id in current_obj_ids - known_obj_ids:
                        if obj_id not in all_obj_ids_ever:
                            object_entry_frames[obj_id] = frame_idx
                            object_visible_frames[obj_id] = 0
                            consecutive_empty[obj_id] = 0
                            all_obj_ids_ever.add(obj_id)
                            print(f"[SAM3 ProgressiveBatcher] Frame {frame_idx}: Object {obj_id} entered")

                    # Track visibility and detect exits
                    for obj_id in all_obj_ids_ever:
                        if obj_id in object_exit_frames:
                            # Already exited, skip
                            continue

                        # Check mask visibility for this object
                        is_visible = False
                        if mask_tensor is not None:
                            obj_idx = obj_id if obj_id in current_obj_ids else -1
                            # Find the actual index in the mask tensor
                            if "obj_ids" in outputs:
                                try:
                                    obj_idx = list(outputs["obj_ids"]).index(obj_id)
                                except ValueError:
                                    obj_idx = -1

                            if obj_idx >= 0:
                                obj_mask = self._get_mask_for_object(mask_tensor, obj_idx)
                                pixel_count = (obj_mask > 0.5).sum().item()
                                if pixel_count >= min_visible_pixels:
                                    is_visible = True
                                    object_visible_frames[obj_id] = object_visible_frames.get(obj_id, 0) + 1
                                    consecutive_empty[obj_id] = 0

                        if not is_visible:
                            consecutive_empty[obj_id] = consecutive_empty.get(obj_id, 0) + 1

                            # Check exit threshold
                            if consecutive_empty[obj_id] >= exit_threshold_frames:
                                exit_frame = frame_idx - exit_threshold_frames
                                if exit_frame >= object_entry_frames.get(obj_id, 0):
                                    object_exit_frames[obj_id] = exit_frame
                                    print(f"[SAM3 ProgressiveBatcher] Frame {frame_idx}: Object {obj_id} exited "
                                          f"(empty for {exit_threshold_frames} consecutive frames)")

                    # Store mask and scores
                    if mask_tensor is not None:
                        all_masks[frame_idx] = {
                            "mask": mask_tensor,
                            "obj_ids": list(current_obj_ids)
                        }

                    # Extract scores
                    for score_key in ["out_probs", "scores", "confidences", "obj_scores"]:
                        if score_key in outputs and outputs[score_key] is not None:
                            probs = outputs[score_key]
                            if hasattr(probs, 'cpu'):
                                probs = probs.cpu()
                            elif isinstance(probs, np.ndarray):
                                probs = torch.from_numpy(probs)
                            all_scores[frame_idx] = probs
                            break

                    known_obj_ids = current_obj_ids

                    # Progress logging
                    if frame_idx % 50 == 0:
                        active_count = len([oid for oid in all_obj_ids_ever if oid not in object_exit_frames])
                        exited_count = len(object_exit_frames)
                        print(f"[SAM3 ProgressiveBatcher] Frame {frame_idx}: {active_count} active, {exited_count} exited")

                    # Memory management
                    if frame_idx % 32 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[SAM3 ProgressiveBatcher] Error during propagation: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print_vram("After propagation")

        # Mark any objects that never explicitly exited as exiting at end of video
        for obj_id in all_obj_ids_ever:
            if obj_id not in object_exit_frames:
                object_exit_frames[obj_id] = end_frame
                print(f"[SAM3 ProgressiveBatcher] Object {obj_id}: still active at end of video, setting exit={end_frame}")

        # Group objects into batches by exit time
        batches = self._group_by_exit_time(
            object_exit_frames=object_exit_frames,
            object_entry_frames=object_entry_frames,
            object_visible_frames=object_visible_frames,
            max_objects_per_batch=max_objects_per_batch,
            exit_grouping_frames=exit_grouping_frames,
            min_visible_frames=min_visible_frames,
            total_frames=total_frames
        )

        # Print batch summary
        print(f"[SAM3 ProgressiveBatcher] Created {len(batches)} batches:")
        for batch in batches:
            ids_str = ",".join(str(i) for i in batch["object_ids"])
            print(f"[SAM3 ProgressiveBatcher]   Batch {batch['batch_index']}: "
                  f"Objects [{ids_str}], frames {batch['start_frame']}-{batch['end_frame']}")

        # Build batch schedule JSON
        schedule = {
            "batches": batches,
            "total_batches": len(batches),
            "total_objects": len(all_obj_ids_ever),
            "total_frames": total_frames,
            "settings": {
                "max_objects_per_batch": max_objects_per_batch,
                "exit_threshold_frames": exit_threshold_frames,
                "exit_grouping_frames": exit_grouping_frames,
                "min_visible_pixels": min_visible_pixels,
                "min_visible_frames": min_visible_frames
            }
        }

        # Build track info for per-object details
        track_info = {
            "objects": [
                {
                    "id": obj_id,
                    "first_frame": object_entry_frames.get(obj_id, 0),
                    "last_frame": object_exit_frames.get(obj_id, end_frame),
                    "visible_frames": object_visible_frames.get(obj_id, 0)
                }
                for obj_id in sorted(all_obj_ids_ever)
            ],
            "total_frames": total_frames,
            "total_objects": len(all_obj_ids_ever)
        }

        return (
            all_masks,
            all_scores,
            json.dumps(schedule, indent=2),
            len(batches),
            json.dumps(track_info, indent=2)
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SAM3ProgressiveBatcher": SAM3ProgressiveBatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SAM3ProgressiveBatcher": "NV SAM3 Progressive Batcher",
}
