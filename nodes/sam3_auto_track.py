"""
SAM3 Auto Track - Automatic multi-instance detection and tracking

This node combines grounding detection with video tracking to automatically
detect and track multiple instances of an object class (e.g., all hockey players)
throughout a video with periodic re-detection.

Key features:
- Periodic grounding detection at configurable keyframe intervals
- IoU-based matching to avoid duplicate object IDs
- Automatic handling of objects entering/exiting the frame
"""
import gc
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

import comfy.model_management

from .video_state import (
    SAM3VideoState,
    VideoPrompt,
    VideoConfig,
    create_video_state,
)
from .utils import (
    comfy_image_to_pil,
    pil_to_comfy_image,
    masks_to_comfy_mask,
    visualize_masks_on_image,
    tensor_to_list,
    cleanup_gpu_memory,
)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def normalize_box(box: List[float], width: int, height: int) -> List[float]:
    """
    Convert pixel coordinates to normalized [0,1] coordinates.

    Args:
        box: [x1, y1, x2, y2] in pixel coordinates
        width: Image width
        height: Image height

    Returns:
        [x1, y1, x2, y2] in normalized coordinates
    """
    return [
        box[0] / width,
        box[1] / height,
        box[2] / width,
        box[3] / height,
    ]


class SAM3AutoTrack:
    """
    Automatic multi-instance detection and tracking for video.

    Detects all instances of an object class (e.g., "hockey player") at regular
    intervals throughout the video and sets up tracking for each detected object.

    Uses IoU matching to avoid creating duplicate tracks for the same object.

    Workflow:
    1. Creates video state from input frames
    2. Runs grounding detection at each keyframe interval
    3. Matches detections to existing tracks using IoU
    4. Creates new tracks for unmatched detections
    5. Returns video state ready for SAM3Propagate
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model from LoadSAM3Model node"
                }),
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch of images [N, H, W, C]"
                }),
                "text_prompt": ("STRING", {
                    "default": "person",
                    "multiline": False,
                    "placeholder": "e.g., 'hockey player', 'person', 'car'",
                    "tooltip": "Object class to detect and track"
                }),
            },
            "optional": {
                "keyframe_interval": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "tooltip": "Frames between re-detection (default 30 = ~1 sec at 30fps). Lower = catches more entering objects but slower."
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum confidence score for grounding detection"
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "IoU threshold for matching detections to existing tracks. Lower = more likely to merge, Higher = more separate tracks."
                }),
                "max_objects": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum total objects to track (-1 for unlimited). Only applies in keyframe mode."
                }),
                "continuous_detection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, uses SAM3's native text detection on EVERY frame (detects new objects entering scene). If False, only detects at keyframe intervals (faster, but may miss objects entering later)."
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Score threshold for video tracking"
                }),
                "offload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move model to CPU after detection to free VRAM"
                }),
                # Memory offload options for long videos
                "offload_video_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Store video frames on CPU (minor overhead, saves ~1-2GB VRAM)"
                }),
                "offload_state_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store inference state on CPU (10-15% slower, saves ~3-5GB VRAM for long videos)"
                }),
                # === Advanced: Hotstart & Detection Tuning ===
                "hotstart_delay": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 60,
                    "step": 1,
                    "tooltip": "Frames before new objects are confirmed. Lower=faster detection of entering objects, Higher=more stable (filters false positives). Set to 0 to disable hotstart filtering."
                }),
                "hotstart_unmatch_thresh": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Unmatched frames within hotstart period before track is removed. Higher=more tolerant of missed detections (good for fast movement/occlusion)."
                }),
                "hotstart_dup_thresh": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Overlapping frames within hotstart period before duplicate track is removed. Higher=more tolerant of overlapping objects."
                }),
                "new_det_thresh": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for creating new object tracks. Lower=easier to detect new objects (may increase false positives)."
                }),
                "assoc_iou_thresh": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "IoU threshold for associating detections with existing tracks. Lower=stricter matching (may create duplicate tracks)."
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE", "INT", "IMAGE")
    RETURN_NAMES = ("video_state", "total_objects", "keyframe_viz")
    FUNCTION = "auto_track"
    CATEGORY = "SAM3/video"

    def _detect_instances(self, sam3_model, pil_image, text_prompt: str,
                          confidence_threshold: float) -> Tuple[Optional[torch.Tensor],
                                                                 Optional[torch.Tensor],
                                                                 Optional[torch.Tensor]]:
        """
        Run grounding detection on a single image.

        Returns:
            Tuple of (masks, boxes, scores) or (None, None, None) if no detections
        """
        processor = sam3_model.processor

        # Update confidence threshold
        processor.set_confidence_threshold(confidence_threshold)

        # Set image and extract features
        state = processor.set_image(pil_image)

        # Add text prompt
        if text_prompt and text_prompt.strip():
            state = processor.set_text_prompt(text_prompt.strip(), state)

        # Extract results
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        # Clean up state
        del state

        if masks is None or len(masks) == 0:
            return None, None, None

        # Sort by score descending
        if scores is not None and len(scores) > 0:
            sorted_indices = torch.argsort(scores, descending=True)
            masks = masks[sorted_indices]
            boxes = boxes[sorted_indices] if boxes is not None else None
            scores = scores[sorted_indices]

        return masks, boxes, scores

    def auto_track(self, sam3_model, video_frames, text_prompt: str,
                   keyframe_interval: int = 30, confidence_threshold: float = 0.2,
                   iou_threshold: float = 0.3, max_objects: int = -1,
                   continuous_detection: bool = True,
                   score_threshold: float = 0.3, offload_model: bool = False,
                   offload_video_to_cpu: bool = True, offload_state_to_cpu: bool = False,
                   hotstart_delay: int = 15, hotstart_unmatch_thresh: int = 8,
                   hotstart_dup_thresh: int = 8, new_det_thresh: float = 0.4,
                   assoc_iou_thresh: float = 0.1):
        """
        Perform automatic multi-instance detection and tracking setup.

        Two modes:
        - continuous_detection=True (default): Uses SAM3's native text prompt detection.
          SAM3 runs detection on EVERY frame during propagation, automatically detecting
          and tracking all objects matching the text prompt including those entering later.

        - continuous_detection=False: Pre-detects at keyframe intervals using grounding,
          then creates box prompts. Faster but may miss objects entering between keyframes.
        """
        # Load model to GPU
        comfy.model_management.load_models_gpu([sam3_model])

        # Sync processor device
        processor = sam3_model.processor
        device = sam3_model.sam3_wrapper.device
        if hasattr(processor, 'sync_device_with_model'):
            processor.sync_device_with_model()

        num_frames = video_frames.shape[0]
        height = video_frames.shape[1]
        width = video_frames.shape[2]

        print(f"[SAM3 AutoTrack] Starting auto-tracking")
        print(f"[SAM3 AutoTrack]   Video: {num_frames} frames, {width}x{height}")
        print(f"[SAM3 AutoTrack]   Text prompt: '{text_prompt}'")
        print(f"[SAM3 AutoTrack]   Mode: {'CONTINUOUS (every frame)' if continuous_detection else f'KEYFRAME (every {keyframe_interval} frames)'}")
        print(f"[SAM3 AutoTrack]   Memory offload: video={offload_video_to_cpu}, state={offload_state_to_cpu}")
        print(f"[SAM3 AutoTrack]   Hotstart config: delay={hotstart_delay}, unmatch_thresh={hotstart_unmatch_thresh}, dup_thresh={hotstart_dup_thresh}")
        print(f"[SAM3 AutoTrack]   Detection config: new_det_thresh={new_det_thresh}, assoc_iou_thresh={assoc_iou_thresh}")

        # 1. Create video state
        config = VideoConfig(
            score_threshold_detection=score_threshold,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            # Hotstart & detection tuning
            hotstart_delay=hotstart_delay,
            hotstart_unmatch_thresh=hotstart_unmatch_thresh,
            hotstart_dup_thresh=hotstart_dup_thresh,
            new_det_thresh=new_det_thresh,
            assoc_iou_thresh=assoc_iou_thresh,
        )
        video_state = create_video_state(video_frames, config=config)

        # ==========================================
        # CONTINUOUS DETECTION MODE (recommended)
        # ==========================================
        if continuous_detection:
            # Use SAM3's native text prompt - detection runs every frame during propagation
            # This ensures ALL objects matching the text are detected throughout the video,
            # including those entering the scene later
            print(f"[SAM3 AutoTrack] Using continuous text detection mode")
            print(f"[SAM3 AutoTrack]   SAM3 will detect '{text_prompt}' on every frame during propagation")

            # Create text prompt - SAM3 handles object tracking automatically via IoU matching
            prompt = VideoPrompt.create_text(
                frame_idx=0,      # Text prompt is applied at frame 0 but affects all frames
                obj_id=1,         # Placeholder - SAM3 assigns actual IDs during detection
                text=text_prompt
            )
            video_state = video_state.with_prompt(prompt)

            # Create simple visualization of first frame
            first_frame = video_frames[0]
            keyframe_viz = first_frame.unsqueeze(0)  # [1, H, W, C]

            print(f"[SAM3 AutoTrack] Complete: Text prompt set for continuous detection")
            print(f"[SAM3 AutoTrack]   Objects will be detected during SAM3Propagate")

            # Offload model if requested
            if offload_model:
                print("[SAM3 AutoTrack] Offloading model to CPU...")
                sam3_model.unpatch_model()
                cleanup_gpu_memory()

            # total_objects is unknown until propagation runs - return -1 to indicate continuous mode
            return (video_state, -1, keyframe_viz)

        # ==========================================
        # KEYFRAME DETECTION MODE (legacy)
        # ==========================================
        print(f"[SAM3 AutoTrack]   Keyframe interval: {keyframe_interval}")
        print(f"[SAM3 AutoTrack]   Confidence threshold: {confidence_threshold}")
        print(f"[SAM3 AutoTrack]   IoU threshold: {iou_threshold}")

        # 2. Determine keyframes
        keyframes = list(range(0, num_frames, keyframe_interval))
        print(f"[SAM3 AutoTrack] Will detect on {len(keyframes)} keyframes: {keyframes[:5]}{'...' if len(keyframes) > 5 else ''}")

        # Track all known objects: {obj_id: last_known_box}
        tracked_objects: Dict[int, List[float]] = {}
        next_obj_id = 1
        keyframe_visualizations = []

        # 3. Process each keyframe
        for keyframe_idx in keyframes:
            print(f"[SAM3 AutoTrack] Processing keyframe {keyframe_idx}/{num_frames-1}")

            # Extract frame and convert to PIL
            frame = video_frames[keyframe_idx]
            pil_image = comfy_image_to_pil(frame.unsqueeze(0))

            # Run grounding detection
            masks, boxes, scores = self._detect_instances(
                sam3_model, pil_image, text_prompt, confidence_threshold
            )

            if masks is None:
                print(f"[SAM3 AutoTrack]   No detections on frame {keyframe_idx}")
                # Add original frame as visualization
                keyframe_visualizations.append(frame)
                continue

            print(f"[SAM3 AutoTrack]   Found {len(masks)} detections")

            # Convert boxes to list format for IoU computation
            boxes_list = boxes.cpu().tolist() if boxes is not None else []
            scores_list = scores.cpu().tolist() if scores is not None else []

            # 4. Match detections to existing tracks
            new_detections = 0
            matched_detections = 0

            for i, (box, score) in enumerate(zip(boxes_list, scores_list)):
                # Skip if we've hit max objects
                if max_objects > 0 and next_obj_id > max_objects:
                    print(f"[SAM3 AutoTrack]   Reached max_objects limit ({max_objects})")
                    break

                # Find best matching existing track
                best_match_id = None
                best_iou = 0

                for obj_id, last_box in tracked_objects.items():
                    iou = compute_iou(box, last_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match_id = obj_id

                if best_match_id is not None:
                    # Update existing track's position
                    tracked_objects[best_match_id] = box
                    matched_detections += 1
                else:
                    # New object detected - add tracking prompt
                    tracked_objects[next_obj_id] = box

                    # Convert box from pixel coords to normalized [0,1]
                    norm_box = normalize_box(box, width, height)

                    # Create video prompt for this object
                    prompt = VideoPrompt.create_box(
                        frame_idx=keyframe_idx,
                        obj_id=next_obj_id,
                        box=norm_box,
                        is_positive=True
                    )
                    video_state = video_state.with_prompt(prompt)

                    print(f"[SAM3 AutoTrack]   New object {next_obj_id}: box={[f'{x:.1f}' for x in box]}, score={score:.3f}")
                    next_obj_id += 1
                    new_detections += 1

            print(f"[SAM3 AutoTrack]   Frame {keyframe_idx}: {new_detections} new, {matched_detections} matched")

            # 5. Create visualization for this keyframe
            vis_image = visualize_masks_on_image(
                pil_image,
                masks,
                boxes,
                scores,
                alpha=0.5
            )
            vis_tensor = pil_to_comfy_image(vis_image)
            keyframe_visualizations.append(vis_tensor.squeeze(0))  # Remove batch dim

            # Cleanup after each keyframe
            del masks, boxes, scores
            gc.collect()

        # Stack visualizations into batch
        if keyframe_visualizations:
            keyframe_viz = torch.stack(keyframe_visualizations, dim=0)
        else:
            # Fallback: single empty frame
            keyframe_viz = torch.zeros(1, height, width, 3)

        total_objects = next_obj_id - 1
        print(f"[SAM3 AutoTrack] Complete: {total_objects} unique objects detected")
        print(f"[SAM3 AutoTrack] Video state has {len(video_state.prompts)} prompts")

        # Offload model if requested
        if offload_model:
            print("[SAM3 AutoTrack] Offloading model to CPU...")
            sam3_model.unpatch_model()
            cleanup_gpu_memory()

        return (video_state, total_objects, keyframe_viz)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SAM3AutoTrack": SAM3AutoTrack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SAM3AutoTrack": "NV SAM3 Auto Track",
}
