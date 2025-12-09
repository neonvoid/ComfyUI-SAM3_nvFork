"""
SAM3 Video Tracking Nodes for ComfyUI - Stateless Architecture

These nodes provide video object tracking and segmentation using SAM3.
All state is encoded in immutable outputs - no global mutable state.

Key design principles:
1. All nodes are stateless - state flows through outputs
2. SAM3VideoState is immutable - adding prompts returns NEW state
3. Inference state is reconstructed on-demand
4. Temp directories are automatically cleaned up at process exit
5. No manual SAM3CloseVideoSession needed
"""
import gc
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import folder_paths
import comfy.model_management

from .video_state import (
    SAM3VideoState,
    VideoPrompt,
    VideoConfig,
    create_video_state,
    create_temp_dir,
    cleanup_temp_dir,
)
from .inference_reconstructor import (
    get_inference_state,
    invalidate_session,
    clear_inference_cache,
)
from .sam3_model_patcher import SAM3ModelWrapper, SAM3ModelPatcher


# =============================================================================
# Autocast dtype detection - handles GPUs without bf16 support
# =============================================================================
def _get_autocast_dtype():
    """
    Get appropriate autocast dtype based on GPU capability.
    Returns None if autocast should not be used.
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere+ supports bf16
        return torch.bfloat16
    elif major >= 7:  # Volta/Turing use fp16
        return torch.float16
    else:
        return None  # Older GPUs - no autocast


def _get_autocast_context():
    """Get autocast context manager based on GPU capability."""
    dtype = _get_autocast_dtype()
    if dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.no_grad()


# =============================================================================
# VRAM Debug Utility
# =============================================================================

def print_vram(label: str, detailed: bool = False):
    """Print current VRAM usage for debugging memory leaks."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {label}: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
        if detailed:
            # Print memory stats breakdown
            stats = torch.cuda.memory_stats()
            print(f"[VRAM]   Active: {stats.get('active_bytes.all.current', 0) / 1024**3:.2f}GB")
            print(f"[VRAM]   Inactive: {stats.get('inactive_split_bytes.all.current', 0) / 1024**3:.2f}GB")
            print(f"[VRAM]   Allocated retries: {stats.get('num_alloc_retries', 0)}")


# =============================================================================
# Bounding Box Utilities for ID Stability
# =============================================================================

def bbox_iou(box1: list, box2: list) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2] format bounding box
        box2: [x1, y1, x2, y2] format bounding box

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# =============================================================================
# VRAM Estimation
# =============================================================================

class SAM3VRAMEstimator:
    """
    Estimate VRAM requirements and max processable frames.

    Use this before SAM3Propagate to check if your video will fit in VRAM.
    Returns estimated max frames and recommended chunk size for chunked mode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state with resolution and frame count"
                }),
            },
            "optional": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model (optional, for more accurate estimation)"
                }),
                "safety_margin_gb": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": "VRAM safety margin in GB (reserve for other processes)"
                }),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "BOOLEAN", "INT", "STRING")
    RETURN_NAMES = ("max_frames", "available_vram_gb", "per_frame_mb", "can_process_all", "recommended_chunk_size", "vram_report")
    FUNCTION = "estimate_vram"
    CATEGORY = "SAM3/video"

    def estimate_vram(self, video_state, sam3_model=None, safety_margin_gb=1.5):
        """
        Estimate VRAM requirements based on video resolution and frame count.

        Empirical formula based on observed memory patterns:
        - Model base: ~3.3GB
        - Frame loading: ~5.7MB per frame (scales with resolution)
        - Propagation: ~2.2MB per frame accumulated (memory bank growth)
        """
        if not torch.cuda.is_available():
            return (0, 0.0, 0.0, False, 0, "CUDA not available")

        # Get video dimensions
        H = video_state.height
        W = video_state.width
        num_frames = video_state.num_frames
        pixels = H * W

        # Get current VRAM status
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        current_used = torch.cuda.memory_allocated() / (1024**3)
        current_reserved = torch.cuda.memory_reserved() / (1024**3)

        # Empirical per-frame costs (based on observed patterns)
        # These scale with resolution relative to 1080p baseline
        resolution_scale = pixels / (1920 * 1080)

        # Base costs (empirical from 1080p testing)
        frame_load_mb = 5.7 * resolution_scale  # Memory for frame in state
        propagation_mb = 2.2 * resolution_scale  # Memory bank growth per frame
        per_frame_total_mb = frame_load_mb + propagation_mb

        # Model overhead (already loaded if sam3_model provided)
        model_overhead_gb = 0.0 if sam3_model is not None else 3.3

        # Available VRAM for frames
        available_gb = total_vram - current_used - safety_margin_gb - model_overhead_gb
        available_mb = available_gb * 1024

        # Calculate max frames
        if per_frame_total_mb > 0:
            max_frames = int(available_mb / per_frame_total_mb)
        else:
            max_frames = num_frames

        max_frames = max(1, max_frames)  # At least 1 frame

        # Check if we can process all frames
        can_process_all = max_frames >= num_frames

        # Recommended chunk size (80% of max for safety)
        recommended_chunk_size = max(50, int(max_frames * 0.8))
        recommended_chunk_size = min(recommended_chunk_size, 500)  # Cap at 500

        # Generate report
        report_lines = [
            f"=== SAM3 VRAM Estimation ===",
            f"Video: {W}x{H}, {num_frames} frames",
            f"Resolution scale: {resolution_scale:.2f}x (vs 1080p)",
            f"",
            f"VRAM Status:",
            f"  Total: {total_vram:.2f} GB",
            f"  Currently used: {current_used:.2f} GB",
            f"  Reserved: {current_reserved:.2f} GB",
            f"  Safety margin: {safety_margin_gb:.1f} GB",
            f"  Available for processing: {available_gb:.2f} GB",
            f"",
            f"Per-frame cost: {per_frame_total_mb:.2f} MB",
            f"  Frame loading: {frame_load_mb:.2f} MB",
            f"  Propagation: {propagation_mb:.2f} MB",
            f"",
            f"Estimation:",
            f"  Max safe frames: {max_frames}",
            f"  Video frames: {num_frames}",
            f"  Can process all: {'YES' if can_process_all else 'NO'}",
        ]

        if not can_process_all:
            report_lines.extend([
                f"",
                f"RECOMMENDATION:",
                f"  Use chunked mode with chunk_size={recommended_chunk_size}",
                f"  Or use stream_to_disk=True for unlimited length",
            ])

        vram_report = "\n".join(report_lines)
        print(f"[SAM3 VRAM] {vram_report}")

        return (
            max_frames,
            round(available_gb, 2),
            round(per_frame_total_mb, 2),
            can_process_all,
            recommended_chunk_size,
            vram_report
        )


# =============================================================================
# Video Segmentation Nodes
# =============================================================================
# NOTE: SAM3VideoModelLoader has been removed.
# Use LoadSAM3Model instead - it returns a unified model that works for both
# image segmentation and video tracking.


# =============================================================================
# Video Segmentation (Unified Node)
# =============================================================================

class SAM3VideoSegmentation:
    """
    Initialize video tracking and add prompts.

    Select prompt_mode to choose between:
    - text: Track objects by text description (comma-separated for multiple)
    - point: Track objects by clicking points (positive/negative)
    - box: Track objects by drawing boxes (positive/negative)

    Note: SAM3 video does NOT support combining different prompt types.
    Each mode is mutually exclusive.
    """
    # Class-level cache for video state results
    _cache = {}

    PROMPT_MODES = ["text", "point", "box"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch of images [N, H, W, C]"
                }),
                "prompt_mode": (cls.PROMPT_MODES, {
                    "default": "text",
                    "tooltip": "Prompt type: text (describe objects), point (click on objects), or box (draw rectangles)"
                }),
            },
            "optional": {
                # Text mode inputs
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "[text mode] Text description(s) to track. Comma-separated for multiple objects (e.g., 'person, dog, car')"
                }),
                # Point mode inputs
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Positive points - click on objects to track"
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Negative points - click on areas to exclude"
                }),
                # Box mode inputs
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Positive boxes - draw around objects to track"
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Negative boxes - draw around areas to exclude"
                }),
                # Common inputs
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to apply prompts (usually 0 for first frame)"
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
                }),
                # Memory offload options
                "offload_video_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Store video frames on CPU (minor overhead, saves ~1-2GB VRAM)"
                }),
                "offload_state_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store inference state on CPU (10-15% slower, saves ~3-5GB VRAM for long videos)"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, video_frames, prompt_mode="text", text_prompt="",
                   positive_points=None, negative_points=None,
                   positive_boxes=None, negative_boxes=None,
                   frame_idx=0, score_threshold=0.3,
                   offload_video_to_cpu=True, offload_state_to_cpu=False):
        # Use a stable hash based on video content
        # Don't use float(mean()) - it has floating point precision issues on GPU
        import hashlib

        # Create a stable hash from video frame content
        # Use shape + corner pixels from first and last frame (deterministic bytes, no float issues)
        h = hashlib.md5()
        h.update(str(video_frames.shape).encode())

        # Sample corner pixels from first and last frame
        first_frame = video_frames[0].cpu().numpy()
        last_frame = video_frames[-1].cpu().numpy()
        h.update(first_frame[0, 0, :].tobytes())      # top-left
        h.update(first_frame[-1, -1, :].tobytes())    # bottom-right
        h.update(last_frame[0, 0, :].tobytes())
        h.update(last_frame[-1, -1, :].tobytes())

        video_hash = h.hexdigest()

        result = hash((
            video_hash,
            prompt_mode,
            text_prompt,
            str(positive_points),
            str(negative_points),
            str(positive_boxes),
            str(negative_boxes),
            frame_idx,
            score_threshold,
            offload_video_to_cpu,
            offload_state_to_cpu,
        ))
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: video_hash={video_hash}, prompt_mode={prompt_mode}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: positive_points={positive_points}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: negative_points={negative_points}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: returning hash={result}")
        return result

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "segment"
    CATEGORY = "SAM3/video"

    def segment(self, video_frames, prompt_mode="text", text_prompt="",
                positive_points=None, negative_points=None,
                positive_boxes=None, negative_boxes=None,
                frame_idx=0, score_threshold=0.3,
                offload_video_to_cpu=True, offload_state_to_cpu=False):
        """Initialize video state and add prompts based on selected mode."""
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(video_frames.shape).encode())
        # Sample corner pixels for video identity
        first_frame = video_frames[0].cpu().numpy()
        last_frame = video_frames[-1].cpu().numpy()
        h.update(first_frame[0, 0, :].tobytes())
        h.update(first_frame[-1, -1, :].tobytes())
        h.update(last_frame[0, 0, :].tobytes())
        h.update(last_frame[-1, -1, :].tobytes())
        h.update(prompt_mode.encode())
        h.update(text_prompt.encode())
        h.update(str(id(positive_points)).encode() if positive_points else b"none")
        h.update(str(id(negative_points)).encode() if negative_points else b"none")
        h.update(str(id(positive_boxes)).encode() if positive_boxes else b"none")
        h.update(str(id(negative_boxes)).encode() if negative_boxes else b"none")
        h.update(str(frame_idx).encode())
        h.update(str(score_threshold).encode())
        h.update(str(offload_video_to_cpu).encode())
        h.update(str(offload_state_to_cpu).encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3VideoSegmentation._cache:
            cached = SAM3VideoSegmentation._cache[cache_key]
            print(f"[SAM3 Video] CACHE HIT - returning cached video_state for key={cache_key[:8]}, session={cached.session_uuid[:8]}")
            return (cached,)

        print(f"[SAM3 Video] CACHE MISS - computing new video_state for key={cache_key[:8]}")
        print_vram("Before video segmentation")

        # 1. Initialize video state
        config = VideoConfig(
            score_threshold_detection=score_threshold,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        video_state = create_video_state(
            video_frames=video_frames,
            config=config,
        )

        print(f"[SAM3 Video] Initialized session {video_state.session_uuid[:8]}")
        print(f"[SAM3 Video] Frames: {video_state.num_frames}, Size: {video_state.width}x{video_state.height}")
        print(f"[SAM3 Video] Prompt mode: {prompt_mode}")

        # 2. Add prompts based on mode (mutually exclusive)
        obj_id = 1

        if prompt_mode == "text":
            # Text mode: parse comma-separated text prompts
            if text_prompt and text_prompt.strip():
                for text in text_prompt.split(","):
                    text = text.strip()
                    if text:
                        prompt = VideoPrompt.create_text(frame_idx, obj_id, text)
                        video_state = video_state.with_prompt(prompt)
                        print(f"[SAM3 Video] Added text prompt: obj={obj_id}, text='{text}'")
                        obj_id += 1
            else:
                print("[SAM3 Video] Warning: text mode selected but no text_prompt provided")

        elif prompt_mode == "point":
            # Point mode: combine positive and negative points
            # Check for multi-object format first
            if positive_points and positive_points.get("objects"):
                # MULTI-OBJECT MODE: Create separate VideoPrompts for each object
                print(f"[SAM3 Video] Multi-object point mode detected")
                for obj_data in positive_points["objects"]:
                    obj_id = obj_data.get("obj_id", 1)
                    pos_pts = obj_data.get("positive_points", [])
                    neg_pts = obj_data.get("negative_points", [])

                    all_points = []
                    all_labels = []

                    for pt in pos_pts:
                        all_points.append([float(pt[0]), float(pt[1])])
                        all_labels.append(1)  # Positive

                    for pt in neg_pts:
                        all_points.append([float(pt[0]), float(pt[1])])
                        all_labels.append(0)  # Negative

                    if all_points:
                        prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
                        video_state = video_state.with_prompt(prompt)
                        print(f"[SAM3 Video] Added point prompt: obj={obj_id}, "
                              f"positive={len(pos_pts)}, negative={len(neg_pts)}")

                if len(positive_points["objects"]) == 0:
                    print("[SAM3 Video] Warning: point mode selected but no objects in multi-object data")
            else:
                # LEGACY SINGLE-OBJECT MODE: Original behavior
                all_points = []
                all_labels = []

                if positive_points and positive_points.get("points"):
                    for pt in positive_points["points"]:
                        all_points.append([float(pt[0]), float(pt[1])])
                        all_labels.append(1)  # Positive

                if negative_points and negative_points.get("points"):
                    for pt in negative_points["points"]:
                        all_points.append([float(pt[0]), float(pt[1])])
                        all_labels.append(0)  # Negative

                if all_points:
                    prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
                    video_state = video_state.with_prompt(prompt)
                    pos_count = len(positive_points.get("points", [])) if positive_points else 0
                    neg_count = len(negative_points.get("points", [])) if negative_points else 0
                    print(f"[SAM3 Video] Added point prompt: obj={obj_id}, "
                          f"positive={pos_count}, negative={neg_count}")
                else:
                    print("[SAM3 Video] Warning: point mode selected but no points provided")

        elif prompt_mode == "box":
            # Box mode: add positive and/or negative boxes
            has_boxes = False

            if positive_boxes and positive_boxes.get("boxes"):
                box_data = positive_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=True)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 Video] Added positive box: obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if negative_boxes and negative_boxes.get("boxes"):
                box_data = negative_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=False)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 Video] Added negative box: obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if not has_boxes:
                print("[SAM3 Video] Warning: box mode selected but no boxes provided")

        # Validate at least one prompt was added
        if len(video_state.prompts) == 0:
            print(f"[SAM3 Video] Warning: No prompts added for mode '{prompt_mode}'")

        print(f"[SAM3 Video] Total prompts: {len(video_state.prompts)}")
        print_vram("After video segmentation")

        # Cache the result
        SAM3VideoSegmentation._cache[cache_key] = video_state

        return (video_state,)


# =============================================================================
# Multi-Frame Prompting
# =============================================================================

class SAM3AddPrompt:
    """
    Add prompts on additional frames to improve tracking.

    Use this to add "correction anchors" on frames where tracking might struggle:
    - After occlusions (player behind goalie)
    - After fast motion (collision, quick direction change)
    - After re-entry (player left and came back)

    Chain multiple SAM3AddPrompt nodes for multiple correction points.

    Example workflow:
        SAM3VideoSegmentation (frame 0) → SAM3AddPrompt (frame 100) → SAM3AddPrompt (frame 200) → SAM3Propagate
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Existing video state with prompts"
                }),
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to add new prompts (different from initial prompts)"
                }),
            },
            "optional": {
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Positive points to add - click on objects to reinforce tracking"
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Negative points - click on areas to exclude"
                }),
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Positive boxes - draw around objects to reinforce"
                }),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Object ID to reinforce (-1 = auto-assign based on existing objects)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "add_prompt"
    CATEGORY = "SAM3/video"

    def add_prompt(self, video_state, frame_idx, positive_points=None, negative_points=None,
                   positive_boxes=None, obj_id=-1):
        """
        Add prompts on an additional frame to reinforce tracking.

        Args:
            video_state: Existing video state with prompts
            frame_idx: Frame index to add new prompts
            positive_points: Points to add (from points editor)
            negative_points: Negative points to exclude
            positive_boxes: Boxes to add
            obj_id: Object ID to reinforce (-1 = auto-assign)

        Returns:
            Updated video_state with new prompts
        """
        print(f"[SAM3 AddPrompt] Adding prompts on frame {frame_idx}")
        print(f"[SAM3 AddPrompt] Existing prompts: {len(video_state.prompts)}")

        # Validate frame_idx
        if frame_idx >= video_state.num_frames:
            print(f"[SAM3 AddPrompt] Warning: frame_idx {frame_idx} >= num_frames {video_state.num_frames}")
            frame_idx = video_state.num_frames - 1

        # Determine starting obj_id
        if obj_id < 0:
            # Auto-assign: use next available or match existing
            existing_ids = video_state.get_object_ids()
            if existing_ids:
                obj_id = max(existing_ids)  # Start from highest existing ID
            else:
                obj_id = 1

        prompts_added = 0

        # Handle point prompts
        if positive_points:
            # Check for multi-object format
            if positive_points.get("objects"):
                # Multi-object mode
                for obj_data in positive_points["objects"]:
                    current_obj_id = obj_data.get("obj_id", obj_id)
                    pos_pts = obj_data.get("positive_points", [])
                    neg_pts = obj_data.get("negative_points", [])

                    all_points = []
                    all_labels = []

                    for pt in pos_pts:
                        all_points.append([float(pt[0]), float(pt[1])])
                        all_labels.append(1)

                    for pt in neg_pts:
                        all_points.append([float(pt[0]), float(pt[1])])
                        all_labels.append(0)

                    if all_points:
                        prompt = VideoPrompt.create_point(frame_idx, current_obj_id, all_points, all_labels)
                        video_state = video_state.with_prompt(prompt)
                        print(f"[SAM3 AddPrompt] Added point prompt: frame={frame_idx}, obj={current_obj_id}, "
                              f"positive={len(pos_pts)}, negative={len(neg_pts)}")
                        prompts_added += 1

            elif positive_points.get("points"):
                # Legacy single-object mode
                all_points = []
                all_labels = []

                for pt in positive_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(1)

                if negative_points and negative_points.get("points"):
                    for pt in negative_points["points"]:
                        all_points.append([float(pt[0]), float(pt[1])])
                        all_labels.append(0)

                if all_points:
                    prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
                    video_state = video_state.with_prompt(prompt)
                    print(f"[SAM3 AddPrompt] Added point prompt: frame={frame_idx}, obj={obj_id}, "
                          f"points={len(all_points)}")
                    prompts_added += 1

        # Handle box prompts
        if positive_boxes and positive_boxes.get("boxes"):
            for box in positive_boxes["boxes"]:
                box_coords = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                prompt = VideoPrompt.create_box(frame_idx, obj_id, box_coords, is_positive=True)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 AddPrompt] Added box prompt: frame={frame_idx}, obj={obj_id}")
                prompts_added += 1
                obj_id += 1

        print(f"[SAM3 AddPrompt] Total prompts added: {prompts_added}")
        print(f"[SAM3 AddPrompt] New total prompts: {len(video_state.prompts)}")

        return (video_state,)


# =============================================================================
# Propagation
# =============================================================================

class SAM3Propagate:
    """
    Run video propagation to track objects across frames.

    Reconstructs inference state on-demand from immutable video state.
    """
    # Class-level cache for propagation results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model (from LoadSAM3Model)"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state with prompts"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Start frame for propagation"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "End frame (-1 for all)"
                }),
                "direction": (["forward", "backward", "both"], {
                    "default": "forward",
                    "tooltip": "Propagation direction: forward (future frames), backward (past frames), or both directions"
                }),
                "offload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move model to CPU after propagation to free VRAM (slower next run)"
                }),
                "enable_chunking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable chunked processing for long videos to prevent OOM. Splits video into chunks and uses mask-guided continuation."
                }),
                "chunk_size": ("INT", {
                    "default": 250,
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                    "tooltip": "Number of frames per chunk. Lower values use less VRAM. 250 is good for ~8GB VRAM."
                }),
                "range_detection_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only detect object entry/exit frames without storing all masks. Prevents OOM on long videos. Returns track_info with first_frame/last_frame per object."
                }),
                "stream_to_disk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write masks to disk during propagation instead of accumulating in memory. Enables processing of arbitrarily long videos. Use SAM3MaskLoader to read masks afterward."
                }),
                "mask_output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Custom directory to save streamed masks (only used when stream_to_disk=True). Leave empty for auto temp dir. Masks are saved as {frame:05d}.npz"
                }),
                "auto_exit_on_empty": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stop propagation when user-selected objects leave frame. Returns only valid frames."
                }),
                "exit_delay_seconds": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Seconds of consecutive empty frames before stopping (scales with video_fps)"
                }),
                "video_fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Video framerate for exit delay calculation"
                }),
                "lock_initial_objects": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stabilize object IDs by remapping new detections back to initial objects when they overlap. Prevents ID reassignment during tracking."
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SCORES", "SAM3_VIDEO_STATE", "STRING", "STRING", "SAM3_VIDEO_OBJ_IDS")
    RETURN_NAMES = ("masks", "scores", "video_state", "track_info", "mask_dir", "obj_ids")
    FUNCTION = "propagate"
    CATEGORY = "SAM3/video"

    @classmethod
    def IS_CHANGED(cls, sam3_model, video_state, start_frame=0, end_frame=-1, direction="forward",
                   offload_model=False, enable_chunking=False, chunk_size=250, range_detection_only=False,
                   stream_to_disk=False, mask_output_path="", auto_exit_on_empty=False,
                   exit_delay_seconds=0.5, video_fps=30.0, lock_initial_objects=True):
        # Use object identity for caching - if upstream node is cached,
        # it returns the same object, so id() will match
        # This is more reliable than hashing content since video_state is immutable
        result = (id(video_state), start_frame, end_frame, direction, enable_chunking, chunk_size, range_detection_only, stream_to_disk, mask_output_path, auto_exit_on_empty, exit_delay_seconds, video_fps, lock_initial_objects)
        print(f"[IS_CHANGED DEBUG] SAM3Propagate: video_state id={id(video_state)}, session={video_state.session_uuid if video_state else None}")
        print(f"[IS_CHANGED DEBUG] SAM3Propagate: returning {result}")
        return result

    def _plan_chunks(self, total_frames: int, chunk_size: int, start_frame: int = 0):
        """
        Split video into chunks for processing.

        Last frame of chunk N = first frame of chunk N+1 (for mask continuity).

        Args:
            total_frames: Total number of frames in range
            chunk_size: Maximum frames per chunk
            start_frame: Global start frame offset

        Returns:
            List of chunk dicts with start_frame, end_frame, is_first, chunk_idx
        """
        chunks = []
        local_start = 0

        while local_start < total_frames:
            local_end = min(local_start + chunk_size - 1, total_frames - 1)
            chunks.append({
                "chunk_idx": len(chunks),
                "local_start": local_start,
                "local_end": local_end,
                "global_start": start_frame + local_start,
                "global_end": start_frame + local_end,
                "is_first": local_start == 0,
            })
            # Next chunk starts at current chunk's last frame (for mask continuity)
            local_start = local_end
            if local_start >= total_frames - 1:
                break

        return chunks

    def _propagate_chunk(self, sam3_model, video_state, chunk, prev_chunk_masks, direction):
        """
        Process a single chunk with mask-guided continuation.

        Args:
            sam3_model: SAM3 model instance
            video_state: Original video state (for first chunk) or chunk-specific state
            chunk: Chunk dict from _plan_chunks
            prev_chunk_masks: Masks from previous chunk's last frame (None for first chunk)
            direction: Propagation direction

        Returns:
            Tuple of (chunk_masks_dict, chunk_scores_dict) with local frame indices
        """
        import os
        import shutil

        chunk_idx = chunk["chunk_idx"]
        global_start = chunk["global_start"]
        global_end = chunk["global_end"]
        is_first = chunk["is_first"]
        num_chunk_frames = global_end - global_start + 1

        print(f"[SAM3 Chunked] Processing chunk {chunk_idx}: frames {global_start}-{global_end} ({num_chunk_frames} frames)")

        # Create a temporary directory for this chunk's frames
        chunk_session_uuid = f"{video_state.session_uuid}_chunk{chunk_idx}"
        chunk_temp_dir = create_temp_dir(chunk_session_uuid)

        try:
            # Copy frames from original temp dir to chunk temp dir with sequential naming
            for local_idx in range(num_chunk_frames):
                global_idx = global_start + local_idx
                src_path = os.path.join(video_state.temp_dir, f"{global_idx:05d}.jpg")
                dst_path = os.path.join(chunk_temp_dir, f"{local_idx:05d}.jpg")
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"[SAM3 Chunked] Warning: Missing frame {src_path}")

            # Create chunk-specific video state
            if is_first:
                # First chunk: use original prompts but adjusted to local frame 0
                # Original prompts may reference any frame, but chunk 0 starts at global_start
                # So we need to remap prompt frames to local indices
                adjusted_prompts = []
                for prompt in video_state.prompts:
                    # Calculate local frame index
                    local_frame = prompt.frame_idx - global_start
                    if 0 <= local_frame < num_chunk_frames:
                        # Create new prompt with adjusted frame index
                        adjusted_prompt = VideoPrompt(
                            frame_idx=local_frame,
                            prompt_type=prompt.prompt_type,
                            obj_id=prompt.obj_id,
                            data=prompt.data
                        )
                        adjusted_prompts.append(adjusted_prompt)
                    else:
                        print(f"[SAM3 Chunked] Warning: Prompt frame {prompt.frame_idx} outside chunk range, skipping")

                chunk_state = SAM3VideoState(
                    session_uuid=chunk_session_uuid,
                    temp_dir=chunk_temp_dir,
                    num_frames=num_chunk_frames,
                    height=video_state.height,
                    width=video_state.width,
                    config=video_state.config,
                    prompts=tuple(adjusted_prompts),
                )
                print(f"[SAM3 Chunked] First chunk: using {len(adjusted_prompts)} prompts (adjusted to local frame indices)")
            else:
                # Subsequent chunks: use mask prompts from previous chunk's last frame
                chunk_state = SAM3VideoState(
                    session_uuid=chunk_session_uuid,
                    temp_dir=chunk_temp_dir,
                    num_frames=num_chunk_frames,
                    height=video_state.height,
                    width=video_state.width,
                    config=video_state.config,
                    prompts=(),  # Start empty, add mask prompts
                )

                # Add mask prompts from previous chunk's last frame
                if prev_chunk_masks is not None:
                    # Convert numpy array to torch tensor if needed
                    if isinstance(prev_chunk_masks, np.ndarray):
                        prev_chunk_masks = torch.from_numpy(prev_chunk_masks)

                    # prev_chunk_masks shape: [num_objects, H, W] or similar
                    # Determine number of objects
                    if prev_chunk_masks.dim() == 2:
                        # Single object mask [H, W]
                        num_objects = 1
                        masks_to_add = [prev_chunk_masks]
                    elif prev_chunk_masks.dim() == 3:
                        # Multi-object mask [num_objects, H, W]
                        num_objects = prev_chunk_masks.shape[0]
                        masks_to_add = [prev_chunk_masks[i] for i in range(num_objects)]
                    elif prev_chunk_masks.dim() == 4:
                        # Batch mask [1, num_objects, H, W]
                        num_objects = prev_chunk_masks.shape[1]
                        masks_to_add = [prev_chunk_masks[0, i] for i in range(num_objects)]
                    else:
                        raise ValueError(f"Unexpected mask shape: {prev_chunk_masks.shape}")

                    print(f"[SAM3 Chunked] Adding {num_objects} mask prompts from previous chunk")
                    for obj_idx, mask in enumerate(masks_to_add):
                        obj_id = obj_idx + 1  # SAM3 uses 1-indexed obj_ids
                        # Normalize mask to 0-1 range if needed
                        if mask.max() > 1.0:
                            mask = mask.float() / 255.0
                        prompt = VideoPrompt.create_mask(frame_idx=0, obj_id=obj_id, mask=mask)
                        chunk_state = chunk_state.with_prompt(prompt)

            # Run propagation for this chunk
            chunk_masks = {}
            chunk_scores = {}

            request = {
                "type": "propagate_in_video",
                "session_id": chunk_state.session_uuid,
                "propagation_direction": direction,
                "start_frame_index": 0,
                "max_frame_num_to_track": num_chunk_frames,
            }

            autocast_context = _get_autocast_context()
            with autocast_context:
                print_vram(f"Chunk {chunk_idx}: Before reconstruction")
                inference_state = get_inference_state(sam3_model, chunk_state)
                print_vram(f"Chunk {chunk_idx}: After reconstruction")

                try:
                    for response in sam3_model.handle_stream_request(request):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        frame_idx = response.get("frame_index", response.get("frame_idx"))
                        if frame_idx is None:
                            continue

                        outputs = response.get("outputs", response)
                        if outputs is None:
                            continue

                        # Extract mask
                        mask_key = None
                        for key in ["out_binary_masks", "video_res_masks", "masks"]:
                            if key in outputs and outputs[key] is not None:
                                mask_key = key
                                break

                        if mask_key:
                            mask = outputs[mask_key]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu()
                            chunk_masks[frame_idx] = mask
                            del outputs[mask_key]

                        # Extract scores
                        for score_key in ["out_probs", "scores", "confidences", "obj_scores"]:
                            if score_key in outputs and outputs[score_key] is not None:
                                probs = outputs[score_key]
                                if hasattr(probs, 'cpu'):
                                    probs = probs.cpu()
                                elif isinstance(probs, np.ndarray):
                                    probs = torch.from_numpy(probs)
                                chunk_scores[frame_idx] = probs
                                del outputs[score_key]
                                break

                        outputs.clear()

                        if frame_idx % 16 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[SAM3 Chunked] Chunk {chunk_idx} error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            print(f"[SAM3 Chunked] Chunk {chunk_idx} complete: {len(chunk_masks)} frames")

            # Clear inference state for this chunk to free memory
            invalidate_session(chunk_state.session_uuid)

        finally:
            # Cleanup chunk temp directory
            cleanup_temp_dir(chunk_temp_dir)

        return chunk_masks, chunk_scores

    def _propagate_with_chunking(self, sam3_model, video_state, start_frame, end_frame, direction, chunk_size):
        """
        Process video in chunks with mask-guided continuation.

        Args:
            sam3_model: SAM3 model instance
            video_state: Original video state
            start_frame: Global start frame
            end_frame: Global end frame
            direction: Propagation direction
            chunk_size: Frames per chunk

        Returns:
            Tuple of (masks_dict, scores_dict) with global frame indices
        """
        total_frames = end_frame - start_frame + 1
        chunks = self._plan_chunks(total_frames, chunk_size, start_frame)

        print(f"[SAM3 Chunked] Processing {total_frames} frames in {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"[SAM3 Chunked]   Chunk {i}: frames {chunk['global_start']}-{chunk['global_end']}")

        all_masks = {}
        all_scores = {}
        prev_chunk_masks = None

        for chunk in chunks:
            # Process chunk
            chunk_masks, chunk_scores = self._propagate_chunk(
                sam3_model, video_state, chunk, prev_chunk_masks, direction
            )

            # Get last frame's mask for next chunk's initialization
            if chunk_masks:
                last_local_frame = max(chunk_masks.keys())
                prev_chunk_masks = chunk_masks[last_local_frame]
                print(f"[SAM3 Chunked] Saved mask from frame {last_local_frame} for next chunk (shape: {prev_chunk_masks.shape if hasattr(prev_chunk_masks, 'shape') else 'N/A'})")

            # Merge chunk results into global results
            # For first chunk, include all frames
            # For subsequent chunks, skip first frame (it's same as prev chunk's last)
            for local_idx, mask in chunk_masks.items():
                global_idx = chunk["global_start"] + local_idx
                if chunk["is_first"] or local_idx > 0:
                    all_masks[global_idx] = mask

            for local_idx, score in chunk_scores.items():
                global_idx = chunk["global_start"] + local_idx
                if chunk["is_first"] or local_idx > 0:
                    all_scores[global_idx] = score

            # Cleanup between chunks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_vram(f"After chunk {chunk['chunk_idx']}")

        print(f"[SAM3 Chunked] All chunks complete: {len(all_masks)} total frames")
        return all_masks, all_scores

    def _propagate_range_detection(self, sam3_model, video_state, start_frame, end_frame, direction):
        """
        Lightweight propagation that only tracks object presence (first/last visible frame).

        This mode prevents OOM on long videos by not accumulating all masks in memory.
        Instead, it only tracks when each object enters and exits the video.

        Args:
            sam3_model: SAM3 model instance
            video_state: Video state with prompts
            start_frame: Start frame index
            end_frame: End frame index
            direction: Propagation direction

        Returns:
            Tuple of (boundary_masks_dict, scores_dict, track_info_json)
        """
        import json

        print(f"[SAM3 Video] RANGE DETECTION MODE: frames {start_frame} to {end_frame}")
        print(f"[SAM3 Video] Prompts: {len(video_state.prompts)}")
        print_vram("Before range detection")

        # Track object presence: {obj_id: {"first": frame, "last": frame, "visible_count": int}}
        object_ranges = {}
        # Store boundary masks: {obj_id: {"first_mask": tensor, "last_mask": tensor, "first_frame": int, "last_frame": int}}
        boundary_masks = {}

        # Build propagation request
        request = {
            "type": "propagate_in_video",
            "session_id": video_state.session_uuid,
            "propagation_direction": direction,
            "start_frame_index": start_frame,
            "max_frame_num_to_track": end_frame - start_frame + 1,
        }

        autocast_context = _get_autocast_context()
        with autocast_context:
            print_vram("Before reconstruction (range detection)")
            inference_state = get_inference_state(sam3_model, video_state)
            print_vram("After reconstruction")

            try:
                for response in sam3_model.handle_stream_request(request):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    frame_idx = response.get("frame_index", response.get("frame_idx"))
                    if frame_idx is None:
                        continue

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        continue

                    # Try different possible mask keys
                    mask = None
                    for key in ["out_binary_masks", "video_res_masks", "masks"]:
                        if key in outputs and outputs[key] is not None:
                            mask = outputs[key]
                            break

                    if mask is None:
                        continue

                    # Move to CPU immediately to avoid GPU accumulation
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu()
                    elif isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)

                    # Determine mask shape and iterate over objects
                    # mask shape is typically [num_objects, H, W] or [1, num_objects, H, W]
                    if mask.dim() == 4:
                        mask = mask.squeeze(0)  # Remove batch dimension

                    num_objects = mask.shape[0] if mask.dim() >= 3 else 1

                    for obj_idx in range(num_objects):
                        obj_id = obj_idx + 1  # SAM3 uses 1-indexed obj_ids

                        if mask.dim() == 3:
                            obj_mask = mask[obj_idx]
                        else:
                            obj_mask = mask

                        # Check if object is visible (mask has significant area)
                        # Use sum > threshold instead of max to avoid noise
                        mask_area = obj_mask.sum().item()
                        is_visible = mask_area > 100  # At least 100 pixels

                        if is_visible:
                            if obj_id not in object_ranges:
                                # First time seeing this object
                                object_ranges[obj_id] = {
                                    "min_frame": frame_idx,
                                    "max_frame": frame_idx,
                                    "visible_count": 1
                                }
                                boundary_masks[obj_id] = {
                                    "first_mask": obj_mask.clone(),
                                    "first_frame": frame_idx,
                                    "last_mask": obj_mask.clone(),
                                    "last_frame": frame_idx
                                }
                            else:
                                # Update min/max frame (handles any iteration order)
                                prev_min = object_ranges[obj_id]["min_frame"]
                                prev_max = object_ranges[obj_id]["max_frame"]
                                object_ranges[obj_id]["visible_count"] += 1

                                if frame_idx < prev_min:
                                    object_ranges[obj_id]["min_frame"] = frame_idx
                                    boundary_masks[obj_id]["first_mask"] = obj_mask.clone()
                                    boundary_masks[obj_id]["first_frame"] = frame_idx
                                if frame_idx > prev_max:
                                    object_ranges[obj_id]["max_frame"] = frame_idx
                                    boundary_masks[obj_id]["last_mask"] = obj_mask.clone()
                                    boundary_masks[obj_id]["last_frame"] = frame_idx

                    # Clear the mask tensor to free memory
                    del mask
                    outputs.clear()

                    # Periodic cleanup - more aggressive in range detection mode
                    if frame_idx % 16 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Progress logging
                    if frame_idx % 50 == 0:
                        print(f"[SAM3 Video] Range detection progress: frame {frame_idx}/{end_frame}")

            except Exception as e:
                print(f"[SAM3 Video] Range detection error: {e}")
                import traceback
                traceback.print_exc()
                raise

        print_vram("After range detection loop")

        # Build track_info JSON
        track_info = {
            "objects": [
                {
                    "id": obj_id,
                    "first_frame": ranges["min_frame"],
                    "last_frame": ranges["max_frame"],
                    "visible_frames": ranges["visible_count"]
                }
                for obj_id, ranges in sorted(object_ranges.items())
            ],
            "total_frames": end_frame - start_frame + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "mode": "range_detection"
        }

        # Log results
        print(f"[SAM3 Video] Range detection complete: {len(object_ranges)} objects tracked")
        for obj_id, ranges in sorted(object_ranges.items()):
            print(f"[SAM3 Video]   Object {obj_id}: frames {ranges['min_frame']}-{ranges['max_frame']} ({ranges['visible_count']} visible)")

        # Build boundary masks output
        # Structure: {frame_idx: mask_tensor} containing only boundary frames
        # For each object, include first and last visible frame masks
        boundary_masks_dict = {}
        for obj_id, masks_data in boundary_masks.items():
            first_frame = masks_data["first_frame"]
            last_frame = masks_data["last_frame"]

            # Initialize frame entries if not present
            if first_frame not in boundary_masks_dict:
                boundary_masks_dict[first_frame] = {}
            if last_frame not in boundary_masks_dict:
                boundary_masks_dict[last_frame] = {}

            # Store masks indexed by obj_id
            boundary_masks_dict[first_frame][obj_id] = masks_data["first_mask"]
            boundary_masks_dict[last_frame][obj_id] = masks_data["last_mask"]

        # Convert to standard format: {frame_idx: stacked_masks_tensor}
        masks_output = {}
        for frame_idx, obj_masks in boundary_masks_dict.items():
            if obj_masks:
                # Stack all object masks for this frame
                max_obj_id = max(obj_masks.keys())
                stacked = []
                for oid in range(1, max_obj_id + 1):
                    if oid in obj_masks:
                        stacked.append(obj_masks[oid])
                    else:
                        # Placeholder empty mask
                        sample_mask = next(iter(obj_masks.values()))
                        stacked.append(torch.zeros_like(sample_mask))
                masks_output[frame_idx] = torch.stack(stacked, dim=0)

        track_info_json = json.dumps(track_info, indent=2)
        return masks_output, {}, track_info_json

    def _propagate_streaming(self, sam3_model, video_state, start_frame, end_frame, direction, custom_mask_path=""):
        """
        Propagation that streams masks to disk as they're processed.

        This mode writes each frame's mask to disk immediately, allowing processing
        of arbitrarily long videos without running out of memory.

        Args:
            sam3_model: SAM3 model instance
            video_state: Video state with prompts
            start_frame: Start frame index
            end_frame: End frame index
            direction: Propagation direction
            custom_mask_path: Optional custom directory for mask output

        Returns:
            Tuple of (empty_masks_dict, empty_scores_dict, track_info_json, mask_dir)
        """
        import os
        import json

        print(f"[SAM3 Video] STREAMING MODE: frames {start_frame} to {end_frame}")
        print(f"[SAM3 Video] Prompts: {len(video_state.prompts)}")
        print_vram("Before streaming propagation")

        # Determine mask output directory
        if custom_mask_path and custom_mask_path.strip():
            mask_dir = custom_mask_path.strip()
            print(f"[SAM3 Video] Using custom mask path: {mask_dir}")
        else:
            mask_dir = os.path.join(video_state.temp_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        # Track object presence (like range detection)
        object_ranges = {}

        # Build propagation request
        request = {
            "type": "propagate_in_video",
            "session_id": video_state.session_uuid,
            "propagation_direction": direction,
            "start_frame_index": start_frame,
            "max_frame_num_to_track": end_frame - start_frame + 1,
        }

        autocast_context = _get_autocast_context()
        with autocast_context:
            print_vram("Before reconstruction (streaming)")
            inference_state = get_inference_state(sam3_model, video_state)
            print_vram("After reconstruction")

            try:
                for response in sam3_model.handle_stream_request(request):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    frame_idx = response.get("frame_index", response.get("frame_idx"))
                    if frame_idx is None:
                        continue

                    # CRITICAL: Clear PREVIOUS frames from SAM3's internal cache IMMEDIATELY
                    # This must happen at the START of each iteration because SAM3 caches
                    # masks BEFORE yielding, and OOM can occur during _postprocess_output
                    # before we even receive this response.
                    cached_outputs = inference_state.get("cached_frame_outputs", {})
                    frames_to_clear = [f for f in list(cached_outputs.keys()) if f < frame_idx]
                    if frames_to_clear:
                        for f in frames_to_clear:
                            del cached_outputs[f]
                        # Also clear associated tracker metadata for old frames
                        tracker_metadata = inference_state.get("tracker_metadata", {})
                        frame_wise_scores = tracker_metadata.get("obj_id_to_tracker_score_frame_wise", {})
                        for f in frames_to_clear:
                            frame_wise_scores.pop(f, None)
                        # Aggressive cleanup after clearing cache
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        continue

                    # Extract mask
                    mask = None
                    for key in ["out_binary_masks", "video_res_masks", "masks"]:
                        if key in outputs and outputs[key] is not None:
                            mask = outputs[key]
                            break

                    if mask is None:
                        continue

                    # Move to CPU immediately
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu()
                    elif isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)

                    # Handle dimensions
                    if mask.dim() == 4:
                        mask = mask.squeeze(0)  # Remove batch dim

                    # Track object visibility (same as range detection)
                    num_objects = mask.shape[0] if mask.dim() >= 3 else 1
                    for obj_idx in range(num_objects):
                        obj_id = obj_idx + 1
                        obj_mask = mask[obj_idx] if mask.dim() == 3 else mask
                        mask_area = obj_mask.sum().item()
                        is_visible = mask_area > 100  # At least 100 pixels

                        if is_visible:
                            if obj_id not in object_ranges:
                                object_ranges[obj_id] = {
                                    "min_frame": frame_idx,
                                    "max_frame": frame_idx,
                                    "visible_count": 1
                                }
                            else:
                                if frame_idx < object_ranges[obj_id]["min_frame"]:
                                    object_ranges[obj_id]["min_frame"] = frame_idx
                                if frame_idx > object_ranges[obj_id]["max_frame"]:
                                    object_ranges[obj_id]["max_frame"] = frame_idx
                                object_ranges[obj_id]["visible_count"] += 1

                    # CRITICAL: Write mask to disk immediately
                    mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.npz")
                    np.savez_compressed(mask_path, mask=mask.numpy())

                    # Also clear the CURRENT frame from cache (after writing to disk)
                    if frame_idx in cached_outputs:
                        del cached_outputs[frame_idx]

                    # Clear reference to allow GC
                    del mask
                    outputs.clear()

                    # Run cleanup every frame during streaming to prevent memory buildup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Progress logging
                    if frame_idx % 50 == 0:
                        print(f"[SAM3 Video] Streaming progress: frame {frame_idx}/{end_frame}")
                        print_vram(f"Streaming frame {frame_idx}")

            except Exception as e:
                print(f"[SAM3 Video] Streaming error: {e}")
                import traceback
                traceback.print_exc()
                raise

        print_vram("After streaming loop")

        # Build track_info (same structure as range detection)
        track_info = {
            "objects": [
                {
                    "id": obj_id,
                    "first_frame": ranges["min_frame"],
                    "last_frame": ranges["max_frame"],
                    "visible_frames": ranges["visible_count"]
                }
                for obj_id, ranges in sorted(object_ranges.items())
            ],
            "total_frames": end_frame - start_frame + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "mode": "streaming",
            "mask_dir": mask_dir
        }

        # Count files written
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.npz')]
        print(f"[SAM3 Video] Streaming complete: {len(mask_files)} masks written to {mask_dir}")
        for obj_id, ranges in sorted(object_ranges.items()):
            print(f"[SAM3 Video]   Object {obj_id}: frames {ranges['min_frame']}-{ranges['max_frame']} ({ranges['visible_count']} visible)")

        return {}, {}, json.dumps(track_info, indent=2), mask_dir

    def propagate(self, sam3_model, video_state, start_frame=0, end_frame=-1, direction="forward",
                  offload_model=False, enable_chunking=False, chunk_size=250, range_detection_only=False,
                  stream_to_disk=False, mask_output_path="", auto_exit_on_empty=False,
                  exit_delay_seconds=0.5, video_fps=30.0, lock_initial_objects=True):
        """Run propagation using reconstructed inference state."""
        # Create cache key using video_state object id (since it's immutable and cached upstream)
        cache_key = (id(video_state), start_frame, end_frame, direction, enable_chunking, chunk_size, range_detection_only, stream_to_disk, mask_output_path, auto_exit_on_empty, exit_delay_seconds, video_fps, lock_initial_objects)

        # Check if we have cached result
        if cache_key in SAM3Propagate._cache:
            cached = SAM3Propagate._cache[cache_key]
            print(f"[SAM3 Propagate] CACHE HIT - returning cached result for session={video_state.session_uuid[:8]}")
            # Still need to handle offload if requested
            if offload_model:
                print("[SAM3 Video] Offloading model to CPU to free VRAM...")
                if hasattr(sam3_model, 'model'):
                    sam3_model.model.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print_vram("After model offload")
            return cached

        print(f"[SAM3 Propagate] CACHE MISS - running propagation for session={video_state.session_uuid[:8]}")

        if len(video_state.prompts) == 0:
            raise ValueError("[SAM3 Video] No prompts added. Add point, box, or text prompts before propagating.")

        # Ensure model is on GPU before inference (may have been offloaded)
        if hasattr(sam3_model, 'model') and hasattr(sam3_model.model, 'to'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sam3_model.model.to(device)

        # Determine frame range
        actual_end_frame = end_frame if end_frame >= 0 else video_state.num_frames - 1
        total_frames = actual_end_frame - start_frame + 1

        # Initialize track_info, mask_dir, and obj_ids (will be populated in range detection/streaming modes)
        track_info_json = ""
        mask_dir = ""
        obj_ids_dict = {}  # For stable color mapping (only populated in standard mode)

        # Branch based on mode
        if stream_to_disk:
            # Streaming mode - write masks to disk as they're processed
            masks_dict, scores_dict, track_info_json, mask_dir = self._propagate_streaming(
                sam3_model, video_state, start_frame, actual_end_frame, direction, mask_output_path
            )
        elif range_detection_only:
            # Lightweight mode - only track object presence
            masks_dict, scores_dict, track_info_json = self._propagate_range_detection(
                sam3_model, video_state, start_frame, actual_end_frame, direction
            )
        elif enable_chunking and total_frames > chunk_size:
            print(f"[SAM3 Video] CHUNKED MODE: {total_frames} frames, chunk_size={chunk_size}")
            masks_dict, scores_dict = self._propagate_with_chunking(
                sam3_model, video_state, start_frame, actual_end_frame, direction, chunk_size
            )
        else:
            # Standard single-pass propagation
            print(f"[SAM3 Video] Starting propagation: frames {start_frame} to {actual_end_frame}")
            print(f"[SAM3 Video] Prompts: {len(video_state.prompts)}")
            print_vram("Before propagation start")

            # Build propagation request - uses predictor's handle_stream_request API
            request = {
                "type": "propagate_in_video",
                "session_id": video_state.session_uuid,
                "propagation_direction": direction,
                "start_frame_index": start_frame,
                "max_frame_num_to_track": actual_end_frame - start_frame + 1,
            }

            # Run ALL inference inside autocast context for dtype consistency
            masks_dict = {}
            scores_dict = {}
            obj_ids_dict = {}  # Store object IDs for stable color mapping

            # Early exit tracking
            consecutive_empty_frames = 0
            last_valid_frame = start_frame
            early_exit_triggered = False
            initial_obj_ids = None  # Track original objects for early exit (not just any objects)
            empty_frames_threshold = max(1, int(exit_delay_seconds * video_fps)) if auto_exit_on_empty else float('inf')
            if auto_exit_on_empty:
                print(f"[SAM3 Video] Early exit enabled: {exit_delay_seconds}s @ {video_fps}fps = {empty_frames_threshold} frames threshold")

            # ID stability tracking (for lock_initial_objects)
            initial_obj_bboxes = {}  # {obj_id: [x1, y1, x2, y2]} - last known bbox per initial object
            id_remap = {}  # {new_id: original_id} - persistent ID remapping

            autocast_context = _get_autocast_context()
            with autocast_context:
                print_vram("Before reconstruction (in autocast)")
                inference_state = get_inference_state(sam3_model, video_state)
                print_vram("After reconstruction")

                # Run propagation
                try:
                    for response in sam3_model.handle_stream_request(request):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        frame_idx = response.get("frame_index", response.get("frame_idx"))
                        if frame_idx is None:
                            continue

                        outputs = response.get("outputs", response)
                        if outputs is None:
                            continue

                        # Try different possible mask keys
                        mask_key = None
                        for key in ["out_binary_masks", "video_res_masks", "masks"]:
                            if key in outputs and outputs[key] is not None:
                                mask_key = key
                                break

                        if mask_key:
                            mask = outputs[mask_key]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu()
                            del outputs[mask_key]

                            # Capture object IDs for stable color mapping
                            frame_obj_ids = None
                            if "obj_ids" in outputs and outputs["obj_ids"] is not None:
                                ids = outputs["obj_ids"]
                                if hasattr(ids, 'tolist'):
                                    ids = ids.tolist()
                                elif isinstance(ids, np.ndarray):
                                    ids = ids.tolist()
                                frame_obj_ids = ids
                                obj_ids_dict[frame_idx] = ids
                                del outputs["obj_ids"]

                            # --- ID stability correction ---
                            # Remap IDs when initial objects disappear and new IDs appear in similar positions
                            if lock_initial_objects and mask is not None and frame_obj_ids:
                                from .sam3_lib.perflib.masks_ops import masks_to_boxes

                                # Capture initial objects and their bboxes on first frame
                                if initial_obj_ids is None:
                                    initial_obj_ids = set(frame_obj_ids)
                                    try:
                                        bboxes = masks_to_boxes(mask, frame_obj_ids)
                                        for i, obj_id in enumerate(frame_obj_ids):
                                            initial_obj_bboxes[obj_id] = bboxes[i].tolist()
                                        print(f"[SAM3 Video] ID stability: locked {len(initial_obj_ids)} initial objects {sorted(initial_obj_ids)}")
                                    except Exception as e:
                                        print(f"[SAM3 Video] Warning: Could not compute initial bboxes: {e}")
                                else:
                                    # Check for potential ID reassignment
                                    current_ids = set(frame_obj_ids)
                                    # Missing: initial IDs that are gone (excluding already remapped ones)
                                    missing_initial = initial_obj_ids - current_ids - set(id_remap.values())
                                    # New: IDs that weren't in initial set (excluding already remapped sources)
                                    new_ids = current_ids - initial_obj_ids - set(id_remap.keys())

                                    if missing_initial and new_ids:
                                        # Compute current bboxes for new IDs
                                        try:
                                            bboxes = masks_to_boxes(mask, frame_obj_ids)
                                            current_bboxes = {}
                                            for i, obj_id in enumerate(frame_obj_ids):
                                                current_bboxes[obj_id] = bboxes[i].tolist()

                                            # Match new IDs to missing initial IDs by bbox IoU
                                            for new_id in list(new_ids):
                                                if new_id not in current_bboxes:
                                                    continue
                                                best_match = None
                                                best_iou = 0.3  # Minimum IoU threshold for remapping
                                                for missing_id in list(missing_initial):
                                                    if missing_id not in initial_obj_bboxes:
                                                        continue
                                                    iou = bbox_iou(initial_obj_bboxes[missing_id], current_bboxes[new_id])
                                                    if iou > best_iou:
                                                        best_iou = iou
                                                        best_match = missing_id

                                                if best_match is not None:
                                                    id_remap[new_id] = best_match
                                                    missing_initial.discard(best_match)
                                                    print(f"[SAM3 Video] ID correction: {new_id} → {best_match} (IoU={best_iou:.2f})")
                                        except Exception as e:
                                            print(f"[SAM3 Video] Warning: ID correction failed: {e}")

                                    # Apply ID remapping to mask and update frame_obj_ids
                                    if id_remap:
                                        # Remap IDs in the mask tensor
                                        for new_id, orig_id in id_remap.items():
                                            if new_id in frame_obj_ids:
                                                mask[mask == new_id] = orig_id
                                        # Update frame_obj_ids list
                                        frame_obj_ids = [id_remap.get(i, i) for i in frame_obj_ids]
                                        obj_ids_dict[frame_idx] = frame_obj_ids

                                    # Update initial object bboxes with current positions (for tracking movement)
                                    try:
                                        bboxes = masks_to_boxes(mask, frame_obj_ids)
                                        for i, obj_id in enumerate(frame_obj_ids):
                                            if obj_id in initial_obj_ids:
                                                initial_obj_bboxes[obj_id] = bboxes[i].tolist()
                                    except Exception:
                                        pass  # Non-critical: keep old bboxes

                            # Bundle mask and obj_ids together for automatic color stability
                            masks_dict[frame_idx] = {
                                "mask": mask,
                                "obj_ids": frame_obj_ids
                            }

                            # Early exit: check if ORIGINAL objects left (not just any objects)
                            # This ensures continuous detection (new objects entering) doesn't prevent early exit
                            if auto_exit_on_empty:
                                # Get current frame's object IDs
                                current_obj_ids = set(frame_obj_ids) if frame_obj_ids else set()

                                # Capture initial objects on first frame with detections
                                if initial_obj_ids is None and current_obj_ids:
                                    initial_obj_ids = current_obj_ids.copy()
                                    print(f"[SAM3 Video] Early exit: tracking initial objects {sorted(initial_obj_ids)}")

                                # Check if original objects remain (not just any objects)
                                if initial_obj_ids:
                                    original_objects_remaining = initial_obj_ids & current_obj_ids
                                    all_original_left = len(original_objects_remaining) == 0
                                else:
                                    # No initial objects captured yet - use original empty check
                                    all_original_left = (mask is None or
                                                        (hasattr(mask, 'shape') and mask.shape[0] == 0) or
                                                        (hasattr(mask, 'numel') and mask.numel() == 0))

                                if all_original_left:
                                    consecutive_empty_frames += 1
                                else:
                                    consecutive_empty_frames = 0
                                    last_valid_frame = frame_idx

                                if consecutive_empty_frames >= empty_frames_threshold:
                                    early_exit_triggered = True
                                    print(f"[SAM3 Video] Early exit: all {len(initial_obj_ids) if initial_obj_ids else 0} initial objects left frame")
                                    print(f"[SAM3 Video] Last valid frame: {last_valid_frame}")
                                    break
                        else:
                            # No mask found - count as empty
                            if auto_exit_on_empty:
                                consecutive_empty_frames += 1
                                if consecutive_empty_frames >= empty_frames_threshold:
                                    early_exit_triggered = True
                                    print(f"[SAM3 Video] Early exit: no masks for {consecutive_empty_frames} consecutive frames")
                                    break

                        # Capture confidence scores
                        for score_key in ["out_probs", "scores", "confidences", "obj_scores"]:
                            if score_key in outputs and outputs[score_key] is not None:
                                probs = outputs[score_key]
                                if hasattr(probs, 'cpu'):
                                    probs = probs.cpu()
                                elif isinstance(probs, np.ndarray):
                                    probs = torch.from_numpy(probs)
                                scores_dict[frame_idx] = probs
                                del outputs[score_key]
                                break

                        outputs.clear()

                        # Periodic cleanup
                        if frame_idx % 16 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print_vram(f"Frame {frame_idx}")

                except Exception as e:
                    print(f"[SAM3 Video] Propagation error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            print_vram("After propagation loop")
            print(f"[SAM3 Video] Propagation complete: {len(masks_dict)} frames processed")
            print(f"[SAM3 Video] Frames with scores: {len(scores_dict)}")

            # Handle early exit: truncate output and create metadata
            if auto_exit_on_empty:
                if early_exit_triggered:
                    # Truncate masks_dict to only include valid frames
                    frames_to_remove = [f for f in masks_dict.keys() if f > last_valid_frame]
                    for f in frames_to_remove:
                        del masks_dict[f]

                    # Truncate scores_dict similarly
                    frames_to_remove = [f for f in scores_dict.keys() if f > last_valid_frame]
                    for f in frames_to_remove:
                        del scores_dict[f]

                    print(f"[SAM3 Video] Truncated output to {len(masks_dict)} valid frames (0-{last_valid_frame})")

                # Create track_info with early exit metadata
                track_info = {
                    "early_exit_enabled": True,
                    "early_exit_triggered": early_exit_triggered,
                    "exit_delay_seconds": exit_delay_seconds,
                    "video_fps": video_fps,
                    "empty_frames_threshold": empty_frames_threshold,
                    "last_valid_frame": last_valid_frame,
                    "total_valid_frames": len(masks_dict),
                    "original_end_frame": actual_end_frame,
                    "start_frame": start_frame,
                }
                track_info_json = json.dumps(track_info, indent=2)

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Offload model to CPU if requested (Issue #28)
        if offload_model:
            print("[SAM3 Video] Offloading model to CPU to free VRAM...")
            if hasattr(sam3_model, 'model'):
                sam3_model.model.cpu()
            # Clear inference state cache to free GPU memory
            from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
            Sam3VideoPredictor._ALL_INFERENCE_STATES.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_vram("After model offload")

        # Cache the result
        result = (masks_dict, scores_dict, video_state, track_info_json, mask_dir, obj_ids_dict)
        SAM3Propagate._cache[cache_key] = result

        return result


# =============================================================================
# Output Extraction
# =============================================================================

class SAM3VideoOutput:
    """
    Extract masks from propagation results.

    Converts SAM3_VIDEO_MASKS to ComfyUI-compatible mask tensors.
    Returns all frames as a batch with all object masks combined.

    For per-object mask selection, use SAM3MaskTracks instead.
    """
    # Class-level cache for extraction results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Masks from SAM3Propagate"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state for dimensions"
                }),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES", {
                    "tooltip": "Confidence scores from SAM3Propagate"
                }),
                "obj_ids": ("SAM3_VIDEO_OBJ_IDS", {
                    "tooltip": "Object IDs from SAM3Propagate (for stable color mapping)"
                }),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Object index for visualization coloring only. Mask output always combines all objects. Use SAM3MaskTracks for per-object masks."
                }),
                "plot_all_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show all object masks in visualization (True) or only selected obj_id (False)"
                }),
                "draw_legend": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw legend on visualization (disable for faster processing)"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, masks, video_state, scores=None, obj_ids=None, obj_id=-1, plot_all_masks=True, draw_legend=True):
        # Always re-run this node when params change, but this is cheap
        # The key is that changing these here does NOT invalidate upstream cache
        # ComfyUI caches based on input values - masks/video_state don't change
        return (id(masks), video_state.session_uuid, id(scores), id(obj_ids), obj_id, plot_all_masks, draw_legend)

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "frames", "visualization")
    FUNCTION = "extract"
    CATEGORY = "SAM3/video"

    def _draw_legend(self, vis_frame, num_objects, colors, obj_id=-1, frame_scores=None):
        """Draw a legend showing object IDs and colors using vectorized operations."""
        h, w = vis_frame.shape[:2]

        # Legend parameters
        box_size = max(16, min(32, h // 20))
        padding = max(4, box_size // 4)
        legend_item_height = box_size + padding

        # Build list of (obj_id, score) pairs
        if obj_id >= 0:
            items = [(obj_id, frame_scores[obj_id] if frame_scores is not None and obj_id < len(frame_scores) else None)]
        else:
            items = []
            for oid in range(num_objects):
                score = frame_scores[oid] if frame_scores is not None and oid < len(frame_scores) else None
                items.append((oid, score))
            # Sort by score descending (highest confidence first), None scores go last
            items.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0)))

        num_items = len(items)
        legend_height = num_items * legend_item_height + padding
        legend_width = box_size + padding * 2

        # Position in top-left corner
        start_x = padding
        start_y = padding

        # Clamp bounds to image size
        end_y = min(start_y + legend_height, h)
        end_x = min(start_x + legend_width, w)

        # Draw semi-transparent background using vectorized operation
        bg_color = torch.tensor([0.1, 0.1, 0.1])
        bg_alpha = 0.7
        vis_frame[start_y:end_y, start_x:end_x] = (
            vis_frame[start_y:end_y, start_x:end_x] * (1 - bg_alpha) + bg_color * bg_alpha
        )

        # Draw legend items using vectorized operations
        for idx, (oid, score) in enumerate(items):
            item_y = start_y + padding + idx * legend_item_height
            box_end_y = min(item_y + box_size, h)
            box_start_x = start_x + padding
            box_end_x = min(box_start_x + box_size, w)

            # Draw color box using tensor slicing (vectorized)
            color = torch.tensor(colors[oid % len(colors)])
            vis_frame[item_y:box_end_y, box_start_x:box_end_x] = color

        return vis_frame

    def extract(self, masks, video_state, scores=None, obj_ids=None, obj_id=-1, plot_all_masks=True, draw_legend=True):
        """Extract all masks as a batch [N, H, W]."""
        from PIL import Image
        import os

        # Create cache key
        cache_key = (id(masks), video_state.session_uuid, id(scores), id(obj_ids), obj_id, plot_all_masks, draw_legend)

        # Check if we have cached result
        if cache_key in SAM3VideoOutput._cache:
            print(f"[SAM3 Video Output] CACHE HIT - returning cached result for session={video_state.session_uuid[:8]}")
            return SAM3VideoOutput._cache[cache_key]

        print(f"[SAM3 Video Output] CACHE MISS - extracting masks for session={video_state.session_uuid[:8]}")
        print_vram("Before extract")
        h, w = video_state.height, video_state.width
        num_frames = video_state.num_frames

        if not masks:
            print("[SAM3 Video] No masks to extract")
            empty_mask = torch.zeros(num_frames, h, w)
            empty_frames = torch.zeros(num_frames, h, w, 3)
            return (empty_mask, empty_frames, empty_frames)

        # Process all frames in order
        mask_list = []
        frame_list = []
        vis_list = []

        # Color palette for multiple objects (RGB, 0-1 range)
        colors = [
            [0.0, 0.5, 1.0],   # Blue
            [1.0, 0.3, 0.3],   # Red
            [0.3, 1.0, 0.3],   # Green
            [1.0, 1.0, 0.0],   # Yellow
            [1.0, 0.0, 1.0],   # Magenta
            [0.0, 1.0, 1.0],   # Cyan
            [1.0, 0.5, 0.0],   # Orange
            [0.5, 0.0, 1.0],   # Purple
        ]

        # Track number of objects for legend
        num_objects = 0

        for frame_idx in range(num_frames):
            comfy.model_management.throw_exception_if_processing_interrupted()
            # Load original frame
            frame_path = os.path.join(video_state.temp_dir, f"{frame_idx:05d}.jpg")
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)  # [H, W, C]
            else:
                img_tensor = torch.zeros(h, w, 3)

            frame_list.append(img_tensor)

            # Get mask for this frame
            if frame_idx in masks:
                mask_data = masks[frame_idx]

                # Check if mask_data is bundled (dict with mask+obj_ids) or legacy (just tensor)
                if isinstance(mask_data, dict):
                    frame_mask = mask_data.get("mask")
                    embedded_obj_ids = mask_data.get("obj_ids")
                else:
                    # Legacy format: just the mask tensor
                    frame_mask = mask_data
                    embedded_obj_ids = None

                # Convert numpy to torch if needed
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)

                # Convert mask to ComfyUI format
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)  # Remove batch dim

                # Create visualization with colored overlays
                vis_frame = img_tensor.clone()

                # Get object IDs for stable color mapping
                # Priority: embedded obj_ids > separate obj_ids input > array index fallback
                frame_obj_ids = embedded_obj_ids
                if frame_obj_ids is None and obj_ids is not None and frame_idx in obj_ids:
                    frame_obj_ids = obj_ids[frame_idx]

                # Check for empty mask (no detections)
                if frame_mask.numel() == 0 or (frame_mask.dim() == 3 and frame_mask.shape[0] == 0):
                    # No detections - use empty mask
                    frame_mask = torch.zeros(h, w)
                    # vis_frame stays as original image
                elif frame_mask.dim() == 3 and frame_mask.shape[0] >= 1:
                    num_objects = max(num_objects, frame_mask.shape[0])
                    combined_mask = torch.zeros(h, w)

                    if plot_all_masks:
                        # Show ALL objects with different colors
                        for oid in range(frame_mask.shape[0]):
                            obj_mask = frame_mask[oid].float()
                            if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                                obj_mask = obj_mask / 255.0
                            # Use stable object ID for color (if available)
                            if frame_obj_ids is not None and oid < len(frame_obj_ids):
                                color_id = frame_obj_ids[oid]
                            else:
                                color_id = oid
                            color = torch.tensor(colors[color_id % len(colors)])
                            mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                            vis_frame = vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
                            combined_mask = torch.max(combined_mask, obj_mask)
                    else:
                        # Show only selected obj_id
                        vis_oid = obj_id if obj_id >= 0 and obj_id < frame_mask.shape[0] else 0
                        obj_mask = frame_mask[vis_oid].float()
                        if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                            obj_mask = obj_mask / 255.0
                        # Use stable object ID for color (if available)
                        if frame_obj_ids is not None and vis_oid < len(frame_obj_ids):
                            color_id = frame_obj_ids[vis_oid]
                        else:
                            color_id = vis_oid
                        color = torch.tensor(colors[color_id % len(colors)])
                        mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
                        # Still compute combined for mask output
                        for oid in range(frame_mask.shape[0]):
                            om = frame_mask[oid].float()
                            if om.numel() > 0 and om.max() > 1.0:
                                om = om / 255.0
                            combined_mask = torch.max(combined_mask, om)

                    # Always output combined mask (all objects merged)
                    # Note: obj_id selection was broken - it treated obj_id as array index
                    # but SAM3 object IDs don't match array indices across frames
                    frame_mask = combined_mask
                else:
                    # Single mask
                    if frame_mask.dim() == 3:
                        frame_mask = frame_mask.squeeze(0)
                    frame_mask = frame_mask.float()
                    if frame_mask.numel() > 0 and frame_mask.max() > 1.0:
                        frame_mask = frame_mask / 255.0
                    num_objects = max(num_objects, 1)
                    color = torch.tensor(colors[0])
                    mask_rgb = frame_mask.unsqueeze(-1) * color.view(1, 1, 3)
                    vis_frame = vis_frame * (1 - 0.5 * frame_mask.unsqueeze(-1)) + 0.5 * mask_rgb

                # Final check for empty masks
                if frame_mask.numel() == 0:
                    frame_mask = torch.zeros(h, w)

                # Draw legend on visualization (skip if disabled for performance)
                if draw_legend and num_objects > 0:
                    legend_obj_id = -1 if plot_all_masks else obj_id
                    # Get scores for this frame
                    frame_scores = None
                    if scores is not None and frame_idx in scores:
                        frame_scores_tensor = scores[frame_idx]
                        if hasattr(frame_scores_tensor, 'tolist'):
                            frame_scores = frame_scores_tensor.tolist()
                            # Handle nested lists (e.g., [[0.95, 0.87]])
                            if frame_scores and isinstance(frame_scores[0], list):
                                frame_scores = frame_scores[0]
                        elif hasattr(frame_scores_tensor, '__iter__'):
                            frame_scores = list(frame_scores_tensor)
                    vis_frame = self._draw_legend(vis_frame, num_objects, colors, obj_id=legend_obj_id, frame_scores=frame_scores)

                vis_list.append(vis_frame.clamp(0, 1))
            else:
                # No mask for this frame - use zeros
                frame_mask = torch.zeros(h, w)
                vis_list.append(img_tensor)

            mask_list.append(frame_mask.cpu())

        # Stack into batches
        all_masks = torch.stack(mask_list, dim=0)  # [N, H, W]
        all_frames = torch.stack(frame_list, dim=0)  # [N, H, W, C]
        all_vis = torch.stack(vis_list, dim=0)  # [N, H, W, C]

        print(f"[SAM3 Video] Output: {all_masks.shape[0]} masks, shape {all_masks.shape}")
        print(f"[SAM3 Video] Objects tracked: {num_objects}, plot_all_masks: {plot_all_masks}")
        print_vram("After extract")

        # Cache the result
        result = (all_masks, all_frames, all_vis)
        SAM3VideoOutput._cache[cache_key] = result

        return result


# =============================================================================
# Video Trimming (for early exit workflow)
# =============================================================================

class SAM3VideoTrim:
    """
    Trim video frames and extract masks based on track_info from SAM3Propagate.

    Use this after SAM3Propagate with auto_exit_on_empty=True to:
    1. Trim original video frames to match the truncated mask output
    2. Extract and combine masks for the valid frame range
    3. Generate colored visualization overlay

    This is a complete endpoint node - outputs trimmed video, masks, and visualization.
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
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Masks dict from SAM3Propagate"
                }),
                "track_info": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON track info from SAM3Propagate (with early exit metadata)"
                }),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES", {
                    "tooltip": "Confidence scores from SAM3Propagate"
                }),
                "obj_ids": ("SAM3_VIDEO_OBJ_IDS", {
                    "tooltip": "Object IDs from SAM3Propagate (for stable color mapping)"
                }),
                "viz_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask overlay transparency for visualization"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("trimmed_frames", "masks", "visualization", "start_frame", "end_frame", "total_frames")
    FUNCTION = "trim_video"
    CATEGORY = "SAM3/video"

    def trim_video(self, video_frames, masks, track_info, scores=None, obj_ids=None, viz_alpha=0.5):
        """
        Trim video frames and extract masks/visualization for valid frame range.

        Args:
            video_frames: Original video frames [N, H, W, C]
            masks: Dict of frame_idx -> mask tensor from SAM3Propagate
            track_info: JSON string with early exit metadata
            scores: Optional confidence scores dict
            viz_alpha: Transparency for mask overlay

        Returns:
            trimmed_frames: Sliced video frames
            masks: Combined mask tensor [N, H, W]
            visualization: Colored overlay frames [N, H, W, C]
            start_frame: Start frame index
            end_frame: End frame index (inclusive)
            total_frames: Number of frames in output
        """
        # Parse track_info JSON
        try:
            info = json.loads(track_info)
        except json.JSONDecodeError as e:
            print(f"[SAM3 VideoTrim] Error parsing track_info JSON: {e}")
            print(f"[SAM3 VideoTrim] Returning original frames unchanged")
            h, w = video_frames.shape[1], video_frames.shape[2]
            empty_mask = torch.zeros(video_frames.shape[0], h, w)
            return (video_frames, empty_mask, video_frames.clone(), 0, video_frames.shape[0] - 1, video_frames.shape[0])

        # Extract frame range
        start_frame = info.get("start_frame", 0)
        last_valid_frame = info.get("last_valid_frame", video_frames.shape[0] - 1)
        early_exit_triggered = info.get("early_exit_triggered", False)

        # Validate frame range
        num_input_frames = video_frames.shape[0]
        start_frame = max(0, min(start_frame, num_input_frames - 1))
        end_frame = max(start_frame, min(last_valid_frame, num_input_frames - 1))

        # Slice video frames
        trimmed = video_frames[start_frame:end_frame + 1]
        total_frames = trimmed.shape[0]
        h, w = trimmed.shape[1], trimmed.shape[2]

        if early_exit_triggered:
            print(f"[SAM3 VideoTrim] Early exit detected - trimming video")
            print(f"[SAM3 VideoTrim] Original: {num_input_frames} frames")
            print(f"[SAM3 VideoTrim] Trimmed: frames {start_frame}-{end_frame} ({total_frames} frames)")
        else:
            print(f"[SAM3 VideoTrim] No early exit - frames {start_frame}-{end_frame} ({total_frames} frames)")

        # Extract and process masks for valid frame range
        mask_list = []
        vis_list = []

        for out_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            # Get frame from trimmed video
            frame_tensor = trimmed[out_idx]
            vis_frame = frame_tensor.clone()

            # Get mask for this frame
            if frame_idx in masks:
                mask_data = masks[frame_idx]

                # Check if mask_data is bundled (dict with mask+obj_ids) or legacy (just tensor)
                if isinstance(mask_data, dict):
                    frame_mask = mask_data.get("mask")
                    embedded_obj_ids = mask_data.get("obj_ids")
                else:
                    # Legacy format: just the mask tensor
                    frame_mask = mask_data
                    embedded_obj_ids = None

                # Convert numpy to torch if needed
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)

                # Handle dimensions
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)  # Remove batch dim

                # Get object IDs for stable color mapping
                # Priority: embedded obj_ids > separate obj_ids input > array index fallback
                frame_obj_ids = embedded_obj_ids
                if frame_obj_ids is None and obj_ids is not None and frame_idx in obj_ids:
                    frame_obj_ids = obj_ids[frame_idx]

                # Check for empty mask
                if frame_mask.numel() == 0 or (frame_mask.dim() == 3 and frame_mask.shape[0] == 0):
                    combined_mask = torch.zeros(h, w)
                elif frame_mask.dim() == 3 and frame_mask.shape[0] >= 1:
                    # Multi-object mask - combine all and create visualization
                    combined_mask = torch.zeros(h, w)
                    for obj_idx in range(frame_mask.shape[0]):
                        obj_mask = frame_mask[obj_idx].float()
                        if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                            obj_mask = obj_mask / 255.0
                        # Add to combined mask
                        combined_mask = torch.max(combined_mask, obj_mask)

                        # Use stable object ID for color (if available), else fall back to array index
                        if frame_obj_ids is not None and obj_idx < len(frame_obj_ids):
                            color_id = frame_obj_ids[obj_idx]
                        else:
                            color_id = obj_idx
                        color = torch.tensor(self.COLORS[color_id % len(self.COLORS)])

                        # Add colored overlay to visualization
                        mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - viz_alpha * obj_mask.unsqueeze(-1)) + viz_alpha * mask_rgb
                else:
                    # Single mask
                    if frame_mask.dim() == 3:
                        frame_mask = frame_mask.squeeze(0)
                    combined_mask = frame_mask.float()
                    if combined_mask.numel() > 0 and combined_mask.max() > 1.0:
                        combined_mask = combined_mask / 255.0

                    # Use stable object ID for color (if available)
                    if frame_obj_ids is not None and len(frame_obj_ids) > 0:
                        color_id = frame_obj_ids[0]
                    else:
                        color_id = 0
                    color = torch.tensor(self.COLORS[color_id % len(self.COLORS)])

                    # Add colored overlay
                    mask_rgb = combined_mask.unsqueeze(-1) * color.view(1, 1, 3)
                    vis_frame = vis_frame * (1 - viz_alpha * combined_mask.unsqueeze(-1)) + viz_alpha * mask_rgb
            else:
                # No mask for this frame
                combined_mask = torch.zeros(h, w)

            mask_list.append(combined_mask.cpu())
            vis_list.append(vis_frame.clamp(0, 1))

        # Stack into batches
        all_masks = torch.stack(mask_list, dim=0)  # [N, H, W]
        all_vis = torch.stack(vis_list, dim=0)  # [N, H, W, C]

        print(f"[SAM3 VideoTrim] Output: {total_frames} frames, masks shape {all_masks.shape}")

        return (trimmed, all_masks, all_vis, start_frame, end_frame, total_frames)


# =============================================================================
# Mask Loading (for streaming mode)
# =============================================================================

class SAM3MaskLoader:
    """
    Load masks from disk that were saved by streaming propagation.

    Use this node after SAM3Propagate with stream_to_disk=True to load
    masks for specific frame ranges. This allows processing very long videos
    by loading only the frames you need at a time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Path to mask directory from SAM3Propagate (stream_to_disk mode)"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "First frame to load (0 for beginning)"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Last frame to load (-1 for all remaining)"
                }),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Specific object ID to load (-1 for all objects)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "STRING")
    RETURN_NAMES = ("masks", "info")
    FUNCTION = "load_masks"
    CATEGORY = "SAM3/video"

    def load_masks(self, mask_dir, start_frame=0, end_frame=-1, obj_id=-1):
        """Load masks from disk into memory."""
        import os
        import glob
        import json

        if not mask_dir or not os.path.isdir(mask_dir):
            print(f"[SAM3 MaskLoader] Error: Invalid mask_dir: {mask_dir}")
            return ({}, json.dumps({"error": f"Invalid mask_dir: {mask_dir}"}))

        # Find all mask files
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npz")))

        if not mask_files:
            print(f"[SAM3 MaskLoader] Error: No .npz files found in {mask_dir}")
            return ({}, json.dumps({"error": "No mask files found"}))

        # Determine frame range from files
        all_frames = [int(os.path.basename(f).replace(".npz", "")) for f in mask_files]
        min_frame = min(all_frames)
        max_frame = max(all_frames)

        actual_start = start_frame if start_frame >= 0 else min_frame
        actual_end = end_frame if end_frame >= 0 else max_frame

        print(f"[SAM3 MaskLoader] Loading masks from {mask_dir}")
        print(f"[SAM3 MaskLoader] Frame range: {actual_start} to {actual_end} (available: {min_frame}-{max_frame})")
        if obj_id >= 0:
            print(f"[SAM3 MaskLoader] Filtering to object ID: {obj_id}")

        # Load masks
        masks_dict = {}
        loaded_count = 0
        skipped_count = 0

        for mask_file in mask_files:
            frame_idx = int(os.path.basename(mask_file).replace(".npz", ""))

            # Check if frame is in requested range
            if actual_start <= frame_idx <= actual_end:
                try:
                    data = np.load(mask_file)
                    mask = torch.from_numpy(data["mask"])

                    # Handle obj_id selection
                    if obj_id >= 0 and mask.dim() >= 3:
                        if obj_id < mask.shape[0]:
                            # Extract single object mask, keep dimension
                            mask = mask[obj_id:obj_id+1]
                        else:
                            # Object ID doesn't exist in this frame
                            skipped_count += 1
                            continue

                    masks_dict[frame_idx] = mask
                    loaded_count += 1

                except Exception as e:
                    print(f"[SAM3 MaskLoader] Warning: Failed to load {mask_file}: {e}")
                    skipped_count += 1

        print(f"[SAM3 MaskLoader] Loaded {loaded_count} masks, skipped {skipped_count}")

        info = {
            "loaded_frames": loaded_count,
            "skipped_frames": skipped_count,
            "frame_range": [actual_start, actual_end],
            "available_range": [min_frame, max_frame],
            "obj_id_filter": obj_id,
            "mask_dir": mask_dir
        }

        return (masks_dict, json.dumps(info, indent=2))


# =============================================================================
# Mask to Video Combiner (for streaming mode)
# =============================================================================

class SAM3MaskToVideo:
    """
    Combine streamed masks back into a visualization video.

    Processes masks one frame at a time from disk to avoid OOM.
    Use this node after SAM3Propagate with stream_to_disk=True to create
    colored overlay visualizations with optional ID labels.
    """

    # Color palette (matching SAM3VideoSegmenter)
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
                "mask_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Path to mask directory from SAM3Propagate (stream_to_disk mode)"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "First frame to process"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Last frame to process (-1 for all)"
                }),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Specific object to visualize (-1 for all)"
                }),
                "viz_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask overlay transparency"
                }),
                "show_ids": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show object ID labels at mask centroids"
                }),
                "label_size": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 72,
                    "tooltip": "Font size for ID labels"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("visualization", "combined_masks", "info")
    FUNCTION = "combine_video"
    CATEGORY = "SAM3/video"

    def _draw_id_label(self, frame_np, text, x, y, color, size=24):
        """Draw ID label on frame using OpenCV."""
        try:
            import cv2
        except ImportError:
            # OpenCV not available, skip drawing
            return frame_np

        # Calculate font parameters
        font_scale = size / 30.0
        thickness = max(1, int(size / 12))

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Center text on position
        text_x = max(0, min(x - text_w // 2, frame_np.shape[1] - text_w))
        text_y = max(text_h, min(y + text_h // 2, frame_np.shape[0]))

        # Draw background rectangle for readability
        cv2.rectangle(
            frame_np,
            (text_x - 2, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + baseline + 2),
            (0, 0, 0),
            -1
        )

        # Draw text
        color_255 = tuple(int(c * 255) for c in color)
        cv2.putText(
            frame_np, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_255, thickness
        )

        return frame_np

    def _visualize_frame(self, frame, mask, alpha, show_ids, label_size):
        """Create colored overlay for a single frame with optional ID labels."""
        vis_frame = frame.clone()
        id_positions = []  # Collect positions for ID labels

        if mask.dim() == 3:
            # Multi-object mask [num_objects, H, W]
            for obj_idx in range(mask.shape[0]):
                obj_mask = mask[obj_idx].float()
                if obj_mask.max() > 1.0:
                    obj_mask = obj_mask / 255.0

                color = torch.tensor(self.COLORS[obj_idx % len(self.COLORS)])
                mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                vis_frame = vis_frame * (1 - alpha * obj_mask.unsqueeze(-1)) + alpha * mask_rgb

                # Find centroid for ID label
                if show_ids and obj_mask.sum() > 100:
                    y_coords, x_coords = torch.where(obj_mask > 0.5)
                    if len(y_coords) > 0:
                        cy = int(y_coords.float().mean())
                        cx = int(x_coords.float().mean())
                        id_positions.append((obj_idx + 1, cx, cy, self.COLORS[obj_idx % len(self.COLORS)]))
        else:
            # Single mask [H, W]
            obj_mask = mask.float()
            if obj_mask.max() > 1.0:
                obj_mask = obj_mask / 255.0
            color = torch.tensor(self.COLORS[0])
            mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
            vis_frame = vis_frame * (1 - alpha * obj_mask.unsqueeze(-1)) + alpha * mask_rgb

            # Find centroid for ID label
            if show_ids and obj_mask.sum() > 100:
                y_coords, x_coords = torch.where(obj_mask > 0.5)
                if len(y_coords) > 0:
                    cy = int(y_coords.float().mean())
                    cx = int(x_coords.float().mean())
                    id_positions.append((1, cx, cy, self.COLORS[0]))

        vis_frame = vis_frame.clamp(0, 1)

        # Draw ID labels using OpenCV
        if show_ids and id_positions:
            # Convert to numpy for OpenCV
            frame_np = (vis_frame.cpu().numpy() * 255).astype(np.uint8)

            for obj_id, cx, cy, color in id_positions:
                frame_np = self._draw_id_label(
                    frame_np, f"ID:{obj_id}", cx, cy, color, label_size
                )

            vis_frame = torch.from_numpy(frame_np.astype(np.float32) / 255.0)

        return vis_frame

    def combine_video(self, video_frames, mask_dir, start_frame=0, end_frame=-1,
                      obj_id=-1, viz_alpha=0.5, show_ids=True, label_size=24):
        """
        Stream masks from disk and create visualization incrementally.
        """
        import os
        import glob
        import json

        if not mask_dir or not os.path.isdir(mask_dir):
            print(f"[SAM3 MaskToVideo] Error: Invalid mask_dir: {mask_dir}")
            return (
                video_frames,
                torch.zeros(video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]),
                json.dumps({"error": f"Invalid mask_dir: {mask_dir}"})
            )

        # Find mask files
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npz")))
        if not mask_files:
            print(f"[SAM3 MaskToVideo] Error: No .npz files found in {mask_dir}")
            return (
                video_frames,
                torch.zeros(video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]),
                json.dumps({"error": "No mask files found"})
            )

        # Determine frame range from mask files
        all_frame_indices = [int(os.path.basename(f).replace(".npz", "")) for f in mask_files]
        min_mask_frame = min(all_frame_indices)
        max_mask_frame = max(all_frame_indices)

        # Determine processing range
        num_video_frames = video_frames.shape[0]
        actual_start = max(0, start_frame)
        actual_end = min(num_video_frames - 1, end_frame if end_frame >= 0 else num_video_frames - 1)

        print(f"[SAM3 MaskToVideo] Processing frames {actual_start} to {actual_end}")
        print(f"[SAM3 MaskToVideo] Mask files available: {min_mask_frame} to {max_mask_frame}")
        if obj_id >= 0:
            print(f"[SAM3 MaskToVideo] Filtering to object ID: {obj_id}")
        print_vram("Before MaskToVideo")

        vis_list = []
        mask_list = []

        # Process one frame at a time (memory efficient)
        for frame_idx in range(actual_start, actual_end + 1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            frame = video_frames[frame_idx]
            h, w = frame.shape[:2]

            # Load mask for this frame if it exists
            mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.npz")

            if os.path.exists(mask_path):
                try:
                    data = np.load(mask_path)
                    mask = torch.from_numpy(data["mask"])

                    # Handle dimensions
                    if mask.dim() == 4:
                        mask = mask.squeeze(0)  # Remove batch dim

                    # Filter by obj_id if specified
                    if obj_id >= 0 and mask.dim() == 3:
                        if obj_id < mask.shape[0]:
                            mask = mask[obj_id:obj_id+1]
                        else:
                            mask = torch.zeros(1, h, w)

                    # Create visualization for this frame
                    vis_frame = self._visualize_frame(frame, mask, viz_alpha, show_ids, label_size)

                    # Create combined mask
                    if mask.dim() == 3:
                        combined = mask.max(dim=0)[0]
                    else:
                        combined = mask

                    # Ensure mask is 2D
                    if combined.dim() > 2:
                        combined = combined.squeeze()
                    if combined.dim() == 0:
                        combined = combined.unsqueeze(0).unsqueeze(0).expand(h, w)

                except Exception as e:
                    print(f"[SAM3 MaskToVideo] Warning: Failed to load {mask_path}: {e}")
                    vis_frame = frame.clone()
                    combined = torch.zeros(h, w)
            else:
                # No mask for this frame
                vis_frame = frame.clone()
                combined = torch.zeros(h, w)

            vis_list.append(vis_frame)
            mask_list.append(combined)

            # Periodic cleanup and progress
            if frame_idx % 50 == 0:
                gc.collect()
                print(f"[SAM3 MaskToVideo] Progress: frame {frame_idx}/{actual_end}")

        visualization = torch.stack(vis_list, dim=0)
        combined_masks = torch.stack(mask_list, dim=0)

        print(f"[SAM3 MaskToVideo] Complete: {len(vis_list)} frames processed")
        print_vram("After MaskToVideo")

        info = {
            "processed_frames": len(vis_list),
            "frame_range": [actual_start, actual_end],
            "mask_range": [min_mask_frame, max_mask_frame],
            "obj_id_filter": obj_id,
            "mask_dir": mask_dir,
            "show_ids": show_ids,
            "viz_alpha": viz_alpha
        }

        return (visualization, combined_masks, json.dumps(info, indent=2))


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3VideoSegmentation": SAM3VideoSegmentation,
    "SAM3AddPrompt": SAM3AddPrompt,
    "SAM3Propagate": SAM3Propagate,
    "SAM3VideoOutput": SAM3VideoOutput,
    "SAM3VideoTrim": SAM3VideoTrim,
    "SAM3VRAMEstimator": SAM3VRAMEstimator,
    "SAM3MaskLoader": SAM3MaskLoader,
    "SAM3MaskToVideo": SAM3MaskToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoSegmentation": "SAM3 Video Segmentation",
    "SAM3AddPrompt": "SAM3 Add Prompt",
    "SAM3Propagate": "SAM3 Propagate",
    "SAM3VideoOutput": "SAM3 Video Output",
    "SAM3VideoTrim": "SAM3 Video Trim",
    "SAM3VRAMEstimator": "SAM3 VRAM Estimator",
    "SAM3MaskLoader": "SAM3 Mask Loader",
    "SAM3MaskToVideo": "SAM3 Mask to Video",
}
