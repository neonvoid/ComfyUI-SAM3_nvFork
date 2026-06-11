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
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

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
    register_teardown_callback,
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
# Bounded session cache + signature helpers
# =============================================================================
# Replaces the four legacy class-level `_cache = {}` dicts that previously held
# GPU tensor refs across unrelated ComfyUI workflows. Three guarantees:
#
#   1. Bounded size — short cache lifetime; stale GPU tensors get released
#      in bounded time even without explicit eviction.
#   2. Session-scoped — when any SAM3 entrypoint sees a different session_uuid
#      than the previous one, all four caches flush. Session is the natural
#      ownership boundary for SAM3's C++ inference state.
#   3. Teardown-synchronized — caches register a flush callback with
#      inference_reconstructor; flushed BEFORE close_session() or
#      clear_inference_cache() runs so we never hold a Python ref to a tensor
#      whose C++ storage has been freed.
#
# Keys are session-scoped semantic tuples (no id() — id reuse after GC was
# the original Mode A footgun).

_SAM3_CACHE_REGISTRY = []  # list of _BoundedSessionCache instances
_CURRENT_SESSION_UUID = {"value": None}  # mutable single-value holder


def _short_key_repr(key, max_len=80):
    """Compact key repr for log lines."""
    s = repr(key)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


class _BoundedSessionCache:
    """Bounded LRU mapping with verbose [SAM3 Cache] logging.

    Drop-in replacement for the legacy `_cache = {}` class attributes. Supports
    `key in cache`, `cache[key]`, `cache[key] = value`, `cache.clear()`,
    `len(cache)`. Keys must be hashable.

    NOT thread-safe by design — ComfyUI executes the prompt graph on one
    worker thread; per-cache GIL ordering matches the legacy bare dict.
    """

    def __init__(self, name: str, max_size: int = 3):
        if max_size < 1:
            raise ValueError(f"_BoundedSessionCache: max_size must be >= 1, got {max_size}")
        self._name = name
        self._max_size = max_size
        self._store: "OrderedDict[Any, Any]" = OrderedDict()
        _SAM3_CACHE_REGISTRY.append(self)

    def __contains__(self, key) -> bool:
        return key in self._store

    def __getitem__(self, key):
        v = self._store[key]
        self._store.move_to_end(key)  # mark as most-recently-used
        return v

    def __setitem__(self, key, value) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        self._store[key] = value
        while len(self._store) > self._max_size:
            evicted_key, _evicted_value = self._store.popitem(last=False)
            print(f"[SAM3 Cache] {self._name} EVICT_LRU key={_short_key_repr(evicted_key)}")

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key, default=None):
        if key in self._store:
            return self.__getitem__(key)
        return default

    def clear(self) -> None:
        n = len(self._store)
        self._store.clear()
        if n > 0:
            print(f"[SAM3 Cache] {self._name} CLEAR n={n}")


def _flush_all_sam3_caches(reason: str) -> None:
    """Flush every registered _BoundedSessionCache. Fired by the teardown
    callback in inference_reconstructor BEFORE C++ inference state is freed."""
    counts = {c._name: len(c._store) for c in _SAM3_CACHE_REGISTRY}
    total = sum(counts.values())
    if total == 0:
        return
    for c in _SAM3_CACHE_REGISTRY:
        c._store.clear()
    print(f"[SAM3 Cache] FLUSH_ALL reason={reason} cleared={counts}")


# Register the flush callback. Keyed by a stable token so module reloads
# (importlib.reload) replace the stale callback rather than accumulating one
# per import — without the token, an old callback would pin the previous
# module's _SAM3_CACHE_REGISTRY and any tensors still in it.
register_teardown_callback(_flush_all_sam3_caches, name="sam3_video_nodes._flush_all_sam3_caches")


def _maybe_evict_on_session_change(new_session_uuid: Optional[str]) -> None:
    """If `new_session_uuid` differs from the previously-seen session UUID,
    flush all SAM3 caches. Call at the entrypoint of any SAM3 node that
    accepts a video_state. The session boundary is the natural ownership
    domain for cached results."""
    if not new_session_uuid:
        return
    cur = _CURRENT_SESSION_UUID["value"]
    if cur is None:
        _CURRENT_SESSION_UUID["value"] = new_session_uuid
        return
    if cur != new_session_uuid:
        print(f"[SAM3 Cache] SESSION_CHANGE old={cur[:8]} new={new_session_uuid[:8]}")
        _flush_all_sam3_caches("session_change")
        _CURRENT_SESSION_UUID["value"] = new_session_uuid


def _tensor_signature(t, n_samples: int = 256):
    """Structural signature for an IMAGE/MASK tensor.

    Returns a hashable tuple stable across reruns with identical content.
    Three discriminators stacked for layered protection:

      1. shape/dtype/device — structural fingerprint (catches resolution
         and type changes)
      2. mass — `t.float().sum()` over the whole tensor (catches localized
         pixel-level edits that a strided sample alone would miss; e.g. a
         100×100 patch on a 1024² mask has only ~1-in-400k chance of
         landing on any of 256 strided samples)
      3. sample_values — actual element values from a strided sample
         (catches spatial/layout changes that preserve total mass — e.g.
         shifting an object's position keeps sum constant but changes
         where the values are)

    Mass alone collides on letterboxing or mass-preserving permutations.
    Samples alone miss localized edits. Combined, the false-hit risk is
    negligible for any realistic SAM3 workflow input.

    SAFETY: only call on tensors known to be live (freshly produced by the
    current call OR provided as input by upstream nodes that just
    completed). Reading a stale CUDA tensor here would trigger an
    uncatchable access violation — but the teardown_callback architecture
    ensures cached entries are flushed before their underlying storage is
    invalidated, so the only tensors that ever reach this function are
    live."""
    if t is None:
        return ("none",)
    if not hasattr(t, "shape"):
        return ("type", type(t).__name__)
    shape = tuple(t.shape) if hasattr(t.shape, "__iter__") else (int(t.shape),)
    dtype = str(t.dtype) if hasattr(t, "dtype") else "?"
    device = str(t.device.type) if hasattr(t, "device") else "cpu"
    try:
        flat = t.reshape(-1)
        n = flat.numel()
        if n == 0:
            return (shape, dtype, device, 0.0, ())
        # Whole-tensor mass — catches localized edits the strided sample
        # would miss. Use `sum(dtype=...)` to:
        #   (a) upcast inside the reduction kernel's accumulator (no
        #       intermediate float32 copy of the full tensor — `.float()`
        #       on a fp16 video would silently allocate ~2× VRAM).
        #   (b) avoid catastrophic cancellation; the +1 from a tiny pixel
        #       edit would otherwise be absorbed and fail to flip the key.
        # MPS (Apple Silicon) doesn't support float64; fall back to float32
        # there. The minor cancellation risk on Mac is far better than the
        # except branch hiding all signatures behind a single error string.
        device_type = t.device.type if hasattr(t, "device") else "cpu"
        acc_dtype = torch.float32 if device_type == "mps" else torch.float64
        mass = round(float(flat.sum(dtype=acc_dtype).detach().cpu().item()), 4)
        # Strided sample for layout discrimination. Sample is at most
        # n_samples elements (~256) so the .float() copy here is ~1KB —
        # no VRAM concern.
        stride = max(1, n // n_samples)
        sample = flat[::stride][:n_samples].detach().to("cpu", copy=True)
        # Round per element so floating-point jitter from non-deterministic
        # ops doesn't flip cache keys for bit-identical reruns.
        sig_values = tuple(round(float(v), 4) for v in sample.float().tolist())
        return (shape, dtype, device, mass, sig_values)
    except Exception as e:
        return (shape, dtype, device, 0.0, f"err:{type(e).__name__}")


def _masks_signature(masks_dict, max_keys_in_sig: int = 5, n_frame_samples: int = 3):
    """Structural signature for a SAM3 propagation masks dict.

    Samples first/middle/last frames; combines tensor signatures with the
    set of frame indices. Hashable, stable across identical reruns."""
    if not masks_dict:
        return ("empty",)
    keys = sorted(masks_dict.keys())
    n = len(keys)
    if n == 0:
        return ("empty",)
    sample_idxs = [keys[0]]
    if n >= 3:
        sample_idxs.append(keys[n // 2])
    if n >= 2 and keys[-1] != sample_idxs[-1]:
        sample_idxs.append(keys[-1])
    parts = []
    for idx in sample_idxs:
        v = masks_dict[idx]
        if isinstance(v, dict):
            t = v.get("mask")
            obj_ids = tuple(v.get("obj_ids", ()) or ())
        else:
            t = v
            obj_ids = ()
        # 64 samples per frame — enough to distinguish mostly-empty masks
        # from each other (8 samples could all land on zero pixels).
        parts.append((idx, _tensor_signature(t, n_samples=64), obj_ids))
    head_keys = tuple(keys[:max_keys_in_sig])
    return (n, head_keys, tuple(parts))


def _scalar_or_str(x):
    """Hashable representation of arbitrary widget input (point/box prompt
    dicts, etc.). NOT truncated — earlier versions truncated at 128 chars but
    real point prompts with 7+ coordinates exceeded that budget, hiding
    trailing edits and producing false cache hits."""
    if x is None:
        return None
    if isinstance(x, (int, float, bool, str)):
        return x
    return repr(x)


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

    Prompt mode is inferred from what you provide:
    - text_prompt non-empty → track objects by text description
    - positive/negative points connected → track objects by points
    - positive/negative boxes connected → track objects by boxes (combinable
      with text OR points as region hints)

    Text and points are mutually exclusive — providing both raises an error.
    Boxes can combine with either as region hints.
    """
    # Bounded session-scoped cache (replaces legacy `_cache = {}`).
    # See _BoundedSessionCache docstring for crash-safety rationale.
    _cache = _BoundedSessionCache("VideoSegmentation", max_size=3)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch of images [N, H, W, C]"
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text description(s) to track. Comma-separated for multiple objects (e.g., 'person, dog, car'). Mutually exclusive with points."
                }),
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Positive points — click on objects to track. Mutually exclusive with text_prompt."
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Negative points — click on areas to exclude. Mutually exclusive with text_prompt."
                }),
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Positive boxes — region hints, combinable with text or points."
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Negative boxes — region hints, combinable with text or points."
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
                # === Advanced: Hotstart & Detection Tuning ===
                # These parameters control how new objects are detected and validated
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

    @classmethod
    def IS_CHANGED(cls, video_frames, text_prompt="",
                   positive_points=None, negative_points=None,
                   positive_boxes=None, negative_boxes=None,
                   frame_idx=0, score_threshold=0.3,
                   offload_video_to_cpu=True, offload_state_to_cpu=False,
                   hotstart_delay=15, hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
                   new_det_thresh=0.4, assoc_iou_thresh=0.1):
        import hashlib

        # Guard: IS_CHANGED is called during cache checking before upstream
        # nodes execute, so video_frames may be None
        if video_frames is None:
            return float("NaN")

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

        return hash((
            video_hash,
            text_prompt,
            str(positive_points),
            str(negative_points),
            str(positive_boxes),
            str(negative_boxes),
            frame_idx,
            score_threshold,
            offload_video_to_cpu,
            offload_state_to_cpu,
            hotstart_delay,
            hotstart_unmatch_thresh,
            hotstart_dup_thresh,
            new_det_thresh,
            assoc_iou_thresh,
        ))

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "segment"
    CATEGORY = "SAM3/video"

    def segment(self, video_frames, text_prompt="",
                positive_points=None, negative_points=None,
                positive_boxes=None, negative_boxes=None,
                frame_idx=0, score_threshold=0.3,
                offload_video_to_cpu=True, offload_state_to_cpu=False,
                hotstart_delay=15, hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
                new_det_thresh=0.4, assoc_iou_thresh=0.1):
        """Initialize video state and add prompts based on what the user provided.

        Mode inference:
          - text_prompt non-empty       → text mode
          - positive/negative points    → point mode
          - positive/negative boxes     → always processed as region hints
        Text and points are mutually exclusive — raises ValueError if both given.
        """
        # Auto-detect which prompts are actually present
        has_text = bool(text_prompt and text_prompt.strip())
        has_points = bool(
            (positive_points and (positive_points.get("points") or positive_points.get("objects")))
            or (negative_points and negative_points.get("points"))
        )
        has_boxes = bool(
            (positive_boxes and positive_boxes.get("boxes"))
            or (negative_boxes and negative_boxes.get("boxes"))
        )

        if has_text and has_points:
            raise ValueError(
                "[SAM3 Video] Both text_prompt and points provided — these are "
                "mutually exclusive. Provide ONE of: text_prompt, or points "
                "(boxes can combine with either)."
            )
        if not (has_text or has_points or has_boxes):
            raise ValueError(
                "[SAM3 Video] No prompts provided — enter text_prompt, or "
                "connect positive/negative points or boxes."
            )

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
        h.update(text_prompt.encode())
        # Content-derived prompt fingerprints (not id() — Python memory addresses
        # get reused after GC, producing false cache hits on stale tensors).
        h.update((repr(positive_points) if positive_points else "none").encode())
        h.update((repr(negative_points) if negative_points else "none").encode())
        h.update((repr(positive_boxes) if positive_boxes else "none").encode())
        h.update((repr(negative_boxes) if negative_boxes else "none").encode())
        h.update(str(frame_idx).encode())
        h.update(str(score_threshold).encode())
        h.update(str(offload_video_to_cpu).encode())
        h.update(str(offload_state_to_cpu).encode())
        h.update(str(hotstart_delay).encode())
        h.update(str(hotstart_unmatch_thresh).encode())
        h.update(str(hotstart_dup_thresh).encode())
        h.update(str(new_det_thresh).encode())
        h.update(str(assoc_iou_thresh).encode())
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
            # Hotstart & detection tuning
            hotstart_delay=hotstart_delay,
            hotstart_unmatch_thresh=hotstart_unmatch_thresh,
            hotstart_dup_thresh=hotstart_dup_thresh,
            new_det_thresh=new_det_thresh,
            assoc_iou_thresh=assoc_iou_thresh,
        )
        video_state = create_video_state(
            video_frames=video_frames,
            config=config,
        )

        print(f"[SAM3 Video] Initialized session {video_state.session_uuid[:8]}")
        print(f"[SAM3 Video] Frames: {video_state.num_frames}, Size: {video_state.width}x{video_state.height}")
        mode_tags = []
        if has_text: mode_tags.append("text")
        if has_points: mode_tags.append("points")
        if has_boxes: mode_tags.append("boxes")
        print(f"[SAM3 Video] Detected prompts: {' + '.join(mode_tags)}")
        print(f"[SAM3 Video] Hotstart config: delay={hotstart_delay}, unmatch_thresh={hotstart_unmatch_thresh}, dup_thresh={hotstart_dup_thresh}")
        print(f"[SAM3 Video] Detection config: new_det_thresh={new_det_thresh}, assoc_iou_thresh={assoc_iou_thresh}")

        # 2. Add prompts. Text and points are mutually exclusive (checked above);
        #    boxes are always additive region hints.
        obj_id = 1

        if has_text:
            for text in text_prompt.split(","):
                text = text.strip()
                if text:
                    prompt = VideoPrompt.create_text(frame_idx, obj_id, text)
                    video_state = video_state.with_prompt(prompt)
                    print(f"[SAM3 Video] Added text prompt: obj={obj_id}, text='{text}'")
                    obj_id += 1

        elif has_points:
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

                    # Optional per-object box ([cx, cy, w, h] normalized, e.g. from
                    # NV_SAM3SeedBuilder.init_points_prompt). Convert to xyxy corners
                    # (VideoPrompt.create_box contract) and chain it -- SAM3 merges
                    # box+points for the same (frame, obj) into one stronger prompt.
                    box_raw = obj_data.get("box")
                    if box_raw and len(box_raw) == 4 and all_points:
                        try:
                            bcx, bcy, bw, bh = (float(v) for v in box_raw)
                            if bw > 0 and bh > 0:
                                bx1 = max(0.0, min(1.0, bcx - bw / 2.0))
                                by1 = max(0.0, min(1.0, bcy - bh / 2.0))
                                bx2 = max(0.0, min(1.0, bcx + bw / 2.0))
                                by2 = max(0.0, min(1.0, bcy + bh / 2.0))
                                if bx2 > bx1 and by2 > by1:
                                    box_prompt = VideoPrompt.create_box(
                                        frame_idx, obj_id, [bx1, by1, bx2, by2], is_positive=True
                                    )
                                    video_state = video_state.with_prompt(box_prompt)
                                    print(f"[SAM3 Video] Added box prompt: obj={obj_id} "
                                          f"(merged with points at apply)")
                        except (TypeError, ValueError) as e:
                            print(f"[SAM3 Video] WARN: skipping malformed box for "
                                  f"obj={obj_id}: {box_raw!r} ({e})")

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
                    print("[SAM3 Video] Warning: positive_points/negative_points connected but contained no points")

        # 3. Always process box inputs (can be combined with any mode)
        # Boxes act as region hints/constraints in addition to text or point prompts
        has_boxes = False

        if positive_boxes and positive_boxes.get("boxes"):
            for box_data in positive_boxes["boxes"]:
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
            for box_data in negative_boxes["boxes"]:
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

        # Validate at least one prompt was added
        if len(video_state.prompts) == 0:
            print(f"[SAM3 Video] Warning: No prompts added (inputs were detected but contained no usable data)")

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
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Negative boxes - draw around areas to exclude from tracking"
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
                   positive_boxes=None, negative_boxes=None, obj_id=-1):
        """
        Add prompts on an additional frame to reinforce tracking.

        Args:
            video_state: Existing video state with prompts
            frame_idx: Frame index to add new prompts
            positive_points: Points to add (from points editor)
            negative_points: Negative points to exclude
            positive_boxes: Boxes to add
            negative_boxes: Boxes for areas to exclude
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

        # Handle box prompts (convert from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2])
        if positive_boxes and positive_boxes.get("boxes"):
            for box_data in positive_boxes["boxes"]:
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=True)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 AddPrompt] Added positive box: frame={frame_idx}, obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                prompts_added += 1

        if negative_boxes and negative_boxes.get("boxes"):
            for box_data in negative_boxes["boxes"]:
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=False)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 AddPrompt] Added negative box: frame={frame_idx}, obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                prompts_added += 1

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
    # Bounded session-scoped cache (replaces legacy `_cache = {}`).
    # max_size=2: propagation results carry the heaviest payload (per-frame
    # GPU mask tensors), so keep the LRU tight to bound stale-tensor exposure.
    _cache = _BoundedSessionCache("Propagate", max_size=2)

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
                    "tooltip": "End frame for forward propagation (-1 for all)"
                }),
                "backward_stop_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Stop backward propagation at this frame (-1 to go all the way to frame 0). Only used when direction is 'backward' or 'both'."
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
    def IS_CHANGED(cls, sam3_model, video_state, start_frame=0, end_frame=-1, backward_stop_frame=-1,
                   direction="forward", offload_model=False, enable_chunking=False, chunk_size=250,
                   range_detection_only=False, stream_to_disk=False, mask_output_path="",
                   auto_exit_on_empty=False, exit_delay_seconds=0.5, video_fps=30.0, lock_initial_objects=True):
        # Use object identity for caching - if upstream node is cached,
        # it returns the same object, so id() will match
        # This is more reliable than hashing content since video_state is immutable
        if video_state is None:
            return float("NaN")
        return (id(video_state), start_frame, end_frame, backward_stop_frame, direction, enable_chunking, chunk_size, range_detection_only, stream_to_disk, mask_output_path, auto_exit_on_empty, exit_delay_seconds, video_fps, lock_initial_objects)

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

    def propagate(self, sam3_model, video_state, start_frame=0, end_frame=-1, backward_stop_frame=-1,
                  direction="forward", offload_model=False, enable_chunking=False, chunk_size=250,
                  range_detection_only=False, stream_to_disk=False, mask_output_path="",
                  auto_exit_on_empty=False, exit_delay_seconds=0.5, video_fps=30.0, lock_initial_objects=True):
        """Run propagation using reconstructed inference state."""
        # Handle None values from ComfyUI (can happen with new parameters on old workflows)
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else -1
        backward_stop_frame = backward_stop_frame if backward_stop_frame is not None else -1

        # Session-scoped semantic cache key (no id() — see _BoundedSessionCache).
        # video_state is a frozen dataclass; session_uuid + prompt count + a
        # repr() of the prompt tuple uniquely identifies the upstream
        # segmentation result. We use repr() instead of hash() because prompts
        # deserialized from workflow JSON (via SAM3VideoState.from_dict) may
        # contain nested lists, which would crash hash() with TypeError:
        # unhashable type: 'list'.
        _maybe_evict_on_session_change(video_state.session_uuid)
        cache_key = (
            "propagate",
            video_state.session_uuid,
            len(video_state.prompts),
            repr(video_state.prompts),
            start_frame, end_frame, backward_stop_frame,
            direction, enable_chunking, chunk_size, range_detection_only,
            stream_to_disk, mask_output_path, auto_exit_on_empty,
            exit_delay_seconds, video_fps, lock_initial_objects,
        )

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
            backward_info = f", backward_stop={backward_stop_frame}" if direction in ["backward", "both"] else ""
            print(f"[SAM3 Video] Starting propagation: frames {start_frame} to {actual_end_frame}, direction={direction}{backward_info}")
            print(f"[SAM3 Video] Prompts: {len(video_state.prompts)}")
            print_vram("Before propagation start")

            # Build propagation request - uses predictor's handle_stream_request API
            # backward_stop_frame: -1 means go to frame 0, otherwise stop at specified frame
            actual_backward_stop = backward_stop_frame if backward_stop_frame >= 0 else 0
            request = {
                "type": "propagate_in_video",
                "session_id": video_state.session_uuid,
                "propagation_direction": direction,
                "start_frame_index": start_frame,
                "max_frame_num_to_track": actual_end_frame - start_frame + 1,
                "backward_stop_frame": actual_backward_stop,
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

            # Multiplex obj_ids backfill (SAM 3.1): the multiplex SAM2-merge
            # propagation path emits obj_ids metadata only on PROMPT frames; pure
            # propagation frames arrive with masks but no obj_ids, so MaskTracks
            # (correctly) skips them and only the prompt frames survive -> masks
            # appear on a handful of frames only. The multiplex demux preserves
            # channel order across frames, so the last frame that DID carry obj_ids
            # gives the canonical channel->obj_id map; carry it forward onto the
            # bare frames. Guarded by an exact mask-channel-count match so a changed
            # object count (entered/left) is left unset rather than mis-mapped.
            last_known_obj_ids = None
            backfilled_obj_id_frames = 0
            backfill_debug_logged = 0  # DIAGNOSTIC: cap bare-frame shape logs

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

                        # DIAGNOSTIC: log the raw per-frame output keys (+ obj_id_to_mask
                        # presence) on the first few frames so we see exactly which key
                        # carries obj_ids on bare propagation frames. The multiplex
                        # _postprocess_output builds out_obj_ids + out_binary_masks from
                        # obj_id_to_mask; this confirms what actually reaches the wrapper.
                        if backfill_debug_logged < 4 and isinstance(outputs, dict):
                            _o2m = outputs.get("obj_id_to_mask")
                            print(
                                f"[SAM3 Video][bf-debug] frame={frame_idx} "
                                f"outputs.keys={sorted(outputs.keys())} "
                                f"obj_id_to_mask={'dict len='+str(len(_o2m)) if isinstance(_o2m, dict) else type(_o2m).__name__} "
                                f"out_obj_ids={type(outputs.get('out_obj_ids')).__name__}/"
                                f"{outputs.get('out_obj_ids') if not hasattr(outputs.get('out_obj_ids'), 'shape') else getattr(outputs.get('out_obj_ids'), 'shape', None)}"
                            )
                            backfill_debug_logged += 1

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

                            # Capture object IDs for stable color mapping.
                            # Key tolerance: legacy fork wrapper emits "obj_ids";
                            # upstream SAM 3.1 predictors emit "out_obj_ids".
                            frame_obj_ids = None
                            obj_ids_key = None
                            for key in ["obj_ids", "out_obj_ids"]:
                                if key in outputs and outputs[key] is not None:
                                    obj_ids_key = key
                                    break
                            if obj_ids_key is not None:
                                ids = outputs[obj_ids_key]
                                if hasattr(ids, 'tolist'):
                                    ids = ids.tolist()
                                elif isinstance(ids, np.ndarray):
                                    ids = ids.tolist()
                                frame_obj_ids = ids
                                obj_ids_dict[frame_idx] = ids
                                del outputs[obj_ids_key]

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

                            # Multiplex obj_ids backfill (see init above). Update the
                            # canonical map from any frame that carries obj_ids; carry
                            # it onto bare frames whose mask channel count matches.
                            if frame_obj_ids:
                                last_known_obj_ids = list(frame_obj_ids)
                            elif last_known_obj_ids is not None and mask is not None and hasattr(mask, "shape"):
                                _shape = tuple(mask.shape)
                                _nd = len(_shape)
                                if _nd == 3:
                                    _nch = _shape[0]
                                elif _nd == 4:
                                    _nch = _shape[1] if _shape[0] == 1 else _shape[0]
                                elif _nd == 2:
                                    _nch = 1
                                else:
                                    _nch = 0
                                # DIAGNOSTIC: surface the bare-frame mask structure on
                                # the first few occurrences so we can see why the guard
                                # matches or not (mask may be merged/single-channel, or
                                # padded to a fixed slot count). Remove once resolved.
                                if backfill_debug_logged < 3:
                                    print(
                                        f"[SAM3 Video][bf-debug] frame={frame_idx} "
                                        f"bare-mask shape={_shape} type={type(mask).__name__} "
                                        f"n_ch={_nch} last_known_obj_ids={last_known_obj_ids} "
                                        f"(match={_nch == len(last_known_obj_ids)})"
                                    )
                                    backfill_debug_logged += 1
                                if _nch == len(last_known_obj_ids):
                                    frame_obj_ids = list(last_known_obj_ids)
                                    obj_ids_dict[frame_idx] = frame_obj_ids
                                    backfilled_obj_id_frames += 1

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
            if backfilled_obj_id_frames:
                print(
                    f"[SAM3 Video] Backfilled obj_ids onto {backfilled_obj_id_frames} "
                    f"multiplex propagation frame(s) (channel-count matched) so "
                    f"MaskTracks keeps them instead of skipping."
                )

            # Handle early exit: truncate output and create metadata
            if auto_exit_on_empty:
                if early_exit_triggered and direction == "forward":
                    # Only truncate for forward-only propagation
                    # For bidirectional, early exit on forward shouldn't remove backward frames
                    frames_to_remove = [f for f in masks_dict.keys() if f > last_valid_frame]
                    for f in frames_to_remove:
                        del masks_dict[f]

                    # Truncate scores_dict similarly
                    frames_to_remove = [f for f in scores_dict.keys() if f > last_valid_frame]
                    for f in frames_to_remove:
                        del scores_dict[f]

                    print(f"[SAM3 Video] Truncated output to {len(masks_dict)} valid frames (0-{last_valid_frame})")

                # Create track_info with early exit metadata
                # For bidirectional propagation, use masks_dict keys to determine actual frame range
                # (last_valid_frame variable gets overwritten by backward pass, so use max of masks_dict)
                actual_start_frame = min(masks_dict.keys()) if masks_dict else start_frame
                actual_last_frame = max(masks_dict.keys()) if masks_dict else actual_end_frame
                track_info = {
                    "early_exit_enabled": True,
                    "early_exit_triggered": early_exit_triggered,
                    "exit_delay_seconds": exit_delay_seconds,
                    "video_fps": video_fps,
                    "empty_frames_threshold": empty_frames_threshold,
                    "last_valid_frame": actual_last_frame,  # Use max of masks_dict, not the variable
                    "total_valid_frames": len(masks_dict),
                    "original_end_frame": actual_end_frame,
                    "start_frame": actual_start_frame,
                }
                track_info_json = json.dumps(track_info, indent=2)
            else:
                # Default track_info when auto_exit is disabled
                # For bidirectional propagation, compute actual frame range from masks_dict
                actual_start_frame = min(masks_dict.keys()) if masks_dict else start_frame
                actual_last_frame = max(masks_dict.keys()) if masks_dict else actual_end_frame
                track_info = {
                    "early_exit_enabled": False,
                    "early_exit_triggered": False,
                    "last_valid_frame": actual_last_frame,
                    "total_valid_frames": len(masks_dict),
                    "original_end_frame": actual_end_frame,
                    "start_frame": actual_start_frame,
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
    # Bounded session-scoped cache (replaces legacy `_cache = {}`).
    _cache = _BoundedSessionCache("VideoOutput", max_size=3)

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
                "mask_colors": ("STRING", {
                    "default": "",
                    "tooltip": "Custom mask colors (comma-separated). Names: red, blue, green, yellow, magenta, cyan, orange, purple, pink, lime, teal, coral, gold, navy. Or hex: #FF0000. Order matches object IDs. Empty = default colors."
                }),
                "previous_visualization": ("IMAGE", {
                    "tooltip": "Previous debug video to overlay new masks onto (for accumulating tracked players across runs)"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, masks, video_state, scores=None, obj_ids=None, obj_id=-1, plot_all_masks=True, draw_legend=True, mask_colors="", previous_visualization=None):
        if masks is None or video_state is None:
            return float("NaN")
        return (id(masks), video_state.session_uuid, id(scores), id(obj_ids), obj_id, plot_all_masks, draw_legend, mask_colors, id(previous_visualization))

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "frames", "visualization", "accumulated_visualization")
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

    def extract(self, masks, video_state, scores=None, obj_ids=None, obj_id=-1, plot_all_masks=True, draw_legend=True, mask_colors="", previous_visualization=None):
        """Extract all masks as a batch [N, H, W]."""
        from PIL import Image
        import os
        from .utils import get_color_palette, DEFAULT_COLORS

        # Session-scoped semantic cache key (no id() — see _BoundedSessionCache).
        # masks/scores/obj_ids come from upstream SAM3Propagate; their structural
        # signatures discriminate different propagation results without using
        # volatile Python memory addresses. Reading these signatures is safe here
        # because the inputs were just produced by the upstream node call (live).
        # NOTE 1: do NOT slice obj_ids — truncating at N entries means edits to
        # objects past index N silently false-hit the cache.
        # NOTE 2: include repr(video_state.prompts) explicitly. _masks_signature
        # only samples 3 frames (start/middle/end); a prompt edit affecting only
        # middle frames could leave those 3 sampled signatures unchanged →
        # false hit. Threading the prompt repr through forces invalidation on
        # any prompt change regardless of which frames it affects.
        _maybe_evict_on_session_change(video_state.session_uuid)
        cache_key = (
            "videoOutput",
            video_state.session_uuid,
            len(video_state.prompts),
            repr(video_state.prompts),
            _masks_signature(masks),
            _masks_signature(scores) if scores else None,
            tuple(sorted(obj_ids.keys())) if obj_ids else None,
            obj_id, plot_all_masks, draw_legend, mask_colors,
            _tensor_signature(previous_visualization) if previous_visualization is not None else None,
        )

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
            return (empty_mask, empty_frames, empty_frames, empty_frames)

        # Process all frames in order
        mask_list = []
        frame_list = []
        vis_list = []
        accumulated_vis_list = []

        # Get color palette - user colors first, then defaults
        # Estimate max objects from first available mask
        max_objects = 8  # Default estimate
        for frame_idx in masks:
            mask_data = masks[frame_idx]
            if isinstance(mask_data, dict):
                m = mask_data.get("mask")
            else:
                m = mask_data
            if m is not None and hasattr(m, 'shape') and len(m.shape) >= 3:
                max_objects = max(max_objects, m.shape[0] if len(m.shape) == 3 else 1)
                break
        colors = get_color_palette(mask_colors, max_objects)
        if mask_colors:
            print(f"[SAM3 Video Output] Using custom colors: {mask_colors}")

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

                # Create accumulated visualization (overlay on previous or original)
                if previous_visualization is not None and frame_idx < previous_visualization.shape[0]:
                    accumulated_vis_frame = previous_visualization[frame_idx].clone()
                else:
                    accumulated_vis_frame = img_tensor.clone()

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
                            accumulated_vis_frame = accumulated_vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
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
                        accumulated_vis_frame = accumulated_vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
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
                    accumulated_vis_frame = accumulated_vis_frame * (1 - 0.5 * frame_mask.unsqueeze(-1)) + 0.5 * mask_rgb

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
                    accumulated_vis_frame = self._draw_legend(accumulated_vis_frame, num_objects, colors, obj_id=legend_obj_id, frame_scores=frame_scores)

                vis_list.append(vis_frame.clamp(0, 1))
                accumulated_vis_list.append(accumulated_vis_frame.clamp(0, 1))
            else:
                # No mask for this frame - use zeros
                frame_mask = torch.zeros(h, w)
                vis_list.append(img_tensor)
                # For accumulated, use previous visualization if available, else original
                if previous_visualization is not None and frame_idx < previous_visualization.shape[0]:
                    accumulated_vis_list.append(previous_visualization[frame_idx])
                else:
                    accumulated_vis_list.append(img_tensor)

            mask_list.append(frame_mask.cpu())

        # Stack into batches
        all_masks = torch.stack(mask_list, dim=0)  # [N, H, W]
        all_frames = torch.stack(frame_list, dim=0)  # [N, H, W, C]
        all_vis = torch.stack(vis_list, dim=0)  # [N, H, W, C]
        all_accumulated_vis = torch.stack(accumulated_vis_list, dim=0)  # [N, H, W, C]

        print(f"[SAM3 Video] Output: {all_masks.shape[0]} masks, shape {all_masks.shape}")
        print(f"[SAM3 Video] Objects tracked: {num_objects}, plot_all_masks: {plot_all_masks}")
        if previous_visualization is not None:
            print(f"[SAM3 Video] Accumulated visualization on top of previous ({previous_visualization.shape[0]} frames)")
        print_vram("After extract")

        # Cache the result
        result = (all_masks, all_frames, all_vis, all_accumulated_vis)
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
                "mask_colors": ("STRING", {
                    "default": "",
                    "tooltip": "Custom mask colors (comma-separated). Names: red, blue, green, yellow, magenta, cyan, orange, purple, pink, lime, teal, coral, gold, navy. Or hex: #FF0000. Order matches object IDs. Empty = default colors."
                }),
                "previous_visualization": ("IMAGE", {
                    "tooltip": "Previous debug video to overlay new masks onto (for accumulating tracked players across runs)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("trimmed_frames", "masks", "visualization", "accumulated_visualization", "start_frame", "end_frame", "total_frames")
    FUNCTION = "trim_video"
    CATEGORY = "SAM3/video"

    def trim_video(self, video_frames, masks, track_info, scores=None, obj_ids=None, viz_alpha=0.5, mask_colors="", previous_visualization=None):
        """
        Trim video frames and extract masks/visualization for valid frame range.

        Args:
            video_frames: Original video frames [N, H, W, C]
            masks: Dict of frame_idx -> mask tensor from SAM3Propagate
            track_info: JSON string with early exit metadata
            scores: Optional confidence scores dict
            viz_alpha: Transparency for mask overlay
            previous_visualization: Optional previous debug video to overlay new masks onto

        Returns:
            trimmed_frames: Sliced video frames
            masks: Combined mask tensor [N, H, W]
            visualization: Colored overlay frames [N, H, W, C]
            accumulated_visualization: Accumulated overlay (on previous_visualization if provided)
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
            return (video_frames, empty_mask, video_frames.clone(), video_frames.clone(), 0, video_frames.shape[0] - 1, video_frames.shape[0])

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

        # Get color palette for visualization
        from .utils import get_color_palette, DEFAULT_COLORS
        # Estimate max objects from masks
        max_objects = 1
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in masks:
                mask_data = masks[frame_idx]
                if isinstance(mask_data, dict):
                    m = mask_data.get("mask")
                else:
                    m = mask_data
                if isinstance(m, (torch.Tensor, np.ndarray)) and len(m.shape) >= 3:
                    max_objects = max(max_objects, m.shape[0])
        colors = get_color_palette(mask_colors, max_objects)

        if early_exit_triggered:
            print(f"[SAM3 VideoTrim] Early exit detected - trimming video")
            print(f"[SAM3 VideoTrim] Original: {num_input_frames} frames")
            print(f"[SAM3 VideoTrim] Trimmed: frames {start_frame}-{end_frame} ({total_frames} frames)")
        else:
            print(f"[SAM3 VideoTrim] No early exit - frames {start_frame}-{end_frame} ({total_frames} frames)")

        # Extract and process masks for valid frame range
        mask_list = []
        vis_list = []
        accumulated_vis_list = []

        for out_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            # Get frame from trimmed video
            frame_tensor = trimmed[out_idx]
            vis_frame = frame_tensor.clone()

            # Create accumulated visualization (overlay on previous or original)
            if previous_visualization is not None and out_idx < previous_visualization.shape[0]:
                accumulated_vis_frame = previous_visualization[out_idx].clone()
            else:
                accumulated_vis_frame = frame_tensor.clone()

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
                        color = torch.tensor(colors[color_id % len(colors)])

                        # Add colored overlay to visualization
                        mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - viz_alpha * obj_mask.unsqueeze(-1)) + viz_alpha * mask_rgb
                        accumulated_vis_frame = accumulated_vis_frame * (1 - viz_alpha * obj_mask.unsqueeze(-1)) + viz_alpha * mask_rgb
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
                    color = torch.tensor(colors[color_id % len(colors)])

                    # Add colored overlay
                    mask_rgb = combined_mask.unsqueeze(-1) * color.view(1, 1, 3)
                    vis_frame = vis_frame * (1 - viz_alpha * combined_mask.unsqueeze(-1)) + viz_alpha * mask_rgb
                    accumulated_vis_frame = accumulated_vis_frame * (1 - viz_alpha * combined_mask.unsqueeze(-1)) + viz_alpha * mask_rgb
            else:
                # No mask for this frame
                combined_mask = torch.zeros(h, w)

            mask_list.append(combined_mask.cpu())
            vis_list.append(vis_frame.clamp(0, 1))
            accumulated_vis_list.append(accumulated_vis_frame.clamp(0, 1))

        # Stack into batches
        all_masks = torch.stack(mask_list, dim=0)  # [N, H, W]
        all_vis = torch.stack(vis_list, dim=0)  # [N, H, W, C]
        all_accumulated_vis = torch.stack(accumulated_vis_list, dim=0)  # [N, H, W, C]

        print(f"[SAM3 VideoTrim] Output: {total_frames} frames, masks shape {all_masks.shape}")
        if previous_visualization is not None:
            print(f"[SAM3 VideoTrim] Accumulated visualization on top of previous ({previous_visualization.shape[0]} frames)")

        return (trimmed, all_masks, all_vis, all_accumulated_vis, start_frame, end_frame, total_frames)


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
# Mask Refinement
# =============================================================================

# =============================================================================
# SAM3MaskRefine compose helpers (2026-05-20 audit Bug #2 — corrective compose).
#
# The SAM3 tracker enforces mutual exclusion between mask + point/box prompts
# at the same (frame_idx, obj_id): `add_new_mask` clears point_inputs_per_frame
# and `add_new_points_or_box` clears mask_inputs_per_frame
# (sam3_lib/model/sam3_tracking_predictor.py:266,406). They CANNOT coexist as
# separate inputs.
#
# To preserve both inputs' intent without violating the tracker contract, we
# pre-compose a SINGLE mask at the corrective frame:
#   1. Run image-mode SAM3 segmentation on that frame with the user's
#      points/box → "corrective_mask" (SAM3's interpretation of the prompts)
#   2. Combine with the user's input_mask using a chosen op (union/intersect/
#      replace/diff). Default `auto` picks op from prompt polarity.
#   3. Feed the composed mask via add_new_mask as a single prompt.
#
# This sidesteps the tracker-level mutual-exclusion entirely. v1 scope is
# single-subject only (user confirmed 2026-05-20) — multi-object compose is a
# v2 follow-up.
# =============================================================================

_VALID_COMPOSE_MODES = ("skip", "auto", "union", "intersect", "replace", "diff")


def _validate_compose_mode(mode: str) -> None:
    """Raise ValueError if `mode` is not a recognized corrective_compose_mode.

    Factored out so the validation is directly unit-testable without hitting
    any downstream propagation work (R1 review follow-up — Codex).
    """
    if mode not in _VALID_COMPOSE_MODES:
        raise ValueError(
            f"[SAM3 MaskRefine] corrective_compose_mode='{mode}' "
            f"not in {_VALID_COMPOSE_MODES}"
        )


def _resolve_compose_op(
    mode: str,
    has_positive: bool,
    has_negative: bool,
) -> str:
    """Auto-pick op from prompt polarity, or honor explicit override.

    Modes:
      auto      — positive-only → union; negative-only → diff;
                  mixed (both polarities) → replace (trust image-mode result)
      union     — force mask_in OR corrective
      intersect — force mask_in AND corrective (conservative)
      replace   — force corrective only (trust image-mode entirely)
      diff      — force mask_in AND NOT corrective (subtract)
    """
    if mode != "auto":
        return mode
    if has_negative and not has_positive:
        return "diff"
    if has_positive and has_negative:
        return "replace"
    return "union"  # positive-only or no-op


def _compose_masks(base_mask: torch.Tensor, corrective_mask: torch.Tensor, op: str) -> torch.Tensor:
    """Element-wise compose two [H, W] float masks in [0, 1] range.

    Both tensors must be same shape. Caller responsible for shape match.
    """
    if op == "union":
        return torch.maximum(base_mask, corrective_mask)
    if op == "intersect":
        return torch.minimum(base_mask, corrective_mask)
    if op == "replace":
        return corrective_mask.clone()
    if op == "diff":
        return base_mask * (1.0 - corrective_mask)
    raise ValueError(
        f"[SAM3 MaskRefine] Unknown compose op '{op}'. "
        f"Expected one of: auto, union, intersect, replace, diff."
    )


def _run_image_mode_segmentation(
    sam3_model,
    frame_tensor: torch.Tensor,
    positive_points_dict,
    negative_points_dict,
    positive_boxes_dict,
    negative_boxes_dict,
) -> Optional[torch.Tensor]:
    """Run image-mode SAM3 segmentation on a single frame to materialize a mask
    from points/box prompts. Returns a [H, W] float mask in [0, 1] range, or
    None if no usable prompts.

    Uses the same processor.set_image + model.predict_inst path as
    SAM3Segmentation in segmentation.py. Single-subject only — multi-object
    `objects` dict format on positive_points is NOT supported in this path
    (user-confirmed v1 scope 2026-05-20: refine input is identity-isolated).
    """
    from .utils import comfy_image_to_pil

    # Convert tensor [H, W, C] (single frame) to a batched [1, H, W, C] image
    # tensor as comfy_image_to_pil expects (it selects index 0 internally).
    if frame_tensor.dim() == 3:
        img_batch = frame_tensor.unsqueeze(0)
    else:
        img_batch = frame_tensor
    pil_image = comfy_image_to_pil(img_batch)
    img_w, img_h = pil_image.size

    processor = sam3_model.processor
    model = processor.model

    # Interactive predictor required — same precondition SAM3Segmentation has.
    if getattr(model, "inst_interactive_predictor", None) is None:
        print(
            "[SAM3 MaskRefine] WARN: inst_interactive_predictor unavailable; "
            "cannot run image-mode segmentation for compose path. "
            "Falling back to base mask (no compose)."
        )
        return None

    if hasattr(processor, "sync_device_with_model"):
        processor.sync_device_with_model()

    state = processor.set_image(pil_image)

    # Collect points + labels in PIXEL coords. Handles both legacy single-
    # object format ({"points": [...]}) AND the multi-object format that the
    # SAM3PointCollector emits ({"objects": [{positive_points, negative_points,
    # ...}, ...]}). For v1 single-subject scope we use the FIRST object only
    # if multi-object arrives; both polarities are extracted from that object.
    #
    # R1 review fix (Gemini): pre-fix parser only extracted `positive_points`
    # when label_val=1, silently dropping multi-object negatives entirely.
    # Post-fix: when multi-object dict is encountered, extract BOTH polarities
    # at once regardless of which caller arg we're processing.
    all_points: list = []
    all_labels: list = []

    def _extend_with_pixel_points(pts_iterable, label_val):
        for pt in pts_iterable:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                all_points.append([float(pt[0]) * img_w, float(pt[1]) * img_h])
                all_labels.append(label_val)

    multi_object_consumed = False

    def _add_points_dict(pts_dict, default_label):
        nonlocal multi_object_consumed
        if not pts_dict:
            return
        if pts_dict.get("objects"):
            # Multi-object dict embeds BOTH polarities inside the first object's
            # positive_points/negative_points lists. Consume both polarities
            # exactly once — track so the negative_points_dict pass doesn't
            # re-walk the same payload.
            if multi_object_consumed:
                return
            first = pts_dict["objects"][0] if pts_dict["objects"] else {}
            _extend_with_pixel_points(first.get("positive_points", []), 1)
            _extend_with_pixel_points(first.get("negative_points", []), 0)
            multi_object_consumed = True
            return
        # Legacy single-object format
        _extend_with_pixel_points(pts_dict.get("points", []), default_label)

    _add_points_dict(positive_points_dict, 1)
    _add_points_dict(negative_points_dict, 0)

    # Positive box: predict_inst takes one box as a region constraint. We use
    # only the first positive box (matches SAM3Segmentation.segment() semantics).
    box_array = None
    if positive_boxes_dict and positive_boxes_dict.get("boxes"):
        b = positive_boxes_dict["boxes"][0]
        cx, cy, bw, bh = b
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        box_array = np.array([x1, y1, x2, y2], dtype=np.float32)

    # Negative box → encode as a negative point at the box centroid.
    # R1 review fix (Gemini): pre-fix dropped negative_boxes entirely, then
    # Step 3 added them as direct prompts AFTER the composed mask — triggering
    # the mutual-exclusion clear. Post-fix: rasterize negatives as in-prompt
    # negative points so predict_inst incorporates them into the corrective_mask
    # and Step 3 then nulls them out (no separate add_new_points call needed).
    if negative_boxes_dict and negative_boxes_dict.get("boxes"):
        for b in negative_boxes_dict["boxes"]:
            cx, cy, _bw, _bh = b
            all_points.append([float(cx) * img_w, float(cy) * img_h])
            all_labels.append(0)

    # Short-circuit: image-mode predict_inst REQUIRES at least one positive
    # signal (positive point OR positive box). With only negatives, the call
    # has no anchor and returns garbage. R1 review (Codex): document and guard.
    # Return None so the caller falls through to base mask — the user's negative
    # intent is lost in this v1 contract; document as limitation.
    has_positive_anchor = (
        any(lbl == 1 for lbl in all_labels) or box_array is not None
    )
    has_any_signal = bool(all_points) or box_array is not None
    if not has_any_signal:
        return None
    if not has_positive_anchor:
        print(
            "[SAM3 MaskRefine] WARN: image-mode segmentation needs at least one "
            "positive point or positive box to anchor; got negative-only "
            "prompts. Falling back to base mask (compose no-op). v1 limitation."
        )
        return None

    point_coords = np.array(all_points, dtype=np.float32) if all_points else None
    point_labels = np.array(all_labels, dtype=np.int32) if all_labels else None

    # R1 review fix (Codex): wrap predict_inst in try/finally so backbone
    # features get released even on exception. Without this, image features
    # stay live indefinitely on GPU when an upstream raise propagates.
    try:
        masks_np, scores_np, _low_res = model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_array,
            mask_input=None,
            multimask_output=True,
            normalize_coords=True,
        )
    finally:
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if masks_np is None or len(masks_np) == 0:
        print(
            "[SAM3 MaskRefine] WARN: image-mode segmentation returned no masks; "
            "compose falls back to base mask."
        )
        return None

    # Pick highest-scoring candidate.
    best_idx = int(np.argmax(scores_np)) if scores_np is not None else 0
    best_np = masks_np[best_idx].astype(np.float32)
    # predict_inst returns binary at image resolution; clamp to [0, 1] just in case.
    best_np = np.clip(best_np, 0.0, 1.0)
    return torch.from_numpy(best_np)


class SAM3MaskRefine:
    """
    Refine existing masks by feeding them back into SAM3 as conditioning.

    Takes masks from a first-pass SAM3 run (or any ComfyUI mask sequence) and
    uses them as mask prompts at regular keyframe intervals. SAM3 then re-tracks
    using these masks as initialization, producing refined masks that can be
    merged with the originals.

    Typical workflow:
        SAM3VideoSegmentation -> SAM3Propagate -> SAM3VideoOutput -> SAM3MaskRefine
        (first pass tracking)                     (extract masks)   (refine pass)

    Corrective compose (2026-05-20 audit Bug #2):
        SAM3's tracker enforces mutual exclusion between mask + point/box
        prompts at the same (frame_idx, obj_id) — see file:line citations in
        the _resolve_compose_op docstring above. To honor BOTH inputs at a
        corrective frame, set `corrective_compose_mode != "skip"` and the node
        will pre-compose a single mask via image-mode SAM3 segmentation +
        union/diff op. Default "skip" preserves pre-2026-05-20 behavior for
        back-compat.
    """
    # Bounded session-scoped cache (replaces legacy `_cache = {}`).
    # max_size=2: refine results include full mask sequences and visualizations.
    _cache = _BoundedSessionCache("MaskRefine", max_size=2)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model (from LoadSAM3Model)"
                }),
                "video_frames": ("IMAGE", {
                    "tooltip": "Original video frames [N, H, W, C]"
                }),
                "input_masks": ("MASK", {
                    "tooltip": "Existing mask sequence [N, H, W] from first pass"
                }),
            },
            "optional": {
                "keyframe_interval": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Use mask from every Nth frame as conditioning. Lower = more constraints (closer to original). Higher = more freedom to refine."
                }),
                "merge_mode": (["union", "replace", "intersect"], {
                    "default": "union",
                    "tooltip": "How to merge refined masks with originals: union (adds regions), replace (use refined only), intersect (conservative refinement)"
                }),
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Additional corrective points to add on frame_idx"
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Negative corrective points to add on frame_idx"
                }),
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Additional corrective boxes to add on frame_idx"
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Negative corrective boxes to add on frame_idx"
                }),
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame to apply optional point/box prompts"
                }),
                "obj_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Object ID for mask prompts (use 1 for single-object workflows)"
                }),
                "offload_video_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Store video frames on CPU (minor overhead, saves VRAM)"
                }),
                "offload_state_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store inference state on CPU (slower, saves more VRAM)"
                }),
                "offload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move model to CPU after refinement to free VRAM"
                }),
                # Appended at end of optional block per D-201 widget-position
                # rule. New widget default ("skip") preserves pre-2026-05-20
                # behavior — saved workflows with no corrective_compose_mode
                # value see no change.
                "corrective_compose_mode": (
                    ["skip", "auto", "union", "intersect", "replace", "diff"],
                    {
                        "default": "skip",
                        "tooltip": (
                            "How to handle the collision when corrective "
                            "points/box land on the same frame as a mask "
                            "keyframe. SAM3's tracker enforces mutual "
                            "exclusion between mask and point/box at the "
                            "same (frame, obj_id) — they cannot coexist as "
                            "separate prompts. Modes: "
                            "skip (default) = legacy behavior, drop the "
                            "mask keyframe at corrective_frame so points "
                            "take effect. "
                            "auto = run image-mode SAM3 on the corrective "
                            "prompts and compose with input_mask (positive-"
                            "only → union, negative-only → diff, mixed → "
                            "replace). RECOMMENDED for new workflows. "
                            "union / intersect / replace / diff = force a "
                            "specific compose op regardless of polarity."
                        ),
                    },
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, sam3_model, video_frames, input_masks,
                   keyframe_interval=10, merge_mode="union",
                   positive_points=None, negative_points=None,
                   positive_boxes=None, negative_boxes=None,
                   frame_idx=0, obj_id=1,
                   offload_video_to_cpu=True, offload_state_to_cpu=False,
                   offload_model=False,
                   corrective_compose_mode="skip"):
        if video_frames is None or input_masks is None:
            return float("NaN")
        return (id(video_frames), id(input_masks), keyframe_interval, merge_mode,
                str(positive_points), str(negative_points),
                str(positive_boxes), str(negative_boxes),
                frame_idx, obj_id, offload_video_to_cpu, offload_state_to_cpu,
                corrective_compose_mode)

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "visualization")
    FUNCTION = "refine"
    CATEGORY = "SAM3/video"

    def _create_visualization(self, video_frames, masks_tensor):
        """Create colored overlay visualization of refined masks on source frames."""
        from .utils import DEFAULT_COLORS

        vis_list = []
        color = torch.tensor(DEFAULT_COLORS[0], dtype=torch.float32)  # Blue

        for fi in range(masks_tensor.shape[0]):
            comfy.model_management.throw_exception_if_processing_interrupted()
            frame = video_frames[fi]  # [H, W, C]
            mask = masks_tensor[fi].float()  # [H, W]

            mask_rgb = mask.unsqueeze(-1) * color.view(1, 1, 3)
            vis_frame = frame * (1 - 0.5 * mask.unsqueeze(-1)) + 0.5 * mask_rgb
            vis_list.append(vis_frame.clamp(0, 1))

        return torch.stack(vis_list, dim=0)

    def refine(self, sam3_model, video_frames, input_masks,
               keyframe_interval=10, merge_mode="union",
               positive_points=None, negative_points=None,
               positive_boxes=None, negative_boxes=None,
               frame_idx=0, obj_id=1,
               offload_video_to_cpu=True, offload_state_to_cpu=False,
               offload_model=False,
               corrective_compose_mode="skip"):
        """Refine masks by feeding keyframes back into SAM3."""
        # Defaults for older workflows
        keyframe_interval = keyframe_interval if keyframe_interval is not None else 10
        merge_mode = merge_mode if merge_mode is not None else "union"
        obj_id = obj_id if obj_id is not None else 1
        corrective_compose_mode = (
            corrective_compose_mode if corrective_compose_mode is not None else "skip"
        )
        # R2 refactor: validation delegated to module-level helper so it's
        # directly unit-testable without instantiating the node + dragging in
        # the rest of the refine pipeline.
        _validate_compose_mode(corrective_compose_mode)

        # --- Cache check ---
        # Content-derived semantic key (no id()). SAM3MaskRefine has no
        # video_state input, so we rely on tensor signatures + the teardown
        # callback for safety; if upstream invalidated the C++ inference state,
        # _flush_all_sam3_caches has already cleared this cache before we get
        # here.
        cache_key = (
            "maskRefine",
            _tensor_signature(video_frames),
            _tensor_signature(input_masks),
            keyframe_interval, merge_mode,
            _scalar_or_str(positive_points), _scalar_or_str(negative_points),
            _scalar_or_str(positive_boxes), _scalar_or_str(negative_boxes),
            frame_idx, obj_id, offload_video_to_cpu, offload_state_to_cpu,
            corrective_compose_mode,
        )

        if cache_key in SAM3MaskRefine._cache:
            print(f"[SAM3 MaskRefine] CACHE HIT")
            if offload_model:
                if hasattr(sam3_model, 'model'):
                    sam3_model.model.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return SAM3MaskRefine._cache[cache_key]

        import time as _time
        _t_start = _time.time()
        print(f"[SAM3 MaskRefine] CACHE MISS - running refinement")
        print_vram("Before mask refine")

        num_video_frames = video_frames.shape[0]
        num_mask_frames = input_masks.shape[0]
        h, w = video_frames.shape[1], video_frames.shape[2]

        # Handle frame count mismatch
        num_frames = min(num_video_frames, num_mask_frames)
        if num_video_frames != num_mask_frames:
            print(f"[SAM3 MaskRefine] Warning: video has {num_video_frames} frames, "
                  f"masks have {num_mask_frames} frames. Using first {num_frames}.")

        working_frames = video_frames[:num_frames]

        # --- Step 1: Create video state ---
        config = VideoConfig(
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        video_state = create_video_state(
            video_frames=working_frames,
            config=config,
        )
        _t_state = _time.time()
        print(f"[SAM3 MaskRefine] Session {video_state.session_uuid[:8]}: "
              f"{num_frames} frames, {w}x{h} ({_t_state - _t_start:.1f}s)")

        # --- Step 2: Add mask prompts at keyframe intervals ---
        # SAM3's tracker enforces mutual exclusion between mask + point/box at
        # the same (frame_idx, obj_id) — `add_new_mask` clears point inputs,
        # `add_new_points_or_box` clears mask inputs (verified at
        # sam3_lib/model/sam3_tracking_predictor.py:266,406). They can't coexist.
        #
        # When corrective_compose_mode != "skip", we pre-compose a single mask
        # at the corrective frame: run image-mode SAM3 with the user's points/
        # box → corrective_mask, then combine with input_mask via the chosen op
        # (auto/union/intersect/replace/diff). Single composed mask feeds the
        # tracker via add_new_mask — no contract violation. v1 scope is single-
        # subject only (caller's obj_id parameter, no multi-object compose).
        # R1 review fix (Gemini): the multi-object format on positive_points
        # embeds both positive and negative points inside `objects[].positive_points`
        # and `objects[].negative_points`. Pre-fix has_corrective_negs only
        # checked legacy `negative_points.points`, missing multi-object negs
        # entirely → auto picked wrong op. Post-fix: check both formats.
        _multi_obj_pos = bool(
            positive_points and positive_points.get("objects")
            and any(o.get("positive_points") for o in positive_points["objects"])
        )
        _multi_obj_neg = bool(
            positive_points and positive_points.get("objects")
            and any(o.get("negative_points") for o in positive_points["objects"])
        )
        has_corrective_points = bool(
            (positive_points and positive_points.get("points")) or _multi_obj_pos
        )
        has_corrective_negs = bool(
            (negative_points and negative_points.get("points")) or _multi_obj_neg
        )
        has_corrective_pos_boxes = bool(positive_boxes and positive_boxes.get("boxes"))
        has_corrective_neg_boxes = bool(negative_boxes and negative_boxes.get("boxes"))
        has_corrective = (
            has_corrective_points or has_corrective_negs
            or has_corrective_pos_boxes or has_corrective_neg_boxes
        )
        corrective_frame = frame_idx if has_corrective else -1
        compose_active = (
            has_corrective
            and corrective_compose_mode != "skip"
            and 0 <= corrective_frame < num_frames
        )

        # Build the set of keyframes to add. Standard interval keyframes plus
        # the corrective_frame itself when compose mode is active (so the
        # composed mask is added even if corrective_frame doesn't align with
        # the interval grid).
        keyframes_to_add = set(range(0, num_frames, keyframe_interval))
        if compose_active:
            keyframes_to_add.add(corrective_frame)
        ordered_keyframes = sorted(keyframes_to_add)

        keyframes_added = 0
        composed_at_frame = -1  # tracks which frame got the composed mask
        num_keyframes = len(ordered_keyframes)
        mask_pixels = h * w

        # Warn about memory if many large masks will be stored as flattened tuples
        estimated_mb = num_keyframes * mask_pixels * 4 / (1024 * 1024)  # 4 bytes per float
        if estimated_mb > 500:
            print(f"[SAM3 MaskRefine] Warning: storing {num_keyframes} mask keyframes "
                  f"at {w}x{h} will use ~{estimated_mb:.0f}MB RAM in video state. "
                  f"Consider increasing keyframe_interval to reduce memory usage.")

        for kf_idx in ordered_keyframes:
            is_corrective_kf = (kf_idx == corrective_frame and has_corrective)

            # Legacy "skip" path: drop the mask keyframe so corrective prompts
            # take effect via the standard add_new_points_or_box path in Step 3.
            if is_corrective_kf and corrective_compose_mode == "skip":
                print(f"[SAM3 MaskRefine] Skipping mask keyframe {kf_idx} "
                      f"(corrective prompts take priority; compose_mode=skip)")
                continue

            frame_mask = input_masks[kf_idx].float()

            # Normalize to 0-1
            if frame_mask.max() > 1.0:
                frame_mask = frame_mask / 255.0

            # Resize if needed
            mask_h, mask_w = frame_mask.shape[-2], frame_mask.shape[-1]
            if mask_h != h or mask_w != w:
                frame_mask = torch.nn.functional.interpolate(
                    frame_mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            # Compose path: run image-mode SAM3 on the corrective prompts and
            # combine with the input mask. Falls through to base mask if
            # image-mode returns nothing or no usable prompts.
            if is_corrective_kf and compose_active:
                corrective_mask = _run_image_mode_segmentation(
                    sam3_model,
                    video_frames[kf_idx],
                    positive_points,
                    negative_points,
                    positive_boxes,
                    negative_boxes,
                )
                if corrective_mask is not None:
                    # Match input-mask resolution.
                    if corrective_mask.shape[-2:] != (h, w):
                        corrective_mask = torch.nn.functional.interpolate(
                            corrective_mask.unsqueeze(0).unsqueeze(0),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    has_pos = has_corrective_points or has_corrective_pos_boxes
                    has_neg = has_corrective_negs or has_corrective_neg_boxes
                    resolved_op = _resolve_compose_op(
                        corrective_compose_mode, has_pos, has_neg
                    )
                    composed = _compose_masks(frame_mask, corrective_mask, resolved_op)
                    print(
                        f"[SAM3 MaskRefine] Composed corrective mask at frame "
                        f"{kf_idx} via op='{resolved_op}' "
                        f"(mode={corrective_compose_mode}, "
                        f"has_pos={has_pos}, has_neg={has_neg})"
                    )
                    frame_mask = composed
                    composed_at_frame = kf_idx
                else:
                    print(
                        f"[SAM3 MaskRefine] Compose at frame {kf_idx} fell "
                        f"through to base mask (no usable corrective prompts "
                        f"or image-mode returned empty)"
                    )

            # Skip empty masks (post-compose). Threshold kept at 100 for
            # back-compat with existing behavior.
            if frame_mask.sum() < 100:
                print(f"[SAM3 MaskRefine] Skipping empty keyframe {kf_idx}")
                continue

            prompt = VideoPrompt.create_mask(
                frame_idx=kf_idx,
                obj_id=obj_id,
                mask=frame_mask,
            )
            video_state = video_state.with_prompt(prompt)
            keyframes_added += 1

        print(f"[SAM3 MaskRefine] Added {keyframes_added} mask keyframes "
              f"(interval={keyframe_interval}, compose_mode={corrective_compose_mode})")

        # --- Step 3: Add optional corrective prompts ---
        # Suppression rule (R2 review fold-in — Codex Medium):
        #
        # If compose_active is True the user explicitly requested compose
        # semantics. We must NOT add direct point/box prompts at the corrective
        # frame in that case — even when _run_image_mode_segmentation returned
        # None (negative-only inputs, missing predictor, empty image-mode
        # result). Pre-R3 guard only suppressed when compose SUCCEEDED
        # (composed_at_frame == corrective_frame), which meant a fallback-to-
        # base-mask path still added the corrective prompts via add_new_points_
        # or_box → triggered the tracker's mutual-exclusion clear → wiped the
        # base mask keyframe we just added. Exactly the bug the compose path
        # was meant to eliminate.
        #
        # New rule: when compose_active, ALWAYS suppress direct prompts. Log
        # whether compose actually produced a mask, so the user can see if
        # corrective intent landed or fell through.
        if compose_active:
            if composed_at_frame == corrective_frame:
                print(
                    f"[SAM3 MaskRefine] Corrective prompts at frame {frame_idx} "
                    f"composed into mask via Step 2 (op resolved); skipping "
                    f"direct add to avoid mutual-exclusion clear."
                )
            else:
                print(
                    f"[SAM3 MaskRefine] WARN: compose was requested at frame "
                    f"{frame_idx} but image-mode segmentation returned no "
                    f"mask (negative-only / no predictor / empty result). "
                    f"Base mask kept; direct corrective prompts SUPPRESSED to "
                    f"protect that mask. To use direct prompts, set "
                    f"corrective_compose_mode='skip'."
                )
            positive_points = None
            negative_points = None
            positive_boxes = None
            negative_boxes = None

        prompts_added = 0

        if positive_points:
            if positive_points.get("objects"):
                # Multi-object format
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
                        prompts_added += 1
            elif positive_points.get("points"):
                # Legacy single-object format
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
                    prompts_added += 1

        if positive_boxes and positive_boxes.get("boxes"):
            for box_data in positive_boxes["boxes"]:
                cx, cy, bw, bh = box_data
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=True)
                video_state = video_state.with_prompt(prompt)
                prompts_added += 1

        if negative_boxes and negative_boxes.get("boxes"):
            for box_data in negative_boxes["boxes"]:
                cx, cy, bw, bh = box_data
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=False)
                video_state = video_state.with_prompt(prompt)
                prompts_added += 1

        if prompts_added > 0:
            print(f"[SAM3 MaskRefine] Added {prompts_added} corrective prompts on frame {frame_idx}")

        # Validate at least one prompt
        if len(video_state.prompts) == 0:
            raise ValueError("[SAM3 MaskRefine] No valid prompts. Input masks may be all empty.")

        # --- Step 4: Ensure model on GPU ---
        if hasattr(sam3_model, 'model') and hasattr(sam3_model.model, 'to'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sam3_model.model.to(device)

        # --- Step 5: Propagate ---
        request = {
            "type": "propagate_in_video",
            "session_id": video_state.session_uuid,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": num_frames,
        }

        refined_masks = {}
        autocast_context = _get_autocast_context()

        with autocast_context:
            print_vram("Before reconstruction")
            _t_recon = _time.time()
            inference_state = get_inference_state(sam3_model, video_state)
            print(f"[SAM3 MaskRefine] State reconstruction: {_time.time() - _t_recon:.1f}s")
            print_vram("After reconstruction")

            _t_prop = _time.time()
            _frames_processed = 0
            _log_interval = max(1, num_frames // 10)  # Log ~10 times during propagation
            try:
                for response in sam3_model.handle_stream_request(request):
                    comfy.model_management.throw_exception_if_processing_interrupted()

                    frame_index = response.get("frame_index", response.get("frame_idx"))
                    if frame_index is None:
                        continue

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        continue

                    # Extract mask (same key search pattern as SAM3Propagate)
                    mask = None
                    mask_key = None
                    for key in ["out_binary_masks", "video_res_masks", "masks"]:
                        if key in outputs and outputs[key] is not None:
                            mask_key = key
                            mask = outputs[key]
                            break

                    if mask is not None:
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu()
                        refined_masks[frame_index] = mask
                        del outputs[mask_key]

                    outputs.clear()
                    _frames_processed += 1

                    if _frames_processed % _log_interval == 0 or _frames_processed == num_frames:
                        _elapsed = _time.time() - _t_prop
                        _pct = (_frames_processed / num_frames) * 100
                        _fps = _frames_processed / max(_elapsed, 0.001)
                        _eta = (num_frames - _frames_processed) / max(_fps, 0.001)
                        print(f"[SAM3 MaskRefine] Propagating: {_frames_processed}/{num_frames} "
                              f"({_pct:.0f}%) | {_fps:.1f} fps | ETA {_eta:.0f}s")

                    if frame_index % 16 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            except Exception as e:
                print(f"[SAM3 MaskRefine] Propagation error at frame {_frames_processed}: {e}")
                import traceback
                traceback.print_exc()
                raise

        _prop_time = _time.time() - _t_prop
        print_vram("After propagation")
        print(f"[SAM3 MaskRefine] Propagation complete: {len(refined_masks)} frames "
              f"in {_prop_time:.1f}s ({len(refined_masks)/max(_prop_time, 0.001):.1f} fps)")

        # --- Step 6: Merge refined masks with originals ---
        _t_merge = _time.time()
        merged_list = []
        _total_original_px = 0
        _total_refined_px = 0
        _total_merged_px = 0
        _frames_with_change = 0
        for fi in range(num_frames):
            original_mask = input_masks[fi].float().cpu()

            if original_mask.max() > 1.0:
                original_mask = original_mask / 255.0

            if fi in refined_masks:
                raw_refined = refined_masks[fi]

                # Convert to combined 2D mask [H, W]
                if isinstance(raw_refined, np.ndarray):
                    raw_refined = torch.from_numpy(raw_refined)
                if raw_refined.dim() == 4:
                    raw_refined = raw_refined.squeeze(0)
                if raw_refined.dim() == 3:
                    # Multi-object [num_obj, H, W] -> combine via max
                    raw_refined = raw_refined.max(dim=0)[0]
                raw_refined = raw_refined.float().cpu()
                if raw_refined.max() > 1.0:
                    raw_refined = raw_refined / 255.0

                # Resize if dimensions don't match
                rh, rw = raw_refined.shape[-2], raw_refined.shape[-1]
                oh, ow = original_mask.shape[-2], original_mask.shape[-1]
                if rh != oh or rw != ow:
                    raw_refined = torch.nn.functional.interpolate(
                        raw_refined.unsqueeze(0).unsqueeze(0),
                        size=(oh, ow),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

                # Apply merge mode
                if merge_mode == "union":
                    merged = torch.max(original_mask, raw_refined)
                elif merge_mode == "replace":
                    merged = raw_refined
                elif merge_mode == "intersect":
                    merged = torch.min(original_mask, raw_refined)
                else:
                    merged = torch.max(original_mask, raw_refined)

                # Track coverage stats
                orig_px = (original_mask > 0.5).sum().item()
                ref_px = (raw_refined > 0.5).sum().item()
                merge_px = (merged > 0.5).sum().item()
                _total_original_px += orig_px
                _total_refined_px += ref_px
                _total_merged_px += merge_px
                if merge_px != orig_px:
                    _frames_with_change += 1
            else:
                # No refined mask for this frame - keep original
                merged = original_mask
                _total_original_px += (original_mask > 0.5).sum().item()
                _total_merged_px += (original_mask > 0.5).sum().item()

            merged_list.append(merged)

        all_masks = torch.stack(merged_list, dim=0)

        # Log merge stats
        _merge_time = _time.time() - _t_merge
        _avg_orig = _total_original_px / max(num_frames, 1)
        _avg_merged = _total_merged_px / max(num_frames, 1)
        _change_pct = ((_total_merged_px - _total_original_px) / max(_total_original_px, 1)) * 100
        print(f"[SAM3 MaskRefine] Merge ({merge_mode}): {_merge_time:.1f}s | "
              f"{_frames_with_change}/{num_frames} frames changed")
        print(f"[SAM3 MaskRefine] Coverage: original avg {_avg_orig:.0f}px → "
              f"merged avg {_avg_merged:.0f}px ({_change_pct:+.1f}%)")

        # --- Step 7: Visualization ---
        visualization = self._create_visualization(working_frames.cpu(), all_masks)

        # --- Step 8: Cleanup ---
        refined_masks.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if offload_model:
            print("[SAM3 MaskRefine] Offloading model to CPU...")
            if hasattr(sam3_model, 'model'):
                sam3_model.model.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_vram("After model offload")

        # Cache result
        result = (all_masks, visualization)
        SAM3MaskRefine._cache[cache_key] = result

        _total_time = _time.time() - _t_start
        print(f"[SAM3 MaskRefine] Done. Output: {all_masks.shape} | Total: {_total_time:.1f}s")
        print_vram("After mask refine complete")

        return result


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
    "SAM3MaskRefine": SAM3MaskRefine,
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
    "SAM3MaskRefine": "SAM3 Mask Refine",
}
