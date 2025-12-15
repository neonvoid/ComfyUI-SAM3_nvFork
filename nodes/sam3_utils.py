"""
SAM3 Utility Nodes - Cleanup and memory management utilities

Provides nodes for managing SAM3 session memory and GPU resources.
"""

import gc
import time
import torch


def clear_all_sam3_caches(verbose=True):
    """
    Clear ALL SAM3 caches throughout the codebase.

    This clears:
    - Sam3VideoPredictor._ALL_INFERENCE_STATES (inference sessions)
    - SAM3VideoSegmentation._cache (video states)
    - SAM3Propagate._cache (propagation results with masks)
    - SAM3VideoOutput._cache (output results)
    - SAM3PointCollector._cache (point prompts)
    - SAM3BBoxCollector._cache (box prompts)
    - InferenceReconstructor._cache (inference state cache)

    Returns dict with counts of what was cleared.
    """
    cleared = {}

    # 1. Clear inference sessions (the main culprit)
    try:
        from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
        cleared["inference_sessions"] = len(Sam3VideoPredictor._ALL_INFERENCE_STATES)
        if cleared["inference_sessions"] > 0:
            Sam3VideoPredictor._ALL_INFERENCE_STATES.clear()
            if verbose:
                print(f"[SAM3 Cleanup] Cleared {cleared['inference_sessions']} inference session(s)")
    except Exception as e:
        if verbose:
            print(f"[SAM3 Cleanup] Warning: Could not clear inference sessions: {e}")
        cleared["inference_sessions"] = 0

    # 2. Clear video node caches (these hold masks and video states!)
    try:
        from .sam3_video_nodes import SAM3VideoSegmentation, SAM3Propagate, SAM3VideoOutput

        cleared["video_segmentation_cache"] = len(SAM3VideoSegmentation._cache)
        cleared["propagate_cache"] = len(SAM3Propagate._cache)
        cleared["video_output_cache"] = len(SAM3VideoOutput._cache)

        if cleared["video_segmentation_cache"] > 0:
            SAM3VideoSegmentation._cache.clear()
            if verbose:
                print(f"[SAM3 Cleanup] Cleared {cleared['video_segmentation_cache']} video segmentation cache entries")

        if cleared["propagate_cache"] > 0:
            SAM3Propagate._cache.clear()
            if verbose:
                print(f"[SAM3 Cleanup] Cleared {cleared['propagate_cache']} propagate cache entries (masks)")

        if cleared["video_output_cache"] > 0:
            SAM3VideoOutput._cache.clear()
            if verbose:
                print(f"[SAM3 Cleanup] Cleared {cleared['video_output_cache']} video output cache entries")
    except Exception as e:
        if verbose:
            print(f"[SAM3 Cleanup] Warning: Could not clear video node caches: {e}")

    # 3. Clear interactive node caches
    try:
        from .sam3_interactive import SAM3PointCollector, SAM3BBoxCollector

        cleared["point_collector_cache"] = len(SAM3PointCollector._cache)
        cleared["bbox_collector_cache"] = len(SAM3BBoxCollector._cache)

        if cleared["point_collector_cache"] > 0:
            SAM3PointCollector._cache.clear()
            if verbose:
                print(f"[SAM3 Cleanup] Cleared {cleared['point_collector_cache']} point collector cache entries")

        if cleared["bbox_collector_cache"] > 0:
            SAM3BBoxCollector._cache.clear()
            if verbose:
                print(f"[SAM3 Cleanup] Cleared {cleared['bbox_collector_cache']} bbox collector cache entries")
    except Exception as e:
        if verbose:
            print(f"[SAM3 Cleanup] Warning: Could not clear interactive caches: {e}")

    # 4. Clear inference reconstructor cache
    try:
        from .inference_reconstructor import InferenceReconstructor
        reconstructor = InferenceReconstructor.get_instance()
        cleared["inference_reconstructor_cache"] = len(reconstructor._cache)
        if cleared["inference_reconstructor_cache"] > 0:
            reconstructor._cache.clear()
            if verbose:
                print(f"[SAM3 Cleanup] Cleared {cleared['inference_reconstructor_cache']} inference reconstructor cache entries")
    except Exception as e:
        if verbose:
            print(f"[SAM3 Cleanup] Warning: Could not clear inference reconstructor cache: {e}")

    # 5. Force garbage collection (multiple passes for nested references)
    gc.collect()
    gc.collect()

    # 6. Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Also synchronize to ensure all operations complete
        try:
            torch.cuda.synchronize()
        except:
            pass

    return cleared


def get_total_cleared(cleared_dict):
    """Sum up all cleared items from the dict."""
    return sum(v for v in cleared_dict.values() if isinstance(v, int))


class SAM3Cleanup:
    """
    Force cleanup of ALL SAM3 caches and GPU memory.

    Use this node at the end of your workflow to ensure all SAM3
    data is cleared from memory before running other workflows.

    Clears:
    - Inference sessions
    - Video segmentation cache
    - Propagation cache (masks)
    - Video output cache
    - Interactive node caches
    - Inference reconstructor cache
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"tooltip": "Connect any output here to ensure cleanup runs after upstream nodes"}),
                "clear_cuda_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also call torch.cuda.empty_cache() to free reserved VRAM"
                }),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("items_cleared", "memory_info")
    FUNCTION = "cleanup"
    OUTPUT_NODE = True
    CATEGORY = "SAM3/Utils"

    def cleanup(self, trigger=None, clear_cuda_cache=True):
        """Clear all SAM3 caches and optionally CUDA cache."""
        # Get memory stats before cleanup
        if torch.cuda.is_available():
            before_alloc = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            before_alloc = before_reserved = 0

        # Run comprehensive cleanup
        cleared = clear_all_sam3_caches(verbose=True)
        total_cleared = get_total_cleared(cleared)

        # Extra CUDA cache clear if requested
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get memory stats after cleanup
        if torch.cuda.is_available():
            after_alloc = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3
            freed_alloc = before_alloc - after_alloc
            freed_reserved = before_reserved - after_reserved

            memory_info = (
                f"Cleared {total_cleared} items. "
                f"VRAM: {after_alloc:.2f}GB allocated ({freed_alloc:+.2f}GB), "
                f"{after_reserved:.2f}GB reserved ({freed_reserved:+.2f}GB)"
            )
        else:
            memory_info = f"Cleared {total_cleared} items. (CPU mode)"

        print(f"[SAM3 Cleanup] {memory_info}")

        return (total_cleared, memory_info)


class SAM3SessionInfo:
    """
    Display information about active SAM3 caches and memory usage.

    Useful for debugging memory issues - shows what caches are
    currently populated and how much memory might be used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("total_cached_items", "cache_info")
    FUNCTION = "get_info"
    CATEGORY = "SAM3/Utils"

    def get_info(self):
        """Get information about all SAM3 caches."""
        lines = ["=== SAM3 Cache Status ==="]
        total_items = 0

        # 1. Inference sessions
        try:
            from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
            num_sessions = len(Sam3VideoPredictor._ALL_INFERENCE_STATES)
            total_items += num_sessions
            if num_sessions > 0:
                lines.append(f"\nInference Sessions: {num_sessions}")
                for sid, session in Sam3VideoPredictor._ALL_INFERENCE_STATES.items():
                    state = session.get("state", {})
                    num_frames = state.get("num_frames", "?")
                    age = time.time() - session.get("start_time", time.time())
                    lines.append(f"  {sid[:8]}...: {num_frames} frames, {age:.0f}s old")
            else:
                lines.append(f"\nInference Sessions: 0")
        except Exception as e:
            lines.append(f"\nInference Sessions: error ({e})")

        # 2. Video node caches
        try:
            from .sam3_video_nodes import SAM3VideoSegmentation, SAM3Propagate, SAM3VideoOutput

            seg_count = len(SAM3VideoSegmentation._cache)
            prop_count = len(SAM3Propagate._cache)
            out_count = len(SAM3VideoOutput._cache)
            total_items += seg_count + prop_count + out_count

            lines.append(f"\nVideo Node Caches:")
            lines.append(f"  VideoSegmentation: {seg_count} entries")
            lines.append(f"  Propagate: {prop_count} entries (contains masks!)")
            lines.append(f"  VideoOutput: {out_count} entries")
        except Exception as e:
            lines.append(f"\nVideo Node Caches: error ({e})")

        # 3. Interactive caches
        try:
            from .sam3_interactive import SAM3PointCollector, SAM3BBoxCollector

            point_count = len(SAM3PointCollector._cache)
            bbox_count = len(SAM3BBoxCollector._cache)
            total_items += point_count + bbox_count

            lines.append(f"\nInteractive Caches:")
            lines.append(f"  PointCollector: {point_count} entries")
            lines.append(f"  BBoxCollector: {bbox_count} entries")
        except Exception as e:
            lines.append(f"\nInteractive Caches: error ({e})")

        # 4. Inference reconstructor
        try:
            from .inference_reconstructor import InferenceReconstructor
            recon_count = len(InferenceReconstructor.get_instance()._cache)
            total_items += recon_count
            lines.append(f"\nInference Reconstructor Cache: {recon_count} entries")
        except Exception as e:
            lines.append(f"\nInference Reconstructor Cache: error ({e})")

        # Add VRAM info
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            lines.append(f"\n=== VRAM ===")
            lines.append(f"Allocated: {alloc:.2f}GB")
            lines.append(f"Reserved: {reserved:.2f}GB")

        lines.append(f"\n=== Total Cached Items: {total_items} ===")

        info = "\n".join(lines)
        print(f"[SAM3 SessionInfo]\n{info}")
        return (total_items, info)


class SAM3AutoCleanup:
    """
    Automatic cleanup that triggers after receiving mask output.

    Connect this after SAM3MaskTracks or SAM3VideoOutput to automatically
    clean up ALL SAM3 caches once masks have been extracted.

    This is the recommended way to ensure SAM3 doesn't leak memory
    and cause OOM errors in subsequent workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "Masks from SAM3 (cleanup triggers after this node executes)"}),
            },
            "optional": {
                "clear_cuda_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also call torch.cuda.empty_cache()"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("masks", "items_cleared")
    FUNCTION = "cleanup_and_passthrough"
    CATEGORY = "SAM3/Utils"

    def cleanup_and_passthrough(self, masks, clear_cuda_cache=True):
        """Pass through masks and clean up all SAM3 caches."""
        # Get memory before
        if torch.cuda.is_available():
            before_alloc = torch.cuda.memory_allocated() / 1024**3

        # Run comprehensive cleanup
        cleared = clear_all_sam3_caches(verbose=True)
        total_cleared = get_total_cleared(cleared)

        # Extra CUDA clear
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            after_alloc = torch.cuda.memory_allocated() / 1024**3
            freed = before_alloc - after_alloc
            print(f"[SAM3 AutoCleanup] Complete. Freed {freed:.2f}GB VRAM")
        else:
            print(f"[SAM3 AutoCleanup] Complete. Cleared {total_cleared} cached items")

        return (masks, total_cleared)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SAM3Cleanup": SAM3Cleanup,
    "NV_SAM3SessionInfo": SAM3SessionInfo,
    "NV_SAM3AutoCleanup": SAM3AutoCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SAM3Cleanup": "NV SAM3 Cleanup (Force)",
    "NV_SAM3SessionInfo": "NV SAM3 Session Info",
    "NV_SAM3AutoCleanup": "NV SAM3 Auto Cleanup",
}
