"""
SAM3 Utility Nodes - Cleanup and memory management utilities

Provides nodes for managing SAM3 session memory and GPU resources.
"""

import gc
import time
import torch


class SAM3Cleanup:
    """
    Force cleanup of SAM3 inference sessions and GPU memory.

    Use this node at the end of your workflow to ensure all SAM3
    video tracking sessions are cleared from memory.

    Sessions accumulate in a class-level dictionary and can cause
    crashes when running other workflows if not cleaned up.
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
    RETURN_NAMES = ("sessions_cleared", "memory_info")
    FUNCTION = "cleanup"
    OUTPUT_NODE = True
    CATEGORY = "SAM3/Utils"

    def cleanup(self, trigger=None, clear_cuda_cache=True):
        """Clear all SAM3 inference sessions and optionally CUDA cache."""
        from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor

        # Get memory stats before cleanup
        if torch.cuda.is_available():
            before_alloc = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            before_alloc = before_reserved = 0

        # Count and clear sessions
        num_sessions = len(Sam3VideoPredictor._ALL_INFERENCE_STATES)

        if num_sessions > 0:
            print(f"[SAM3 Cleanup] Clearing {num_sessions} inference session(s)...")
            # Get session info before clearing
            session_info = []
            for sid, session in Sam3VideoPredictor._ALL_INFERENCE_STATES.items():
                state = session.get("state", {})
                frames = state.get("num_frames", "?")
                age = time.time() - session.get("start_time", time.time())
                session_info.append(f"  - {sid[:8]}... ({frames} frames, {age:.0f}s old)")

            for info in session_info:
                print(f"[SAM3 Cleanup] {info}")

            Sam3VideoPredictor._ALL_INFERENCE_STATES.clear()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if requested
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get memory stats after cleanup
        if torch.cuda.is_available():
            after_alloc = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3
            freed_alloc = before_alloc - after_alloc
            freed_reserved = before_reserved - after_reserved

            memory_info = (
                f"Cleared {num_sessions} sessions. "
                f"VRAM: {after_alloc:.2f}GB allocated ({freed_alloc:+.2f}GB), "
                f"{after_reserved:.2f}GB reserved ({freed_reserved:+.2f}GB)"
            )
        else:
            memory_info = f"Cleared {num_sessions} sessions. (CPU mode)"

        print(f"[SAM3 Cleanup] {memory_info}")

        return (num_sessions, memory_info)


class SAM3SessionInfo:
    """
    Display information about active SAM3 inference sessions.

    Useful for debugging memory issues - shows what sessions are
    currently stored and how much memory they might be using.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("num_sessions", "session_info")
    FUNCTION = "get_info"
    CATEGORY = "SAM3/Utils"

    def get_info(self):
        """Get information about active SAM3 sessions."""
        from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor

        num_sessions = len(Sam3VideoPredictor._ALL_INFERENCE_STATES)

        if num_sessions == 0:
            info = "No active SAM3 sessions"
        else:
            lines = [f"Active SAM3 sessions: {num_sessions}"]
            for sid, session in Sam3VideoPredictor._ALL_INFERENCE_STATES.items():
                state = session.get("state", {})
                num_frames = state.get("num_frames", "?")
                age = time.time() - session.get("start_time", time.time())
                lines.append(f"  {sid[:8]}...: {num_frames} frames, {age:.0f}s old")
            info = "\n".join(lines)

        # Add VRAM info
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            info += f"\n\nVRAM: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved"

        print(f"[SAM3 SessionInfo]\n{info}")
        return (num_sessions, info)


class SAM3AutoCleanup:
    """
    Automatic cleanup that triggers after receiving mask output.

    Connect this after SAM3MaskTracks or SAM3VideoOutput to automatically
    clean up sessions once masks have been extracted.

    This is the recommended way to ensure SAM3 doesn't leak memory.
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
    RETURN_NAMES = ("masks", "sessions_cleared")
    FUNCTION = "cleanup_and_passthrough"
    CATEGORY = "SAM3/Utils"

    def cleanup_and_passthrough(self, masks, clear_cuda_cache=True):
        """Pass through masks and clean up SAM3 sessions."""
        from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor

        num_sessions = len(Sam3VideoPredictor._ALL_INFERENCE_STATES)

        if num_sessions > 0:
            print(f"[SAM3 AutoCleanup] Clearing {num_sessions} session(s) after mask extraction...")
            Sam3VideoPredictor._ALL_INFERENCE_STATES.clear()
            gc.collect()
            if clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[SAM3 AutoCleanup] Cleanup complete")

        return (masks, num_sessions)


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
