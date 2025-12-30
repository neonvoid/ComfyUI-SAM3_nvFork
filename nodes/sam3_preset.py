"""
SAM3 Preset Save/Load - Batch processing for pre-defined marker sets

Save manual point markers, start frame, and colors to JSON files for later batch processing.
This enables a workflow where you:
1. Pre-process: Manually mark players on start frames, assign colors
2. Save: Store markers + start frame + colors to preset files
3. Batch: Load multiple preset files and process them all automatically
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folder_paths


class SAM3PresetSave:
    """
    Save SAM3 point markers and settings to a JSON preset file.

    Takes the output from SAM3PointCollector (positive_points) plus start frame
    and colors, and saves everything to a reusable preset file.

    Workflow:
    1. Use SAM3PointCollector to mark players on a frame
    2. Connect positive_points output to this node
    3. Set start_frame and mask_colors
    4. Run to save the preset

    The preset can later be loaded with SAM3PresetLoad for batch processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_name": ("STRING", {
                    "default": "hockey_preset_01",
                    "multiline": False,
                    "tooltip": "Name for the preset file (without .json extension)"
                }),
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Connect to SAM3PointCollector's positive_points output"
                }),
                "start_frame_forward": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to start forward tracking from"
                }),
                "start_frame_backward": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to start backward tracking from (usually same as forward or end of clip)"
                }),
                "mask_colors": ("STRING", {
                    "default": "cyan, teal, purple, magenta",
                    "multiline": False,
                    "tooltip": "Comma-separated colors for each object (e.g., 'cyan, teal, purple, magenta')"
                }),
            },
            "optional": {
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Optional: Connect to SAM3PointCollector's negative_points output"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom directory to save presets. Leave empty for default (output/sam3_presets)"
                }),
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional: Path to source video for reference"
                }),
                "notes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional notes about this preset (e.g., 'Red team players batch 1')"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("preset_path", "status")
    FUNCTION = "save_preset"
    CATEGORY = "SAM3/presets"
    OUTPUT_NODE = True

    def save_preset(
        self,
        preset_name: str,
        positive_points: dict,
        start_frame_forward: int,
        start_frame_backward: int,
        mask_colors: str,
        negative_points: dict = None,
        output_directory: str = "",
        video_path: str = "",
        notes: str = ""
    ):
        """Save markers and settings to a JSON preset file."""

        # Handle multi-object format (from SAM3PointCollector multi-object mode)
        if "objects" in positive_points:
            objects = positive_points["objects"]
        else:
            # Legacy single-object format - convert to multi-object
            pos_pts = positive_points.get("points", [])
            neg_pts = []
            if negative_points and "points" in negative_points:
                neg_pts = negative_points["points"]

            objects = [{
                "obj_id": 1,
                "positive_points": pos_pts,
                "negative_points": neg_pts
            }] if pos_pts else []

        # Validate we have objects
        if not objects:
            return ("", "Error: No objects found. Mark some players first!")

        # Build preset data
        preset = {
            "version": "1.1",
            "preset_name": preset_name,
            "start_frame_forward": start_frame_forward,
            "start_frame_backward": start_frame_backward,
            "start_frame": start_frame_forward,  # Legacy compatibility
            "mask_colors": mask_colors,
            "video_path": video_path,
            "notes": notes,
            "objects": objects,
            "object_count": len(objects),
        }

        # Determine save path
        if output_directory and output_directory.strip():
            presets_dir = output_directory.strip()
        else:
            presets_dir = os.path.join(folder_paths.get_output_directory(), "sam3_presets")
        os.makedirs(presets_dir, exist_ok=True)

        # Clean preset name
        safe_name = "".join(c for c in preset_name if c.isalnum() or c in "._- ")
        preset_path = os.path.join(presets_dir, f"{safe_name}.json")

        # Save to file
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset, f, indent=2)
        except Exception as e:
            return ("", f"Error saving preset: {e}")

        # Build status message
        obj_summary = []
        for obj in objects:
            obj_id = obj.get("obj_id", "?")
            pos_count = len(obj.get("positive_points", []))
            neg_count = len(obj.get("negative_points", []))
            obj_summary.append(f"Obj {obj_id}: {pos_count}+ {neg_count}-")

        status = (
            f"Saved preset: {preset_name}\n"
            f"Path: {preset_path}\n"
            f"Start frame (fwd): {start_frame_forward}, (bwd): {start_frame_backward}\n"
            f"Colors: {mask_colors}\n"
            f"Objects ({len(objects)}): {', '.join(obj_summary)}"
        )

        print(f"[SAM3 PresetSave] {status}")

        return (preset_path, status)


class SAM3PresetLoad:
    """
    Load SAM3 point markers and settings from a JSON preset file.

    Outputs the data needed to feed into the SAM3 video tracking pipeline:
    - positive_points: Multi-object points prompt for SAM3AddVideoPrompt
    - negative_points: Negative points prompt (if any were saved)
    - start_frame_forward: Frame index to start forward tracking
    - start_frame_backward: Frame index to start backward tracking
    - mask_colors: Color string for visualization

    Workflow:
    1. Point this node at a preset JSON file via preset_path
    2. Connect outputs to your tracking workflow
    3. Process multiple presets in batch using a loop or queue
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path to preset JSON file"
                }),
            },
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_points", "negative_points", "start_frame_forward", "start_frame_backward", "mask_colors", "video_path", "preset_info")
    FUNCTION = "load_preset"
    CATEGORY = "SAM3/presets"

    @classmethod
    def IS_CHANGED(cls, preset_path):
        # Check file modification time
        if preset_path and os.path.exists(preset_path):
            return os.path.getmtime(preset_path)
        return float("nan")

    def load_preset(self, preset_path: str):
        """Load markers and settings from a JSON preset file."""

        # Check file exists
        if not preset_path or not preset_path.strip():
            print(f"[SAM3 PresetLoad] Error: No preset path provided")
            empty_prompt = {"objects": []}
            return (empty_prompt, empty_prompt, 0, 0, "", "", "Error: No preset path provided")

        preset_path = preset_path.strip()

        if not os.path.exists(preset_path):
            print(f"[SAM3 PresetLoad] Error: Preset file not found: {preset_path}")
            empty_prompt = {"objects": []}
            return (empty_prompt, empty_prompt, 0, 0, "", "", f"Error: File not found: {preset_path}")

        # Load preset
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
        except Exception as e:
            print(f"[SAM3 PresetLoad] Error loading preset: {e}")
            empty_prompt = {"objects": []}
            return (empty_prompt, empty_prompt, 0, 0, "", "", f"Error loading: {e}")

        # Extract data
        preset_name = preset.get("preset_name", "unknown")
        # Support both new format (start_frame_forward/backward) and legacy (start_frame)
        legacy_start = preset.get("start_frame", 0)
        start_frame_forward = preset.get("start_frame_forward", legacy_start)
        start_frame_backward = preset.get("start_frame_backward", legacy_start)
        mask_colors = preset.get("mask_colors", "cyan, teal, purple, magenta")
        video_path = preset.get("video_path", "")
        notes = preset.get("notes", "")
        objects = preset.get("objects", [])

        # Build positive_points output (multi-object format)
        # This matches SAM3PointCollector's multi-object output format
        positive_points = {"objects": objects}

        # Build negative_points output (extract negative points from each object)
        negative_objects = []
        for obj in objects:
            neg_pts = obj.get("negative_points", [])
            if neg_pts:
                negative_objects.append({
                    "obj_id": obj.get("obj_id", 1),
                    "positive_points": [],  # No positive points in negative output
                    "negative_points": neg_pts
                })
        negative_points = {"objects": negative_objects}

        # Build info string
        obj_summary = []
        for obj in objects:
            obj_id = obj.get("obj_id", "?")
            pos_count = len(obj.get("positive_points", []))
            neg_count = len(obj.get("negative_points", []))
            obj_summary.append(f"Obj {obj_id}: {pos_count}+ {neg_count}-")

        preset_info = (
            f"Preset: {preset_name}\n"
            f"Start frame (fwd): {start_frame_forward}, (bwd): {start_frame_backward}\n"
            f"Colors: {mask_colors}\n"
            f"Objects ({len(objects)}): {', '.join(obj_summary)}\n"
            f"Notes: {notes}"
        )

        print(f"[SAM3 PresetLoad] Loaded preset: {preset_name}")
        print(f"[SAM3 PresetLoad]   Start frame (fwd): {start_frame_forward}, (bwd): {start_frame_backward}")
        print(f"[SAM3 PresetLoad]   Objects: {len(objects)}")
        print(f"[SAM3 PresetLoad]   Colors: {mask_colors}")

        return (positive_points, negative_points, start_frame_forward, start_frame_backward, mask_colors, video_path, preset_info)


class SAM3PresetBatchLoader:
    """
    Load multiple SAM3 presets for batch processing.

    Scans a directory for preset files and outputs them one at a time
    based on the batch_index parameter. Use with a loop node to process
    all presets automatically.

    Workflow:
    1. Save multiple presets using SAM3PresetSave
    2. Use this node with a loop to process all presets
    3. Connect batch_index to the loop counter
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Index of preset to load (0-based). Connect to loop counter."
                }),
            },
            "optional": {
                "input_directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Directory containing preset files. Leave empty for default (output/sam3_presets)"
                }),
                "file_pattern": ("STRING", {
                    "default": "*.json",
                    "multiline": False,
                    "tooltip": "Glob pattern to filter preset files (e.g., 'hockey_*.json')"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "INT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("positive_points", "start_frame", "mask_colors", "video_path", "total_presets", "current_preset_name")
    FUNCTION = "load_batch"
    CATEGORY = "SAM3/presets"

    @classmethod
    def IS_CHANGED(cls, batch_index, input_directory="", file_pattern="*.json"):
        # Always re-run to check for new presets
        return float("nan")

    def load_batch(self, batch_index: int, input_directory: str = "", file_pattern: str = "*.json"):
        """Load a specific preset by batch index."""
        import glob

        # Determine directory
        if input_directory and input_directory.strip():
            presets_dir = input_directory.strip()
        else:
            presets_dir = os.path.join(folder_paths.get_output_directory(), "sam3_presets")

        # Find preset files
        pattern = os.path.join(presets_dir, file_pattern)
        preset_files = sorted(glob.glob(pattern))

        total_presets = len(preset_files)

        if total_presets == 0:
            print(f"[SAM3 PresetBatchLoader] No presets found in: {presets_dir}")
            empty_prompt = {"objects": []}
            return (empty_prompt, 0, "", "", 0, "no_presets_found")

        # Clamp batch_index to valid range
        if batch_index >= total_presets:
            print(f"[SAM3 PresetBatchLoader] batch_index {batch_index} >= total {total_presets}, wrapping")
            batch_index = batch_index % total_presets

        preset_path = preset_files[batch_index]
        preset_name = os.path.splitext(os.path.basename(preset_path))[0]

        print(f"[SAM3 PresetBatchLoader] Loading preset {batch_index + 1}/{total_presets}: {preset_name}")

        # Load the preset
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
        except Exception as e:
            print(f"[SAM3 PresetBatchLoader] Error loading {preset_path}: {e}")
            empty_prompt = {"objects": []}
            return (empty_prompt, 0, "", "", total_presets, f"error_{preset_name}")

        # Extract data
        start_frame = preset.get("start_frame", 0)
        mask_colors = preset.get("mask_colors", "cyan, teal, purple, magenta")
        video_path = preset.get("video_path", "")
        objects = preset.get("objects", [])

        positive_points = {"objects": objects}

        print(f"[SAM3 PresetBatchLoader]   Start frame: {start_frame}, Objects: {len(objects)}")

        return (positive_points, start_frame, mask_colors, video_path, total_presets, preset_name)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3PresetSave": SAM3PresetSave,
    "SAM3PresetLoad": SAM3PresetLoad,
    "SAM3PresetBatchLoader": SAM3PresetBatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3PresetSave": "SAM3 Preset Save",
    "SAM3PresetLoad": "SAM3 Preset Load",
    "SAM3PresetBatchLoader": "SAM3 Preset Batch Loader",
}
