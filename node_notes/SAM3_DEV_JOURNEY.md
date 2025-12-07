# SAM3 nvFork Development Journey

This document captures the development work done on the SAM3 nvFork ComfyUI nodes. Use this as context for future development sessions.

---

## Project Overview

**Repository**: `ComfyUI-SAM3_nvFork`
**Purpose**: Video object tracking and segmentation using SAM3 (Segment Anything Model 3)
**Key Files**:
- `nodes/sam3_video_nodes.py` - Main video tracking nodes
- `nodes/sam3_mask_tracks.py` - Mask extraction and batch planning nodes
- `nodes/sam3_lib/sam3_video_predictor.py` - Wrapper for SAM3 model
- `nodes/sam3_lib/model/sam3_video_inference.py` - Core SAM3 inference

---

## Bugs Fixed

### Phase 0: Value Unpacking Bug (CRITICAL)

**Symptom**: `ValueError: not enough values to unpack (expected 5, got 2)`

**Root Cause**: Incorrectly assumed `propagate_in_video` returned 5 values based on `sam3_tracking_predictor.py`, but the actual model used is `sam3_video_inference.py` which yields only 2 values:
```python
yield yield_frame_idx, postprocessed_out  # postprocessed_out is a dict with "out_binary_masks"
```

**Fix**: Reverted to original 2-value unpacking in `sam3_video_predictor.py`:
```python
for frame_idx, outputs in self.model.propagate_in_video(...):
    yield {"frame_index": frame_idx, "outputs": outputs}
```

**Location**: `nodes/sam3_lib/sam3_video_predictor.py` lines 295-324

---

### Phase 1: Mask Output Flickering

**Symptom**: Mask output from `SAM3VideoOutput` was a "flickering mess" - different objects appearing randomly across frames.

**Root Cause**: Object IDs were discarded during propagation!

The model returns:
- `out_binary_masks` - mask tensors at indices [0, 1, 2, ...]
- `obj_ids` - actual object IDs [1, 3, 7, ...] mapping to each index

But `sam3_video_nodes.py` only stored the masks:
```python
masks_dict[frame_idx] = mask  # obj_ids is LOST!
```

Then `SAM3VideoOutput` treated `obj_id` as an array index:
```python
output_mask = frame_mask[obj_id].float()  # WRONG! obj_id != array index
```

**Example of the bug**:
- Frame 0: objects [1,2,3,4,5] at indices [0,1,2,3,4]
- Frame 50: objects [1,3,7] at indices [0,1,2]
- User asks for obj_id=3:
  - Frame 0: gets mask[3] = object 4 (WRONG!)
  - Frame 50: obj_id=3 >= len(3), falls back to combined mask
- Result: **flickering between different objects**

**Fix**: Always output combined mask (all objects merged):
```python
# Always output combined mask (all objects merged)
frame_mask = combined_mask
```

**Location**: `nodes/sam3_video_nodes.py` SAM3VideoOutput.extract() method

---

## Features Implemented

### Phase 2: Early Exit When Objects Leave Frame

**User Story**:
1. User selects specific subjects with point editor on first frame
2. Runs propagation (forward only)
3. System tracks ONLY those user-selected objects
4. When ALL user-selected objects leave frame → stop processing
5. Return **only valid frames** (truncated output)

**New Parameters for SAM3Propagate**:
```python
"auto_exit_on_empty": ("BOOLEAN", {
    "default": False,
    "tooltip": "Stop propagation when user-selected objects leave frame"
}),
"exit_delay_seconds": ("FLOAT", {
    "default": 0.5,
    "min": 0.1,
    "max": 5.0,
    "tooltip": "Seconds of consecutive empty frames before stopping"
}),
"video_fps": ("FLOAT", {
    "default": 30.0,
    "min": 1.0,
    "max": 120.0,
    "tooltip": "Video framerate for exit delay calculation"
}),
```

**How It Works**:
1. Calculate threshold: `empty_frames_threshold = int(exit_delay_seconds * video_fps)`
2. In propagation loop, check if mask is empty: `mask.shape[0] == 0`
3. Track `consecutive_empty_frames` counter
4. When threshold reached, break loop
5. Truncate `masks_dict` and `scores_dict` to valid frames only
6. Output `track_info` JSON with metadata:
   ```json
   {
     "early_exit_enabled": true,
     "early_exit_triggered": true,
     "last_valid_frame": 429,
     "total_valid_frames": 430,
     "exit_delay_seconds": 0.5,
     "video_fps": 60.0,
     "empty_frames_threshold": 30
   }
   ```

**Location**: `nodes/sam3_video_nodes.py` SAM3Propagate.propagate()

---

### Phase 3: SAM3VideoTrim Node

**Purpose**: Complete endpoint node for early exit workflow - trims video, extracts masks, generates visualization.

**Inputs**:
| Input | Type | Required |
|-------|------|----------|
| `video_frames` | IMAGE | Yes |
| `masks` | SAM3_VIDEO_MASKS | Yes |
| `track_info` | STRING (JSON) | Yes |
| `scores` | SAM3_VIDEO_SCORES | No |
| `viz_alpha` | FLOAT (0.0-1.0) | No |

**Outputs**:
| Output | Type | Description |
|--------|------|-------------|
| `trimmed_frames` | IMAGE | Video sliced to valid range |
| `masks` | MASK | Combined mask tensor [N, H, W] |
| `visualization` | IMAGE | Colored overlay frames |
| `start_frame` | INT | Start index |
| `end_frame` | INT | End index |
| `total_frames` | INT | Frame count |

**Location**: `nodes/sam3_video_nodes.py` SAM3VideoTrim class

---

### Phase 4: VRAM Estimator Node

**Purpose**: Pre-check VRAM availability before processing to prevent OOM crashes.

**New Node**: `SAM3VRAMEstimator`

**Inputs**:
| Input | Type | Description |
|-------|------|-------------|
| `video_state` | SAM3_VIDEO_STATE | For resolution/frame count |
| `sam3_model` | SAM3_MODEL | To check if loaded |
| `safety_margin_gb` | FLOAT (default 1.0) | Buffer space |

**Outputs**:
| Output | Type | Description |
|--------|------|-------------|
| `max_frames` | INT | Estimated safe frame count |
| `available_vram_gb` | FLOAT | Free VRAM after model |
| `per_frame_mb` | FLOAT | Estimated MB per frame |
| `can_process_all` | BOOLEAN | True if all frames fit |
| `recommended_chunk_size` | INT | For chunked mode |

**Estimation Formula**:
```python
# Empirical per-frame costs (resolution-scaled)
pixels = H * W
frame_load_cost = pixels * 3 * 4        # ~5.7MB/frame at 720p
propagation_cost = pixels * 2.2         # ~2.2MB/frame accumulation
per_frame = frame_load_cost + propagation_cost
max_frames = available_vram / per_frame
```

**Location**: `nodes/sam3_video_nodes.py` SAM3VRAMEstimator class

---

### Phase 5: Color ID Stability Fix

**Symptom**: Object colors jump/swap when other objects leave frame.

**Root Cause**: Colors assigned by **array index** instead of **stable object ID**.

**Example of the bug**:
```
Frame 0:   objects [1,2,3,4] at indices [0,1,2,3] → colors [blue,red,green,yellow]
Frame 100: object 2 leaves
Frame 101: objects [1,3,4] at indices [0,1,2] → colors [blue,red,green]
           Object 3 now has index 1 → gets RED instead of GREEN!
```

**Fix**: Bundle obj_ids inside masks_dict for automatic color stability.

**Implementation**:

**Step 1**: Bundle obj_ids with mask during propagation:
```python
# In SAM3Propagate.propagate()
masks_dict[frame_idx] = {
    "mask": mask,
    "obj_ids": frame_obj_ids  # Bundled for automatic color stability
}
```

**Step 2**: Unpack bundled data in visualization nodes:
```python
# In SAM3VideoTrim and SAM3VideoOutput
mask_data = masks[frame_idx]
if isinstance(mask_data, dict):
    frame_mask = mask_data.get("mask")
    embedded_obj_ids = mask_data.get("obj_ids")
else:
    # Legacy format support
    frame_mask = mask_data
    embedded_obj_ids = None

# Use embedded obj_ids for stable colors
frame_obj_ids = embedded_obj_ids
```

**Step 3**: Use stable IDs for color:
```python
if frame_obj_ids is not None and obj_idx < len(frame_obj_ids):
    color_id = frame_obj_ids[obj_idx]  # Use stable ID
else:
    color_id = obj_idx  # Fallback to array index
color = COLORS[color_id % len(COLORS)]
```

**Benefit**: No manual wiring needed - colors are automatically stable!

**Locations**:
- `SAM3Propagate.propagate()` - Bundle obj_ids inside masks_dict
- `SAM3VideoTrim.trim_video()` - Unpack and use embedded obj_ids
- `SAM3VideoOutput.extract()` - Unpack and use embedded obj_ids

---

### Phase 6: Multi-Frame Prompting (SAM3AddPrompt)

**Purpose**: Add prompts on additional frames to improve tracking through challenging scenarios.

**Problem solved**: SAM3 can lose track of objects through:
- Occlusions (player behind goalie)
- Fast motion (collisions, quick direction changes)
- Similar appearances (same team uniforms)
- Frame exits/entries

**New Node**: `SAM3AddPrompt`

**Inputs**:
| Input | Type | Description |
|-------|------|-------------|
| `video_state` | SAM3_VIDEO_STATE | Existing state with prompts |
| `frame_idx` | INT | Frame to add correction prompts |
| `positive_points` | SAM3_POINTS_PROMPT | Points to reinforce tracking |
| `negative_points` | SAM3_POINTS_PROMPT | Points to exclude |
| `positive_boxes` | SAM3_BOXES_PROMPT | Boxes to reinforce |
| `obj_id` | INT | Object ID to reinforce (-1 = auto) |

**Workflow**:
```
SAM3VideoSegmentation (frame 0, select 5 players)
        ↓
    video_state
        ↓
SAM3AddPrompt (frame 100, re-click player 3 after occlusion)
        ↓
    video_state (2 prompt sets)
        ↓
SAM3AddPrompt (frame 200, re-click player 1 after collision)
        ↓
    video_state (3 prompt sets)
        ↓
SAM3Propagate (all prompts applied during propagation)
```

**Use Cases for Hockey**:
| Scenario | When to Add Prompt |
|----------|-------------------|
| Player behind goalie | After they emerge from occlusion |
| Board collision/pile-up | After players separate |
| Line change | When new players enter |
| Camera angle change | Help re-identify from new angle |

**Location**: `nodes/sam3_video_nodes.py` SAM3AddPrompt class

---

## Architecture Notes

### Data Flow

```
Load Video (600 frames)
    ↓
Points Editor → SAM3 Video Segmentation → SAM3 Propagate (early exit)
    ↓                                           ↓
    └──────────────→ SAM3 Video Trim ←──────────┘
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
        trimmed_frames  masks   visualization
```

### Key Types

| Type | Description |
|------|-------------|
| `SAM3_VIDEO_MASKS` | Dict[int, Tensor] - frame_idx → mask tensor |
| `SAM3_VIDEO_SCORES` | Dict[int, Tensor] - frame_idx → confidence scores |
| `SAM3_VIDEO_STATE` | Immutable state with session_uuid, prompts, temp_dir |
| `SAM3_MODEL` | Loaded SAM3 model wrapper |

### Model Internals

The SAM3 model uses memory-based tracking:
- `propagate_in_video()` yields `(frame_idx, outputs_dict)`
- `outputs_dict` contains:
  - `out_binary_masks`: Tensor [num_objects, H, W] - can be [0, H, W] when empty!
  - `out_obj_ids`: Array of object IDs (often discarded, hence flickering bug)
  - Other scores and metadata

### Empty Mask Detection

When all tracked objects leave frame:
```python
# Model returns empty tensors:
out_binary_masks.shape  # [0, H, W]  <- 0 objects!

# Detection logic:
mask_is_empty = (mask is None or
                (hasattr(mask, 'shape') and mask.shape[0] == 0) or
                (hasattr(mask, 'numel') and mask.numel() == 0))
```

---

## Existing Nodes (Reference)

### SAM3BatchPlanner
Groups objects into batches for downstream workflows with actor limits.
- Expects `track_info` from `SAM3MaskTracks` with per-object visibility data
- Noise filtering (min_visible_frames, min_visibility_ratio)
- Two modes: `by_stability` and `temporal`

### SAM3VideoSegmenter
Segments video and masks based on batch schedule from BatchPlanner.
- Extracts frame ranges per batch
- Creates colored visualization
- Outputs per-batch segments

### SAM3MaskTracks
Converts `SAM3_VIDEO_MASKS` dict to tensor `[N_frames, N_objects, H, W]`.
- Generates per-object track_info with first/last frame
- Different from Propagate's track_info (early exit metadata)

---

## Known Issues / Backlog

### Memory/OOM
- Class-level caches (`SAM3Propagate._cache`, `SAM3VideoOutput._cache`) grow unbounded
- Need LRU eviction
- Chunking mode doesn't actually free memory between chunks

### Code Cleanup
- Debug prints in `IS_CHANGED` methods should be removed/conditionalized
- Mask normalization logic is duplicated across nodes

### Future Features
- Backward propagation early-exit
- Make track_info formats consistent between Propagate and MaskTracks

---

## Testing Workflow

To test early exit feature:

1. Load a video where subjects exit partway through
2. Use Points Editor to select subjects on first frame
3. Connect to SAM3 Video Segmentation
4. Connect to SAM3 Propagate with:
   - `auto_exit_on_empty = True`
   - `exit_delay_seconds = 0.5`
   - `video_fps = <your video fps>`
5. Connect to SAM3 Video Trim with:
   - `video_frames` from Load Video
   - `masks` from Propagate
   - `track_info` from Propagate
6. Check outputs:
   - `track_info` should show `early_exit_triggered: true`
   - `total_frames` should be less than input video length
   - Visualization should show colored masks

---

## File Locations Quick Reference

| File | Key Classes/Functions |
|------|----------------------|
| `nodes/sam3_video_nodes.py` | SAM3VideoSegmentation, SAM3AddPrompt, SAM3Propagate, SAM3VideoOutput, SAM3VideoTrim, SAM3VRAMEstimator |
| `nodes/sam3_mask_tracks.py` | SAM3MaskTracks, SAM3BatchPlanner, SAM3VideoSegmenter |
| `nodes/sam3_lib/sam3_video_predictor.py` | Sam3VideoPredictor wrapper |
| `nodes/sam3_lib/model/sam3_video_inference.py` | Core model, `propagate_in_video()` |
| `nodes/video_state.py` | SAM3VideoState, VideoPrompt, VideoConfig |

---

*Last updated: December 2024*
