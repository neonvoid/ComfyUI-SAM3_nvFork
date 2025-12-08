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

### Phase 7: Memory Optimization (Trimming)

**Problem**: SAM3's memory grows unboundedly during long video propagation, causing OOM.

**Investigation**: Compared nvFork settings to Meta SAM3 defaults:

| Setting | Meta SAM3 | nvFork | Effect |
|---------|-----------|--------|--------|
| `num_maskmem` | 7 | 7 | 7-frame sliding window |
| `max_cond_frames_in_attn` | -1 (unlimited) | **4** | nvFork already optimized! |
| `trim_past_non_cond_mem_for_eval` | False | False→**True** | Now trimming old frames |
| `offload_output_to_cpu_for_eval` | False | False | Keep on GPU for speed |

**Key Insight**: The nvFork author already optimized `max_cond_frames_in_attn` from unlimited to 4. But `trim` was left off.

**What Trimming Does**:
```
WITHOUT trim (memory grows forever):
Frame 600 memory: [0, 1, 2, 3, ... 598, 599, 600]  ← 600+ frames!

WITH trim (bounded memory):
Frame 600 memory: [0, 100, 300] + [594-600]  ← Only anchors + 7-frame window!
                   ↑ conditioning frames (your prompts - KEPT)
                              ↑ recent 7 frames (KEPT)
```

**How Prompts Create Anchors**:
- SAM3AddPrompt on frame 100 → Frame 100 = conditioning frame (KEPT)
- SAM3AddPrompt on frame 300 → Frame 300 = conditioning frame (KEPT)
- Everything else beyond 7-frame window → DELETED

**Memory Savings**: ~1GB+ for 600-frame videos

**Change Made**:
```python
# model_builder.py line 449
trim_past_non_cond_mem_for_eval=True  # Was False
```

**Synergy with SAM3AddPrompt**:
1. Trimming cleans up old non-conditioning frames
2. SAM3AddPrompt creates conditioning frames (anchors)
3. Anchors survive trimming, providing recovery points
4. Perfect for hockey: add prompts after occlusions, line changes, etc.

**Location**: `nodes/sam3_lib/model_builder.py` line 449

---

### Phase 8: Stop/Cancel Support

**Problem**: Pressing ComfyUI's stop button during SAM3 processing had no effect - nodes would continue processing until completion.

**Root Cause**: Long-running loops didn't check ComfyUI's interrupt flag.

**Solution**: Add `comfy.model_management.throw_exception_if_processing_interrupted()` at the start of each major loop iteration.

**ComfyUI Interrupt Mechanism**:
- User clicks Cancel → sets global `interrupt_processing = True`
- `throw_exception_if_processing_interrupted()` raises `InterruptProcessingException`
- Exception propagates to ComfyUI's execution handler for graceful stop

**Loops Updated** (in `nodes/sam3_video_nodes.py`):

| Method | Loop Purpose |
|--------|--------------|
| `_propagate_chunk()` | Chunk streaming inference |
| `_propagate_range_detection()` | Range detection inference |
| `_propagate_streaming()` | Stream-to-disk inference |
| `propagate()` | Standard propagation |
| `SAM3VideoOutput.extract()` | Frame visualization |
| `SAM3MaskToVideo` | Mask-to-video conversion |

**Location**: `nodes/sam3_video_nodes.py` - 6 lines added

---

### Phase 9: Feature Cache Memory Optimization

**Problem**: OOM still occurring at frame 1214/1454 despite Phase 7 memory trimming. VRAM grew from 42.55GB to 46GB.

**Root Cause**: Phase 7 fixed the **memory bank** but OOM was happening in **feature cache** (`_prepare_backbone_feats`). The feature cache cleanup only removed 1 previous frame - not aggressive enough for 1400+ frame videos.

**Changes Made**:

1. **Enable CPU offloading** - `model_builder.py:460`
```python
offload_output_to_cpu_for_eval=True,  # Was False
```
Moves mask outputs to CPU RAM instead of VRAM. No quality impact, slight speed reduction.

2. **Aggressive feature cache cleanup** - `sam3_video_base.py:400`
```python
# OLD: only removes 1 previous frame
feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)

# NEW: remove ALL frames outside 2-frame window
frames_to_remove = [f for f in feature_cache.keys()
                   if isinstance(f, int) and abs(f - frame_idx) > 2]
for f in frames_to_remove:
    feature_cache.pop(f, None)
```
SAM3 only needs current frame's backbone features. Old frames lingering was a memory leak.

3. **Fix CPU offload bug** - `sam3_tracker_base.py:1059-1068`

When `offload_output_to_cpu_for_eval=True`, the `trimmed_out` dict was missing `maskmem_features` and `maskmem_pos_enc` keys when `run_mem_encoder=False`, causing KeyError in `sam3_tracking_predictor.py:1113`.

```python
# Added default None values to trimmed_out:
trimmed_out = {
    ...
    "maskmem_features": None,  # ADDED
    "maskmem_pos_enc": None,   # ADDED
}
```

**Quality Impact**: None. SAM3 uses memory bank for tracking, not feature cache. Feature cache is just temporary backbone computation.

**Locations**:
- `nodes/sam3_lib/model_builder.py` line 460
- `nodes/sam3_lib/model/sam3_video_base.py` line 400
- `nodes/sam3_lib/model/sam3_tracker_base.py` line 1065-1068

---

### Phase 9.1: Fix OOM with Many Objects (AutoTrack Regression)

**Problem**: OOM at frame 571/1000 when tracking 20 objects from AutoTrack.

**Root Cause**: Phase 9 introduced a regression! Enabling `offload_output_to_cpu_for_eval=True` broke the far-old-frame cleanup logic.

In `sam3_tracker_base.py:1104-1106`:
```python
# BEFORE (broken condition):
if (
    self.use_memory_selection and not self.offload_output_to_cpu_for_eval
):  ## design for memory selection, trim too old frames to save memory
```

When `offload_output_to_cpu_for_eval=True` (set in Phase 9), this condition is ALWAYS `False`, so the far-old-frame cleanup at `frame_idx - 20 * max_obj_ptrs_in_encoder` (320 frames back) is **completely skipped**.

**Fix**: Remove the `not self.offload_output_to_cpu_for_eval` check:
```python
# AFTER:
if self.use_memory_selection:
    # Trim very old frames regardless of CPU offloading to prevent OOM
```

This ensures old frames are still cleaned up even with CPU offloading enabled.

**Why Memory Scales with Objects**: With 20 tracked objects:
- Each frame stores 20× mask data
- `non_cond_frame_outputs` grows much faster
- Without far-old-frame cleanup, memory accumulates until OOM

**Location**: `nodes/sam3_lib/model/sam3_tracker_base.py` line 1104-1105

**Tip**: AutoTrack has a `max_objects` parameter (default -1 = unlimited). For memory-constrained setups, consider limiting to 10-15 objects.

---

### Phase 10: AutoTrack Continuous Detection Mode

**Problem**: Objects entering the scene AFTER frame 0 weren't being tracked. User reported "new hockey players that enter the scene werent being masked at all".

**Root Cause**: AutoTrack's keyframe-based box prompts were being converted to tracker points, but `per_frame_geometric_prompt` wasn't set. During propagation:
- `has_geometric_prompt = False` for frames without box prompts
- `allow_new_detections = False` → detector couldn't create new objects
- Objects at later keyframes existed in tracker metadata but weren't tracked

**Solution**: Added `continuous_detection` mode (default True) that uses SAM3's native text prompt feature:

```python
# When continuous_detection=True:
prompt = VideoPrompt.create_text(frame_idx=0, obj_id=1, text=text_prompt)
video_state = video_state.with_prompt(prompt)
```

This sets `inference_state["text_prompt"]` which makes `has_text_prompt=True` for ALL frames during propagation. SAM3 then runs detection on every frame, automatically detecting and tracking all matching objects.

**New Parameter** in `SAM3AutoTrack`:
- `continuous_detection`: BOOLEAN, default True
  - `True`: Uses SAM3's native text detection on every frame (detects new objects entering scene)
  - `False`: Original keyframe-only detection (faster but may miss objects entering later)

**Key code paths**:
- `sam3_video_inference.py:903-905` - text prompt sets `inference_state["text_prompt"]`
- `sam3_video_inference.py:395` - `has_text_prompt` checked every frame
- `sam3_video_inference.py:423` - `allow_new_detections=True` when text prompt set

**Trade-offs**:
- PRO: Detects ALL matching objects throughout video, including those entering later
- PRO: No more "missing objects" issue
- CON: Runs detection every frame (slower than keyframe-only)
- CON: May detect more objects than intended (configure confidence_threshold to limit)

**Location**: `nodes/sam3_auto_track.py` lines 143-146, 264-293

---

### Phase 10.1: Fix Device Mismatch in cal_mem_score

**Problem**: RuntimeError during propagation with text prompts:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Root Cause**: Phase 9's CPU offloading moves some tensors to CPU via `trimmed_out.cpu()`. When `remove_object` is called during propagation (to remove low-confidence detections), `cal_mem_score` tries to multiply `object_score_logits` (on CPU) with `iou_score` (on GPU).

**Call path**:
1. `sam3_video_base.py:937` - `_tracker_remove_objects()` called
2. `sam3_tracking_predictor.py:1286` - `_slice_state` calls `cal_mem_score()`
3. `sam3_tracker_base.py:522` - tensor multiplication fails

**Fix**: Ensure both tensors are on the same device before multiplication:
```python
def cal_mem_score(self, object_score_logits, iou_score):
    # Ensure tensors are on the same device (handle CPU offloading)
    device = object_score_logits.device
    iou_score = iou_score.to(device)
    ...
```

**Location**: `nodes/sam3_lib/model/sam3_tracker_base.py` line 516-519

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
