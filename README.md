# ComfyUI-SAM3 (nvFork)

Fork of [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) by **Andrea Pozzetti** — ComfyUI nodes for Meta's [SAM3](https://github.com/facebookresearch/sam3).

This fork adds video tracking stability fixes, memory leak patches, multi-object mask extraction, and batch processing.

For installation instructions, see the [original repo](https://github.com/PozzettiAndrea/ComfyUI-SAM3).

## Quick Start

### Image Segmentation
```
Load SAM3 Model → SAM3 Text Segmentation → masks, boxes, scores
                        ↑
                  image + "person"
```

### Video Tracking
```
Load SAM3 Model
     ↓
SAM3 Video Segmentation  ←  video frames + "person"
     ↓
SAM3 Propagate  (continuous_detection=True)
     ↓
NV SAM3 Mask Tracks  →  NV SAM3 Select Mask  →  MASK
     ↓
NV SAM3 Auto Cleanup
```

## Nodes

### Model Loading
| Node | Purpose |
|------|---------|
| Load SAM3 Model | Load model weights, auto-downloads from HuggingFace |

### Image Segmentation
| Node | Purpose |
|------|---------|
| SAM3 Text Segmentation | Find all instances matching a text prompt |
| SAM3 Point Segmentation | Segment using point/box coordinates |
| SAM3 Create Point / Create Box | Create individual geometric prompts |
| SAM3 Combine Points / Combine Boxes | Merge multiple prompts together |

### Video Tracking
| Node | Purpose |
|------|---------|
| SAM3 Video Segmentation | Initialize a video tracking session |
| SAM3 Add Prompt | Add prompts on later frames (corrections, new objects) |
| SAM3 Propagate | Run tracking across all frames |
| SAM3 Video Output | Extract combined mask (simple) |
| SAM3 Video Trim | Trim video to match early-exit mask output |
| SAM3 VRAM Estimator | Estimate memory requirements before running |
| SAM3 Mask Loader | Load masks saved to disk (stream-to-disk mode) |
| SAM3 Mask to Video | Create visualization overlay from masks |
| SAM3 Mask Refine | Refine existing masks by feeding them back into SAM3 |

### Mask Extraction & Processing (Fork additions)
| Node | Purpose |
|------|---------|
| NV SAM3 Mask Tracks | Extract per-object masks with metadata |
| NV SAM3 Select Mask | Select/combine specific objects by ID |
| NV SAM3 Batch Planner | Split objects into batches for downstream limits |
| NV SAM3 Video Segmenter | Extract video segment for a specific batch |
| NV SAM3 Progressive Batcher | Single-pass tracking with grouped batch output |

### Interactive
| Node | Purpose |
|------|---------|
| SAM3 Point Collector | Click-based point editor in ComfyUI UI |
| SAM3 BBox Collector | Draw bounding boxes in ComfyUI UI |

### Presets
| Node | Purpose |
|------|---------|
| SAM3 Preset Save / Load | Save and load prompt configurations |
| SAM3 Preset Batch Loader | Load presets for batch processing |

### Utilities
| Node | Purpose |
|------|---------|
| NV SAM3 Auto Cleanup | Auto-free sessions after mask extraction |
| NV SAM3 Cleanup (Force) | Force-clear all sessions and VRAM |
| NV SAM3 Session Info | Debug active sessions |

## What This Fork Adds

**Stability fixes:**
- Fixed memory leaks in video tracking (feature cache, non-cond frame outputs, inference sessions)
- Fixed object ID loss during propagation (caused mask flickering)
- Fixed color instability when objects leave/enter frame
- Fixed device mismatches when using CPU offloading
- Fixed NMS OOM on scenes with many detection candidates
- Fixed point+box prompt merging — points and boxes on the same object are now combined into a single API call (previously `clear_old_points=True` caused the box call to erase click points)
- Fixed bbox coordinate conversion in SAM3 Add Prompt (was passing center-format `[cx,cy,w,h]` raw instead of converting to corner-format `[x1,y1,x2,y2]`)
- Fixed `add_new_mask()` in video predictor wrapper — was calling `self.model.add_new_mask()` which doesn't exist on `Sam3VideoInferenceWithInstanceInteractivity`; now correctly routes through `self.model.tracker.add_new_mask()`

**New nodes:**
- Per-object mask extraction with metadata (NV SAM3 Mask Tracks)
- Object selection and combination (NV SAM3 Select Mask)
- Batch planning for downstream object limits (NV SAM3 Batch Planner)
- Progressive batch output (NV SAM3 Progressive Batcher)
- Automatic session cleanup (NV SAM3 Auto Cleanup)
- VRAM estimation, session debugging, preset save/load
- Mask refinement — feed first-pass masks back into SAM3 to refine and fill gaps (SAM3 Mask Refine)

**Architecture changes:**
- Immutable video state (no global mutable state between nodes)
- ComfyUI model management integration (ModelPatcher, load_models_gpu)
- Interrupt handling in all long-running loops

## Tips

### Combining points and boxes

You can plug both a **SAM3 Point Collector** and a **SAM3 BBox Collector** into the same **SAM3 Video Segmentation** (or **SAM3 Add Prompt**) node. The bbox acts as a region constraint while click points provide precise foreground/background guidance within that region. Internally, box corners are encoded as special point labels (`2`=top-left, `3`=bottom-right) and merged with your click points into a single prompt — matching how SAM2/SAM3 was trained.

### Mask Refinement (Two-Pass)

When the first pass gets ~85% of the mask right, feed the result back in with **SAM3 Mask Refine** to catch the rest:

```
SAM3 Video Segmentation → SAM3 Propagate → SAM3 Video Output → masks
                                                                  ↓
Load Video → video_frames → SAM3 Mask Refine ←── masks (+ optional corrective clicks)
                                 ↓
                          refined masks (union of original + SAM3 refinement)
```

Key parameters:
- `keyframe_interval`: Seed SAM3 with a mask every N frames (default 10). Lower = closer to original, higher = more freedom.
- `merge_mode`: `union` (default, adds regions), `replace`, or `intersect`
- Optional corrective points/boxes for manual fixes on any frame

### Multi-frame corrections

Chain **SAM3 Add Prompt** nodes after **SAM3 Video Segmentation** to add correction anchors on later frames where tracking drifts (occlusions, fast motion, re-entry):
```
SAM3 Video Segmentation (frame 0)
     → SAM3 Add Prompt (frame 100)
     → SAM3 Add Prompt (frame 200)
     → SAM3 Propagate
```

## Troubleshooting

### Nodes not appearing

If you see "running in pytest mode - skipping initialization" in logs:
```bash
# Windows
set SAM3_FORCE_INIT=1

# Linux/Mac
export SAM3_FORCE_INIT=1
```

### OOM on long videos

- Use `offload_video_to_cpu=True` (saves 1-2 GB)
- Cap tracked objects at 10-15
- Use chunked processing for videos over 600 frames
- Add **NV SAM3 Auto Cleanup** after mask extraction

## Credits

- **SAM3 model**: [Meta AI Research](https://github.com/facebookresearch/sam3)
- **Original ComfyUI integration**: [Andrea Pozzetti](https://github.com/PozzettiAndrea) — [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
- **Interactive point/box editor**: Adapted from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai (Apache 2.0)
- **This fork**: [neonvoid](https://github.com/neonvoid)
