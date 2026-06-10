# SAM3 Vendoring Provenance

This file is the canonical record of where every vendored SAM3 library snapshot
in this repo came from. **Any new `sam3_lib*` directory MUST get an entry here
with a full upstream commit hash before it is committed.** The legacy lib was
vendored without a pin and the base commit had to be recovered by exhaustive
diff bisection — do not repeat that.

## Legacy vendor: `nodes/sam3_lib/` (SAM 3.0-era, FROZEN)

- Source repo: https://github.com/facebookresearch/sam3
- Upstream base commit: **`d0b1b9d`** (2025-11-21, "Fix SA-CO benchmark link in README")
  - Recovered 2026-06-10 by diff-bisection across all 13 upstream commits in the
    Oct-2025→Jan-2026 window (matched-file diff minimized at 1,258 changed lines,
    tied with `84cc43b` 2025-11-19; intervening commits are docs-only).
  - Method: per-relpath diff of the 46 `.py` files present in both trees,
    excluding `train/` and `__pycache__`.
- Status: **FROZEN.** Do not modify. Old saved workflows depend on the SAM3
  (3.0) node classes backed by this snapshot. It CANNOT load
  `sam3.1_multiplex.pt` (multiplex classes/builders do not exist here).
- Fork-local delta vs base: `patches/recovered/fork_delta_vs_d0b1b9d.patch`
  (2,319 patch lines; 906 added/changed). Content profile:
  - ~129 lines device-mismatch fixes (largest: `model/sam3_video_inference.py`)
  - ~34 lines object-ID stability fixes
  - ~12 NMS + 4 OOM lines (the NMS OOM fix, `model/edt.py` + `sam3_video_inference.py`)
  - ~73 lines vendored-import adaptations (`model_builder.py` etc.)
  - Fork-only files (no upstream relpath at base): `sam3_video_predictor.py`
    (session-API wrapper), `model/masks_ops.py`
  - NOTE: delta includes BOTH PozzettiAndrea/ComfyUI-SAM3 vendoring adaptations
    AND nvFork fixes; they are not yet separated per-fix. Categorize before
    porting anything to a new vendor (check whether upstream main already fixed
    each item).

## Vendor: `nodes/sam3_lib_v31/` (SAM 3.1 Object Multiplex) — VENDORED 2026-06-10

- Source repo: https://github.com/facebookresearch/sam3 (main; SAM 3.1 released
  2026-03-27 per RELEASE_SAM3p1.md, no tag)
- Upstream commit: **`8e451d5eb43c817b64ae7577fb7b9ae223db88a9`** (2026-05-23 HEAD
  at vendor time). Full package vendored (5.1MB incl. train/ — model/ imports
  train.masks_ops + train.data.collator). Details in
  `nodes/sam3_lib_v31/VENDORED_FROM.yaml`.
- Local modifications (all recorded in the yaml):
  1. Import rewrite `sam3.` → `sam3_v31.` (sed, leading-whitespace-tolerant) +
     `sys.modules["sam3_v31"]` alias registered in `__init__.py` BEFORE any
     submodule import. Both libs coexist in one process (legacy uses relative
     imports; v31 uses the alias).
  2. `model/edt.py` replaced with the legacy fork's triton-guarded version
     (HAS_TRITON try/except + torch fallback `edt_triton`) — upstream's only
     drift since the legacy base was a `# pyre-unsafe` comment (re-applied).
     This was the ONLY module-level triton import on the multiplex hot path;
     perflib triton kernels are lazy (function-level imports).
  3. `pkg_resources.resource_filename` package arg → `"sam3_v31"` (3 sites,
     defensive — the loader always passes bpe_path explicitly).
- Loader: `nodes/load_model_v31.py` → `LoadSAM31Model` (registry-distinct
  sibling; legacy LoadSAM3Model untouched). Uses upstream
  `build_sam3_multiplex_video_predictor` AS-IS (its internal torch.load +
  internal→OSS key remap + strict=False with printed missing/unexpected keys).
  `.safetensors` supported via one-time convert-to-`.pt` cache (upstream is
  torch.load-only). CUDA required (upstream `.cuda()` hardcoded). use_fa3 +
  torch_compile surfaced as widgets, default OFF (Windows).
- Node-layer compat:
  - `sam3_video_nodes.py` obj_ids extraction made key-tolerant
    (`obj_ids` | `out_obj_ids`) — upstream emits `out_obj_ids`; without this,
    MaskTracks silently falls back to channel-index identity.
  - `sam3_v31_compat.py` `apply_node_layer_compat(predictor)` (called from
    `load_model_v31.py`) patches a freshly-built multiplex predictor IN PLACE
    with two additive shims (vendored code stays pristine). Both runtime-found
    2026-06-10 (model loads clean — 64 missing keys = regenerated RoPE freqs_cis
    buffers, benign — and crashes only in the node layer):
      1. **session-registry alias** `_ALL_INFERENCE_STATES` -> `_all_inference_states`.
         inference_reconstructor.py (D-239/D-240 cache-safety) reads the live
         session dict by the fork's legacy name (upper); upstream Sam3BasePredictor
         stores the SAME dict as `_all_inference_states` (lower), mutated in place.
         Fixes `'SAM3UnifiedModel' object has no attribute '_ALL_INFERENCE_STATES'`.
      2. **start_session kwarg-filter**. Upstream Sam3BasePredictor.start_session
         forwards `offload_state_to_cpu` (+ maybe `video_loader_type`) to
         `model.init_state()`, but multiplex init_state =
         (resource_path, offload_video_to_cpu, async_loading_frames, use_torchcodec,
         use_cv2, input_is_mp4) rejects them. Replaced with a version that filters
         init_state kwargs to the accepted set (mirrors upstream's own
         add_prompt/propagate filtering + fork-3.0 behavior of stashing
         offload_state_to_cpu on the state). Fixes
         `init_state() got an unexpected keyword argument 'offload_state_to_cpu'`.
- KNOWN DEFERRED GAPS (not on the text/point smoke path; fix when hit):
  1. `add_new_mask` — upstream Sam3BasePredictor does NOT expose it; the fork's
     reconstructor calls it for MASK-type prompts (chunked video continuation).
     Needs a multiplex mask-add path before chunked/mask-seeded workflows run.
  2. `_apply_config` config knobs: `score_threshold_detection`, `assoc_iou_thresh`,
     `det_nms_thresh`, `init_trk_keep_alive` appear renamed/relocated on the
     multiplex tracking model — setting them is harmless (no crash) but becomes a
     no-op, so those tuning levers don't bite on 3.1 yet. Reconcile names for
     quality tuning. Builder defaults (score_threshold_detection=0.4 etc.) govern
     meanwhile.
- Import smoke PASSED on Windows embedded python (no triton, no flash-attn):
  package import + multiplex predictor + builder + Sam3Processor.
- Recovered legacy fixes: NOT yet ported beyond edt.py (see
  patches/recovered/). Evaluate per-fix against this snapshot during runtime
  validation — device/NMS behavior may already be fixed upstream.
- Checkpoints: `facebook/sam3.1` HF (gated, fp32 `sam3.1_multiplex.pt`) or
  `Comfy-Org/sam3.1` (ungated, `sam3.1_multiplex_fp16.safetensors`). License:
  "SAM License" (code + weights; LICENSE copied into sam3_lib_v31/).
