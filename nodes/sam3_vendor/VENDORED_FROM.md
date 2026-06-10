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

## Planned vendor: `nodes/sam3_lib_v31/` (SAM 3.1 Object Multiplex) — NOT YET VENDORED

- Source repo: https://github.com/facebookresearch/sam3 (main; SAM 3.1 released
  2026-03-27 per RELEASE_SAM3p1.md, no tag)
- Upstream commit: TBD at vendor time — record the full 40-char hash here AND in
  `nodes/sam3_lib_v31/VENDORED_FROM.yaml`.
- Required steps at vendor time (per migration plan, multi-AI R0 2026-06-10,
  archives `~/.multi-ai/results/20260610-134633/`):
  1. Internal import rewrite (`sam3.` → relative/namespaced) so both libs
     coexist in one process.
  2. Curated state-dict drift mitigation, strict=True (purge regenerable RoPE
     `freqs_cis_*` buffers ONLY after confirming non-persistent in upstream
     code; dot↔underscore attn remap; hard-fail diagnostics). See upstream
     issues #526/#506.
  3. Port applicable recovered fork fixes (check each against upstream main
     first — some may be fixed upstream).
  4. New sibling node classes with distinct registry keys (LoadSAM31Model,
     SAM31VideoSegmentation, ...). Never mutate the 3.0 classes.
  5. Shared node layer (MaskTracks/SelectMask/MultiFrameAddPrompt/cache ops)
     stays version-agnostic — 3.1-specific logic goes in v31 siblings only.
- Checkpoints: `facebook/sam3.1` HF (gated, fp32 `sam3.1_multiplex.pt`) or
  `Comfy-Org/sam3.1` (ungated, `sam3.1_multiplex_fp16.safetensors`). License:
  "SAM License" (code + weights).
