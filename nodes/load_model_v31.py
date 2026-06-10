"""
LoadSAM31Model node — loads the SAM 3.1 Object Multiplex model.

Sibling of LoadSAM3Model (load_model.py). The legacy node + nodes/sam3_lib
(3.0-era) are FROZEN and untouched; this node builds a predictor from the
vendored post-3.27 upstream snapshot in nodes/sam3_lib_v31 (see
VENDORED_FROM.yaml there). Output is the same SAM3_MODEL type — downstream
nodes (SAM3VideoSegmentation, SAM3AddPrompt, SAM3MultiFrameAddPrompt,
SAM3Propagate, NV_SAM3MaskTracks, ...) consume the predictor through the
version-agnostic session API (handle_request / handle_stream_request), so
they work with either loader.

Checkpoints:
  - Ungated fp16:  https://huggingface.co/Comfy-Org/sam3.1
        checkpoints/sam3.1_multiplex_fp16.safetensors  (~1.7 GB)
  - Gated fp32:    https://huggingface.co/facebook/sam3.1
        sam3.1_multiplex.pt  (requires approved access + HF token)
Place under <ComfyUI>/models/sam3/.

Known build behaviors (upstream, vendored as-is):
  - The upstream builder calls .cuda() unconditionally — a CUDA device is
    REQUIRED at load time.
  - Upstream loads the state dict with strict=False and prints missing /
    unexpected key counts to the console. Watch the console on first load —
    a large missing-key cascade means a checkpoint/code mismatch (cf.
    upstream issues #526/#506), not a corrupt file.
  - use_fa3=True requires the flash_attn_interface (FlashAttention 3)
    package, which is typically NOT available on Windows — default False.
"""
from pathlib import Path

import torch
import comfy.model_management
import comfy.utils
from folder_paths import base_path as comfy_base_path

from .load_model import SAM3UnifiedModel

_PREFIX = "[SAM31]"


class LoadSAM31Model:
    """Load SAM 3.1 Object Multiplex model (multi-object shared forward pass)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "models/sam3/sam3.1_multiplex_fp16.safetensors",
                    "tooltip": (
                        "Path to a SAM 3.1 multiplex checkpoint (relative to ComfyUI root or "
                        "absolute). Supports .safetensors (ungated Comfy-Org/sam3.1 fp16 — "
                        "converted to .pt once, cached next to the file) and .pt "
                        "(gated facebook/sam3.1 fp32). Do NOT point this at a 3.0 sam3.pt — "
                        "use the legacy (down)Load SAM3 Model node for that."
                    ),
                }),
                "max_num_objects": ("INT", {
                    "default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Maximum tracked objects in one session (SAM 3.1 multiplex).",
                }),
                "multiplex_count": ("INT", {
                    "default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": (
                        "Objects per multiplex bucket (shared forward pass group). "
                        "Upstream default 16. Leave at 16 unless experimenting."
                    ),
                }),
            },
            "optional": {
                "use_fa3": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Use FlashAttention 3 kernels. Requires the flash_attn_interface "
                        "package (typically unavailable on Windows). OFF = standard "
                        "attention path."
                    ),
                }),
                "torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Enable torch.compile on model components (~2x speedup per upstream, "
                        "Linux-oriented). First runs are slow while compiling. OFF for "
                        "first bring-up / debugging."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"
    DESCRIPTION = (
        "Loads the SAM 3.1 Object Multiplex model (vendored upstream snapshot, "
        "sam3_lib_v31). Same SAM3_MODEL output as the legacy loader — existing "
        "downstream SAM3 nodes work unchanged. Requires CUDA."
    )

    def _resolve_path(self, model_path: str) -> Path:
        p = Path(model_path)
        if not p.is_absolute():
            p = Path(comfy_base_path) / p
        return p

    def _ensure_pt(self, ckpt_path: Path) -> Path:
        """Upstream loads via torch.load only. Convert .safetensors once, cache as .pt."""
        if ckpt_path.suffix != ".safetensors":
            return ckpt_path
        converted = ckpt_path.with_suffix(".converted.pt")
        if converted.exists():
            print(f"{_PREFIX} Using cached safetensors->pt conversion: {converted}")
            return converted
        print(f"{_PREFIX} Converting safetensors -> pt (one-time): {ckpt_path}")
        state_dict = comfy.utils.load_torch_file(str(ckpt_path), safe_load=True)
        tmp = converted.with_suffix(".pt.tmp")
        torch.save(state_dict, str(tmp))
        tmp.replace(converted)  # atomic-ish publish; avoids half-written cache on crash
        print(f"{_PREFIX} Cached: {converted}")
        return converted

    def load_model(self, model_path, max_num_objects=16, multiplex_count=16,
                   use_fa3=False, torch_compile=False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{_PREFIX} SAM 3.1 multiplex requires CUDA — the upstream builder "
                f"moves the model to GPU unconditionally. No CUDA device is available."
            )

        ckpt = self._resolve_path(model_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{_PREFIX} Checkpoint not found: {ckpt}\n"
                f"Download options:\n"
                f"  - UNGATED fp16: https://huggingface.co/Comfy-Org/sam3.1 "
                f"(checkpoints/sam3.1_multiplex_fp16.safetensors)\n"
                f"  - gated fp32:   https://huggingface.co/facebook/sam3.1 "
                f"(sam3.1_multiplex.pt, requires approved access)\n"
                f"Place the file at <ComfyUI>/models/sam3/ or give an absolute path."
            )
        ckpt = self._ensure_pt(ckpt)

        # Lazy import — keeps ComfyUI startup unaffected and the legacy 3.0
        # path importable even if a v31 dep (e.g. pycocotools) is missing.
        from .sam3_lib_v31 import build_sam3_multiplex_video_predictor

        bpe_path = Path(__file__).parent / "sam3_lib_v31" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        print(f"{_PREFIX} Building SAM 3.1 multiplex predictor "
              f"(max_num_objects={max_num_objects}, multiplex_count={multiplex_count}, "
              f"use_fa3={use_fa3}, compile={torch_compile})")
        print(f"{_PREFIX} Checkpoint: {ckpt}")
        print(f"{_PREFIX} NOTE: upstream prints missing/unexpected state-dict keys below; "
              f"large cascades indicate checkpoint/code mismatch.")

        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(ckpt),
            bpe_path=str(bpe_path),
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            compile=torch_compile,
        )

        # --- Node-layer compat shim: session-registry alias ---------------------
        # nodes/inference_reconstructor.py drives the predictor through the fork's
        # legacy Sam3VideoPredictor API, which exposes the live session registry as
        # `_ALL_INFERENCE_STATES` (upper). Upstream Sam3BasePredictor stores the SAME
        # registry as the instance attr `_all_inference_states` (lower) and only ever
        # mutates it in place (start_session/close_session/clear — never reassigns
        # after __init__). Alias the two names to the SAME dict object so the
        # reconstructor's reads and the predictor's writes stay in sync. Without this:
        # `'SAM3UnifiedModel' object has no attribute '_ALL_INFERENCE_STATES'` at
        # propagate time. (add_new_mask + a few renamed config knobs remain TODO —
        # not on the text/point path; see VENDORED_FROM.md.)
        if hasattr(predictor, "_all_inference_states") and not hasattr(predictor, "_ALL_INFERENCE_STATES"):
            predictor._ALL_INFERENCE_STATES = predictor._all_inference_states
            print(f"{_PREFIX} Session-registry alias _ALL_INFERENCE_STATES -> _all_inference_states installed.")
        else:
            print(f"{_PREFIX} WARN: could not install session-registry alias "
                  f"(has _all={hasattr(predictor, '_all_inference_states')}, "
                  f"has _ALL={hasattr(predictor, '_ALL_INFERENCE_STATES')}). "
                  f"Propagation may fail in inference_reconstructor.")

        # Image-mode processor: best-effort. The multiplex detector may not be
        # drop-in compatible with Sam3Processor's image pipeline — video nodes
        # don't need it, so degrade gracefully rather than fail the load.
        processor = None
        try:
            from .sam3_lib_v31.model.sam3_image_processor import Sam3Processor
            processor = Sam3Processor(
                model=predictor.model.detector,
                resolution=1008,
                device=str(load_device),
                confidence_threshold=0.2,
            )
            print(f"{_PREFIX} Sam3Processor (image mode) created.")
        except Exception as e:  # noqa: BLE001 — diagnostic fallback, video path unaffected
            print(f"{_PREFIX} WARN: image-mode Sam3Processor unavailable for the "
                  f"multiplex detector ({type(e).__name__}: {e}). Video tracking "
                  f"nodes are unaffected; image segmentation nodes will not work "
                  f"with this loader yet.")

        unified = SAM3UnifiedModel(
            video_predictor=predictor,
            processor=processor,
            load_device=load_device,
            offload_device=offload_device,
        )
        print(f"{_PREFIX} SAM 3.1 multiplex model ready "
              f"(size: {unified.model_size() / 1024 / 1024:.1f} MB)")
        return (unified,)


NODE_CLASS_MAPPINGS = {
    "LoadSAM31Model": LoadSAM31Model,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM31Model": "Load SAM 3.1 Model (Multiplex)",
}
