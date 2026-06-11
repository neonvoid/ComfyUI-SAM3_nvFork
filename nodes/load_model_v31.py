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
                f"{_PREFIX} SAM 3.1 multiplex requires CUDA - the upstream builder "
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

        # --- Node-layer compatibility shims -------------------------------------
        # Make the upstream multiplex predictor satisfy the contract that
        # inference_reconstructor.py + sam3_video_nodes.py expect (session-registry
        # name alias + start_session kwarg-filtering). See sam3_v31_compat.py.
        from .sam3_v31_compat import apply_node_layer_compat
        apply_node_layer_compat(predictor)

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

        # --- Clean, copy-paste-ready LOAD SUMMARY ------------------------------
        # The upstream builder prints a wrapped `Missing keys (N): [...]` line
        # above; that's expected (RoPE freqs_cis buffers, regenerated). This
        # block is the authoritative one-glance verdict — paste THIS, ignore the
        # raw scroll. For deeper state, drop an "NV SAM3 Diagnostics" node.
        def _has(name):
            return hasattr(predictor, name)
        api = [a for a in ("start_session", "add_prompt", "add_new_mask",
                           "propagate_in_video", "close_session",
                           "handle_stream_request")
               if _has(a)]
        api_missing = [a for a in ("add_new_mask",) if not _has(a)]
        size_mb = unified.model_size() / 1024 / 1024
        summary = "\n".join([
            "", "=" * 10 + " [SAM31] LOAD SUMMARY (copy below) " + "=" * 10,
            f"version        : SAM 3.1 multiplex",
            f"checkpoint     : {ckpt.name} ({ckpt.suffix})",
            f"predictor      : {type(predictor).__name__}",
            f"model          : {type(predictor.model).__name__}",
            f"multiplex/maxobj: {multiplex_count}/{max_num_objects}  fa3={use_fa3}  compile={torch_compile}",
            f"session alias  : {'installed' if _has('_ALL_INFERENCE_STATES') else 'MISSING (propagate will crash)'}",
            f"image processor: {'ready' if processor is not None else 'unavailable (video path OK)'}",
            f"predictor API  : {' '.join(api)}",
            f"API gaps       : {(' '.join(api_missing)) or 'none on smoke path'}  "
            f"(add_new_mask only needed for chunked/mask prompts)",
            f"state-dict keys: see upstream 'Missing keys' line above - expected RoPE "
            f"freqs_cis buffers (regenerated, benign)",
            f"size / device  : {size_mb:.1f} MB / {load_device}",
            "=" * 54, "",
        ])
        try:
            print(summary)
        except Exception:  # noqa: BLE001 — never let a console-encoding issue fail the load
            print(summary.encode("ascii", "replace").decode("ascii"))
        return (unified,)


class LoadSAM31TrackerModel(LoadSAM31Model):
    """Load the SAM 3.1 TRACKER (PVS) predictor -- the interactive point/box path.

    SAM 3 ships TWO inference contracts (paper arXiv:2511.16719 sec 2-3 + HF
    facebook/sam3 model card):
      - PCS (Promptable Concept Segmentation) = the MULTIPLEX predictor
        (LoadSAM31Model). TEXT/exemplar driven: detects + tracks ALL instances of
        a concept. Its VG state machine (hotstart / new_det_thresh / assoc_iou /
        masklet confirmation) SUPPRESSES objects not confirmed by recurring text
        detections -- so purely point-prompted objects vanish off prompt frames
        (field-verified 2026-06-11: out_obj_ids=(0,) on every bare frame).
      - PVS (Promptable Visual Segmentation) = THIS loader. SAM2-style dense
        tracker (Sam3VideoPredictor -> Sam3VideoInferenceWithInstanceInteractivity):
        interactive point/box objects are first-class and propagate across the
        whole video. This is the same contract as the proven 3.0 workflow, with
        the v31 code line.

    Use THIS loader for the seed-driven multi-region flow (NV_SAM3SeedBuilder
    points -> SAM3VideoSegmentation -> SAM3MultiFrameAddPrompt -> SAM3Propagate).
    Use LoadSAM31Model (multiplex) for text-concept detect-and-track.

    Checkpoint: the dense tracker uses the BASE SAM3 weights (sam3.pt family),
    NOT sam3.1_multiplex.pt (multiplex-specific memory-encoder weights).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "models/sam3/sam3.pt",
                    "tooltip": (
                        "Path to the BASE SAM3 checkpoint (sam3.pt / safetensors). "
                        "Relative paths resolve against the ComfyUI base dir. Do NOT "
                        "point this at sam3.1_multiplex.pt -- the multiplex weights "
                        "belong to LoadSAM31Model (PCS)."
                    ),
                }),
            },
            "optional": {
                "strict_state_dict": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Strict checkpoint key matching. If the load fails with "
                        "missing/unexpected keys (minor upstream drift between the "
                        "3.0 checkpoint and the v31 code), set False and check the "
                        "LOAD SUMMARY for what was skipped."
                    ),
                }),
                "torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "torch.compile the model (slow first run; leave off while validating).",
                }),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"
    DESCRIPTION = (
        "Load SAM 3.1 TRACKER (PVS) -- the SAM2-style interactive point/box video "
        "tracker. Point-prompted objects propagate across the whole video (unlike "
        "the multiplex/PCS loader, whose concept state machine suppresses them). "
        "Pair with NV_SAM3SeedBuilder for automated multi-region tracking."
    )

    def load_model(self, model_path, strict_state_dict=True, torch_compile=False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{_PREFIX} SAM 3.1 tracker requires CUDA - the upstream builder "
                f"moves the model to GPU unconditionally. No CUDA device is available."
            )

        ckpt = self._resolve_path(model_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{_PREFIX} Checkpoint not found: {ckpt}\n"
                f"This loader wants the BASE SAM3 weights (sam3.pt), the same file "
                f"the 3.0 LoadSAM3Model uses. Place it at <ComfyUI>/models/sam3/ or "
                f"give an absolute path."
            )
        if "multiplex" in ckpt.name.lower():
            raise ValueError(
                f"{_PREFIX} {ckpt.name} looks like the MULTIPLEX (PCS) checkpoint. "
                f"The tracker (PVS) loader needs the base sam3.pt weights -- the "
                f"multiplex memory-encoder keys will not match the dense tracker."
            )
        ckpt = self._ensure_pt(ckpt)

        # Lazy import (same discipline as the multiplex loader).
        from .sam3_lib_v31.model.sam3_video_predictor import Sam3VideoPredictor

        bpe_path = Path(__file__).parent / "sam3_lib_v31" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        print(f"{_PREFIX} Building SAM 3.1 TRACKER (PVS / dense) predictor "
              f"(strict={strict_state_dict}, compile={torch_compile})")
        print(f"{_PREFIX} Checkpoint: {ckpt}")

        predictor = Sam3VideoPredictor(
            checkpoint_path=str(ckpt),
            bpe_path=str(bpe_path),
            strict_state_dict_loading=strict_state_dict,
            compile=torch_compile,
        )

        # Same node-layer shims as the multiplex loader (registry alias +
        # start_session kwarg-filter). Each shim no-ops where the predictor
        # already satisfies the contract.
        from .sam3_v31_compat import apply_node_layer_compat
        apply_node_layer_compat(predictor)

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
            print(f"{_PREFIX} WARN: image-mode Sam3Processor unavailable "
                  f"({type(e).__name__}: {e}). Video tracking nodes unaffected.")

        unified = SAM3UnifiedModel(
            video_predictor=predictor,
            processor=processor,
            load_device=load_device,
            offload_device=offload_device,
        )

        def _has(name):
            return hasattr(predictor, name)
        api = [a for a in ("start_session", "add_prompt", "add_new_mask",
                           "propagate_in_video", "close_session",
                           "handle_stream_request")
               if _has(a)]
        size_mb = unified.model_size() / 1024 / 1024
        summary = "\n".join([
            "", "=" * 10 + " [SAM31] LOAD SUMMARY (copy below) " + "=" * 10,
            f"version        : SAM 3.1 TRACKER (PVS / dense, point-box path)",
            f"checkpoint     : {ckpt.name} ({ckpt.suffix})",
            f"predictor      : {type(predictor).__name__}",
            f"model          : {type(predictor.model).__name__}",
            f"session alias  : {'installed' if _has('_ALL_INFERENCE_STATES') else 'MISSING (propagate will crash)'}",
            f"image processor: {'ready' if processor is not None else 'unavailable (video path OK)'}",
            f"predictor API  : {' '.join(api)}",
            f"size / device  : {size_mb:.1f} MB / {load_device}",
            "=" * 54, "",
        ])
        try:
            print(summary)
        except Exception:  # noqa: BLE001
            print(summary.encode("ascii", "replace").decode("ascii"))
        return (unified,)


NODE_CLASS_MAPPINGS = {
    "LoadSAM31Model": LoadSAM31Model,
    "LoadSAM31TrackerModel": LoadSAM31TrackerModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM31TrackerModel": "Load SAM 3.1 Tracker (PVS Points)",
    "LoadSAM31Model": "Load SAM 3.1 Model (Multiplex)",
}
