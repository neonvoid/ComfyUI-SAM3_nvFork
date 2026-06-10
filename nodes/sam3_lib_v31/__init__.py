# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
SAM 3.1 Object Multiplex library — vendored for ComfyUI-SAM3_nvFork.

Vendored from https://github.com/facebookresearch/sam3 at commit
8e451d5eb43c817b64ae7577fb7b9ae223db88a9 (2026-05-23, post SAM 3.1 release
2026-03-27). See VENDORED_FROM.yaml in this directory and
nodes/sam3_vendor/VENDORED_FROM.md for full provenance.

Internal absolute imports were rewritten `sam3.` -> `sam3_v31.` so this
package coexists in-process with the frozen 3.0-era `nodes/sam3_lib/`
(which uses relative imports). The sys.modules alias below makes those
absolute imports resolve to THIS package regardless of where ComfyUI
mounts the custom-node package. The alias MUST be registered before any
submodule import runs.
"""

import sys as _sys

# Register top-level alias FIRST — submodules import via `sam3_v31.*`.
_sys.modules.setdefault("sam3_v31", _sys.modules[__name__])

from .model_builder import (  # noqa: E402  (alias must precede this import)
    build_sam3_image_model,
    build_sam3_multiplex_video_predictor,
    build_sam3_predictor,
)

__version__ = "3.1-multiplex-8e451d5"

__all__ = [
    "build_sam3_image_model",
    "build_sam3_multiplex_video_predictor",
    "build_sam3_predictor",
]
