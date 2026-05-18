"""
SAM3MultiFrameAddPrompt (D-354) — batch sibling of SAM3AddPrompt.

Consumes SAM3_SEED_PROMPTS STRING JSON and applies each seed as a point
prompt onto the existing SAM3_VIDEO_STATE. Replaces N sequential
SAM3AddPrompt nodes for auto-seeding workflows.

Source-agnostic: accepts payloads from any producer that emits the
sam3_seed_prompts schema. Known producers in the NV_Comfy_Utils suite:
  - NV_DWPoseToSAM3Seeds (D-359, DWPose-driven, face + body)
  - NV_VLMToSAM3Seeds    (D-364, Gemini-VLM-driven, face + head + hair +
                          open-vocab)

Pipeline position:
  SAM3VideoSegmentation (frame 0 init) → SAM3MultiFrameAddPrompt (batch) →
  SAM3Propagate → SAM3MaskTracks → NV_MaskCoverageAnalyzer (closes loop)

Architecture (v6 spec, locked across 6 multi-AI rounds):
  - Pre-scan seeds JSON for duplicate (frame_idx, obj_id) → fail loud
  - Validate schema_version (major mismatch → raise; minor mismatch with
    schema_minor_compatible_with → warn; minor mismatch without → raise)
  - Iterate sorted seeds, fail-soft per seed (NaN/inf rejected; epsilon
    drift clamped to [0,1])
  - State-chaining: state_cur = state_cur.with_prompt(VideoPrompt.create_point(...))
  - skip_init_frame_dups (default True): silently de-dup collisions with
    existing video_state.prompts; raise loud when False
  - applied_count == 0 raises (chain broken)

Coordinate convention: SAM3_SEED_PROMPTS carries already-normalized [0,1]
coords (NV_DWPoseToSAM3Seeds adapter normalizes at the producer side).
VideoPrompt.create_point stores them as-is; SAM3's internal point encoder
expects normalized coords, matching SAM3PointCollector's pipeline.

See node_notes/architecture/2026-05-18_dwpose_to_sam3_seeds_v6_spec.md
for the locked spec.
"""

import json
import math
from typing import Any, Dict, List, Set, Tuple

from .video_state import VideoPrompt


_PREFIX = "[SAM3 MultiFrameAddPrompt]"

# Schema enforcement constants (D-354 consumer side)
SUPPORTED_MAJOR = 1
SUPPORTED_MINOR = 0
SCHEMA_TYPE_EXPECTED = "sam3_seed_prompts"
SUPPORTED_FAMILY = "1.x"

# Coordinate epsilon: drift within +/- this clamps to [0,1]; larger drift rejects.
COORD_EPSILON = 1e-3


def _parse_semver(s: str) -> Tuple[int, int, int]:
    """Parse '1.2.3' → (1, 2, 3). Raises ValueError on bad shape."""
    parts = s.split(".")
    if len(parts) != 3:
        raise ValueError(f"version '{s}' must be MAJOR.MINOR.PATCH")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise ValueError(f"version '{s}' has non-int component: {e}")


def _validate_schema_version(payload: Dict[str, Any]) -> None:
    """Validate schema_type + schema_version + minor compat per v6 spec.

    Rules:
      - schema_type must equal "sam3_seed_prompts"
      - major(payload) > supported_major → raise
      - major == supported_major AND minor(payload) > supported_minor:
          check schema_minor_compatible_with includes "1.x"
          if yes → log warning, proceed (ignore unknown fields)
          if no or absent → raise
    """
    schema_type = payload.get("schema_type")
    if schema_type != SCHEMA_TYPE_EXPECTED:
        raise ValueError(
            f"{_PREFIX} seed_prompts schema_type='{schema_type}' "
            f"(expected '{SCHEMA_TYPE_EXPECTED}'). Refusing to consume."
        )
    sver = payload.get("schema_version")
    if not isinstance(sver, str):
        raise ValueError(
            f"{_PREFIX} seed_prompts missing/non-string schema_version"
        )
    major, minor, patch = _parse_semver(sver)
    if major > SUPPORTED_MAJOR:
        raise ValueError(
            f"{_PREFIX} seed_prompts schema_version={sver} has major>{SUPPORTED_MAJOR}. "
            f"This SAM3MultiFrameAddPrompt only supports major=1.x. "
            f"Update the SAM3 nvFork package."
        )
    if major == SUPPORTED_MAJOR and minor > SUPPORTED_MINOR:
        compat = payload.get("schema_minor_compatible_with")
        if compat != SUPPORTED_FAMILY:
            raise ValueError(
                f"{_PREFIX} seed_prompts schema_version={sver} has minor "
                f">{SUPPORTED_MINOR} and schema_minor_compatible_with="
                f"{compat!r} does not match consumer family '{SUPPORTED_FAMILY}'. "
                f"Refusing to consume (forward-compatibility not guaranteed)."
            )
        print(
            f"{_PREFIX} WARN: seed_prompts schema_version={sver} is newer "
            f"than consumer's supported 1.{SUPPORTED_MINOR}.x. "
            f"schema_minor_compatible_with='{compat}' grants forward-compat — "
            f"unknown fields will be ignored."
        )


def _validate_required_keys(payload: Dict[str, Any]) -> None:
    """Top-level required keys (R3 narrow-validate).

    Source-agnostic: only fields the consumer actually reads downstream are
    required. Producer-specific provenance keys (e.g. dwpose_person_index
    from NV_DWPoseToSAM3Seeds) are accepted but not required, so VLM-sourced
    payloads from NV_VLMToSAM3Seeds (D-364) pass without emitting them.
    """
    required = ("schema_type", "schema_version", "schema_minor_compatible_with",
                "seeds", "accepted_frames")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"{_PREFIX} seed_prompts missing required top-level keys: {missing}"
        )


def _prescan_duplicates(seeds: List[Dict[str, Any]]) -> None:
    """Reject duplicate (frame_idx, obj_id) in the payload itself.
    Per v6 spec: payload malformation must raise BEFORE any state mutation.

    R3-code-review-r1 hardening: validate that frame_idx/obj_id are present
    and integer-coercible BEFORE the duplicate scan. Missing/malformed keys
    raise loud here (single source of truth) rather than crashing mid-loop.
    """
    seen: Set[Tuple[int, int]] = set()
    for i, s in enumerate(seeds):
        if not isinstance(s, dict):
            raise ValueError(
                f"{_PREFIX} seed_prompts.seeds[{i}] is not a dict: "
                f"got {type(s).__name__}"
            )
        for k in ("frame_idx", "obj_id"):
            if k not in s:
                raise ValueError(
                    f"{_PREFIX} seed_prompts.seeds[{i}] missing '{k}'"
                )
        try:
            frame_idx = int(s["frame_idx"])
            obj_id = int(s["obj_id"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{_PREFIX} seed_prompts.seeds[{i}] frame_idx/obj_id "
                f"not integer-coercible: {e}"
            )
        if isinstance(s["frame_idx"], bool) or isinstance(s["obj_id"], bool):
            raise ValueError(
                f"{_PREFIX} seed_prompts.seeds[{i}] frame_idx/obj_id "
                f"must not be bool"
            )
        if frame_idx < 0:
            raise ValueError(
                f"{_PREFIX} seed_prompts.seeds[{i}] has negative "
                f"frame_idx={frame_idx}"
            )
        key = (frame_idx, obj_id)
        if key in seen:
            raise ValueError(
                f"{_PREFIX} seed_prompts.seeds has duplicate "
                f"(frame_idx={key[0]}, obj_id={key[1]}) at index {i}. "
                f"Producer node must emit unique (frame_idx, obj_id) keys."
            )
        seen.add(key)


def _validate_and_clamp_coord(
    coord_pair: Any, seed_idx: int, fail_soft: bool
) -> Tuple[float, float, str]:
    """Validate [x, y] in normalized [0, 1] with epsilon tolerance.

    Returns (x_clamped, y_clamped, reason_or_ok).
    reason_or_ok: 'ok' / 'coord_nan_inf' / 'coord_out_of_range' / 'coord_not_numeric'

    On reject when fail_soft=False, raises immediately.

    R3-code-review-r1 hardening: wraps float() coercion in try/except so
    non-numeric coord members (e.g. ["x", 0.5]) don't bypass fail-soft.
    """
    if not isinstance(coord_pair, (list, tuple)) or len(coord_pair) != 2:
        n = len(coord_pair) if hasattr(coord_pair, "__len__") else "?"
        msg = (
            f"seed[{seed_idx}] coord pair must be 2-element list/tuple, "
            f"got {type(coord_pair).__name__} len={n}"
        )
        if not fail_soft:
            raise ValueError(f"{_PREFIX} {msg}")
        return (0.0, 0.0, "coord_out_of_range")
    # Reject bools explicitly (bool is subclass of int → float() accepts True/False)
    if isinstance(coord_pair[0], bool) or isinstance(coord_pair[1], bool):
        msg = f"seed[{seed_idx}] coord contains bool: {coord_pair}"
        if not fail_soft:
            raise ValueError(f"{_PREFIX} {msg}")
        return (0.0, 0.0, "coord_not_numeric")
    try:
        x, y = float(coord_pair[0]), float(coord_pair[1])
    except (TypeError, ValueError) as e:
        msg = f"seed[{seed_idx}] coord not numeric: {coord_pair} ({e})"
        if not fail_soft:
            raise ValueError(f"{_PREFIX} {msg}")
        return (0.0, 0.0, "coord_not_numeric")
    if not (math.isfinite(x) and math.isfinite(y)):
        msg = f"seed[{seed_idx}] non-finite coord ({x}, {y})"
        if not fail_soft:
            raise ValueError(f"{_PREFIX} {msg}")
        return (0.0, 0.0, "coord_nan_inf")
    # Range check with epsilon tolerance for floating drift
    lo, hi = -COORD_EPSILON, 1.0 + COORD_EPSILON
    if not (lo <= x <= hi and lo <= y <= hi):
        msg = f"seed[{seed_idx}] coord ({x}, {y}) outside [-{COORD_EPSILON}, 1+{COORD_EPSILON}]"
        if not fail_soft:
            raise ValueError(f"{_PREFIX} {msg}")
        return (0.0, 0.0, "coord_out_of_range")
    # Epsilon drift clamp into [0, 1]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return (x, y, "ok")


class SAM3MultiFrameAddPrompt:
    """D-354 — Batch apply N DWPose-derived point prompts onto SAM3_VIDEO_STATE.

    Consumes SAM3_SEED_PROMPTS STRING JSON from NV_DWPoseToSAM3Seeds (or any
    producer emitting schema_type='sam3_seed_prompts'). Iterates the seeds
    array in deterministic order, applying each as a point prompt via
    VideoPrompt.create_point. Returns the chained video_state plus applied
    count + skipped seeds log.

    Drop-in replacement for chaining N SAM3AddPrompt nodes when the prompts
    are auto-generated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Initialized state from SAM3VideoSegmentation."
                }),
                "seed_prompts": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "STRING JSON from NV_DWPoseToSAM3Seeds. "
                        "schema_type='sam3_seed_prompts'."
                    ),
                }),
            },
            "optional": {
                "skip_init_frame_dups": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True (default), silently skip any seed whose "
                        "(frame_idx, obj_id) already has a prompt on "
                        "video_state from SAM3VideoSegmentation init. "
                        "When False, collision raises loud."
                    ),
                }),
                "fail_soft_per_seed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True (default), log + skip seeds with "
                        "malformed coords (NaN, inf, out-of-range). "
                        "When False, first malformed coord raises."
                    ),
                }),
                "error_on_noop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True (default), raise loud if 0 seeds applied "
                        "(prevents silent failure where SAM3 propagation runs "
                        "with no seeds — wastes minutes of compute). "
                        "When False, return unchanged video_state + "
                        "applied_count=0 + log; useful for 2-pass refinement "
                        "loops where an empty flagged_frames pass is benign."
                    ),
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Per-seed application logging.",
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE", "INT", "STRING")
    RETURN_NAMES = ("video_state", "applied_seed_count", "skipped_seeds_log")
    FUNCTION = "add_prompts"
    CATEGORY = "SAM3/prompts"
    DESCRIPTION = (
        "D-354 — Batch apply N DWPose-derived point prompts onto "
        "SAM3_VIDEO_STATE. Consumes SAM3_SEED_PROMPTS JSON from "
        "NV_DWPoseToSAM3Seeds. Replaces N chained SAM3AddPrompt nodes for "
        "auto-seeded workflows."
    )

    def add_prompts(
        self,
        video_state,
        seed_prompts: str,
        skip_init_frame_dups: bool = True,
        fail_soft_per_seed: bool = True,
        error_on_noop: bool = True,
        verbose_debug: bool = False,
    ):
        # ----- Parse + validate -----
        if not isinstance(seed_prompts, str) or not seed_prompts.strip():
            raise ValueError(
                f"{_PREFIX} seed_prompts must be non-empty STRING JSON"
            )
        try:
            payload = json.loads(seed_prompts)
        except json.JSONDecodeError as e:
            raise ValueError(f"{_PREFIX} seed_prompts invalid JSON: {e}")
        if not isinstance(payload, dict):
            raise ValueError(
                f"{_PREFIX} seed_prompts root must be JSON object, "
                f"got {type(payload).__name__}"
            )
        _validate_required_keys(payload)
        _validate_schema_version(payload)

        seeds = payload.get("seeds", [])
        if not isinstance(seeds, list):
            raise ValueError(
                f"{_PREFIX} seed_prompts.seeds must be a list, "
                f"got {type(seeds).__name__}"
            )

        # Pre-scan: reject duplicate (frame_idx, obj_id) BEFORE mutation.
        _prescan_duplicates(seeds)

        # ----- Build collision set from existing video_state.prompts -----
        collision_set: Set[Tuple[int, int]] = set()
        if hasattr(video_state, "prompts"):
            for p in video_state.prompts:
                # Only point prompts; other prompt types don't collide for this op
                if getattr(p, "prompt_type", None) == "point":
                    collision_set.add((int(p.frame_idx), int(p.obj_id)))

        num_frames = getattr(video_state, "num_frames", None)

        # A3-integration-audit fix: stream-mismatch warning. If payload's
        # total_frames exceeds video_state.num_frames, the producer was
        # likely wired to a different image stream than the SAM3 init —
        # normalized coords remain syntactically valid but may be
        # semantically off. Warn but don't raise (out-of-range seeds will
        # be soft-skipped per frame anyway).
        payload_total = payload.get("total_frames")
        if (
            isinstance(payload_total, int)
            and num_frames is not None
            and payload_total > num_frames
        ):
            print(
                f"{_PREFIX} WARN: payload total_frames={payload_total} > "
                f"video_state.num_frames={num_frames}. Likely wiring mismatch "
                f"(producer's image_for_dimensions stream differs from "
                f"SAM3VideoSegmentation's video stream). Out-of-range seeds "
                f"will be soft-skipped, but coordinate semantics may also "
                f"be off if frame dimensions differ. Verify both producer "
                f"and SAM3 init consume the SAME image batch."
            )

        if verbose_debug:
            print(
                f"{_PREFIX} payload accepted_frames={payload.get('accepted_frames')}, "
                f"seeds={len(seeds)}, existing point prompts={len(collision_set)}, "
                f"num_frames={num_frames}"
            )

        # ----- Sort seeds deterministically: frame_idx asc → obj_id asc -----
        sorted_seeds = sorted(
            seeds, key=lambda s: (int(s["frame_idx"]), int(s["obj_id"]))
        )

        applied_count = 0
        skipped: List[Dict[str, Any]] = []
        state_cur = video_state

        for i, seed in enumerate(sorted_seeds):
            # frame_idx/obj_id presence + integer-coercibility + non-negativity
            # validated at _prescan_duplicates upstream; safe to coerce here.
            frame_idx = int(seed["frame_idx"])
            obj_id = int(seed["obj_id"])

            # num_frames bounds check
            if num_frames is not None and frame_idx >= num_frames:
                if not fail_soft_per_seed:
                    raise ValueError(
                        f"{_PREFIX} seed[{i}] frame_idx={frame_idx} >= "
                        f"video_state.num_frames={num_frames}"
                    )
                skipped.append({
                    "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                    "reason": "frame_idx_out_of_range",
                    "num_frames": num_frames,
                })
                continue

            # Collision check
            collision_key = (frame_idx, obj_id)
            if collision_key in collision_set:
                if skip_init_frame_dups:
                    skipped.append({
                        "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                        "reason": "duplicate_existing_prompt",
                    })
                    if verbose_debug:
                        print(
                            f"{_PREFIX} skip seed[{i}] dup (frame={frame_idx}, "
                            f"obj_id={obj_id})"
                        )
                    continue
                else:
                    raise ValueError(
                        f"{_PREFIX} seed[{i}] collides with existing prompt "
                        f"at (frame={frame_idx}, obj_id={obj_id}) and "
                        f"skip_init_frame_dups=False"
                    )

            # Coord validation + epsilon clamp
            pos_pts_raw = seed.get("pos_pts", [])
            neg_pts_raw = seed.get("neg_pts", [])
            if not isinstance(pos_pts_raw, list):
                if not fail_soft_per_seed:
                    raise ValueError(
                        f"{_PREFIX} seed[{i}].pos_pts must be a list"
                    )
                skipped.append({
                    "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                    "reason": "pos_pts_not_list",
                })
                continue
            # neg_pts container guard (R3-code-review-r2 fix): if producer emits
            # None/scalar/dict/string, iterating directly would crash outside
            # the fail-soft path. Mirror the pos_pts guard.
            if not isinstance(neg_pts_raw, list):
                if not fail_soft_per_seed:
                    raise ValueError(
                        f"{_PREFIX} seed[{i}].neg_pts must be a list "
                        f"(or omitted), got {type(neg_pts_raw).__name__}"
                    )
                skipped.append({
                    "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                    "reason": "neg_pts_not_list",
                })
                continue

            # Build flattened (points, labels) lists
            all_points: List[Tuple[float, float]] = []
            all_labels: List[int] = []
            coord_rejected = False
            coord_reject_reason = ""
            for pt in pos_pts_raw:
                x, y, reason = _validate_and_clamp_coord(pt, i, fail_soft_per_seed)
                if reason != "ok":
                    coord_rejected = True
                    coord_reject_reason = reason
                    break
                all_points.append((x, y))
                all_labels.append(1)
            if coord_rejected:
                skipped.append({
                    "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                    "reason": coord_reject_reason, "field": "pos_pts",
                })
                continue
            for pt in neg_pts_raw:
                x, y, reason = _validate_and_clamp_coord(pt, i, fail_soft_per_seed)
                if reason != "ok":
                    coord_rejected = True
                    coord_reject_reason = reason
                    break
                all_points.append((x, y))
                all_labels.append(0)
            if coord_rejected:
                skipped.append({
                    "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                    "reason": coord_reject_reason, "field": "neg_pts",
                })
                continue

            if not all_points:
                # No usable points → skip (a SAM3 prompt needs at least 1 point)
                skipped.append({
                    "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                    "reason": "no_points_in_seed",
                })
                continue

            # Construct + chain
            try:
                prompt = VideoPrompt.create_point(
                    frame_idx, obj_id,
                    [list(p) for p in all_points],
                    list(all_labels),
                )
                state_cur = state_cur.with_prompt(prompt)
            except Exception as e:
                if not fail_soft_per_seed:
                    raise
                skipped.append({
                    "seed_index": i, "frame_idx": frame_idx, "obj_id": obj_id,
                    "reason": "with_prompt_failed",
                    "detail": str(e),
                })
                continue

            collision_set.add(collision_key)
            applied_count += 1
            if verbose_debug:
                print(
                    f"{_PREFIX} applied seed[{i}] frame={frame_idx} "
                    f"obj_id={obj_id} pos={len([l for l in all_labels if l == 1])} "
                    f"neg={len([l for l in all_labels if l == 0])}"
                )

        if applied_count == 0:
            if error_on_noop:
                raise RuntimeError(
                    f"{_PREFIX} no seeds applied (input had {len(sorted_seeds)} "
                    f"seeds, all skipped). Chain is broken — check "
                    f"skipped_seeds_log for reasons. Skipped: {skipped!r}. "
                    f"Set error_on_noop=False to allow benign no-op (e.g. 2-pass "
                    f"refinement where flagged_frames produced 0 seedable frames)."
                )
            print(
                f"{_PREFIX} WARN: applied_count=0 (all {len(sorted_seeds)} "
                f"seeds skipped); returning unchanged video_state per "
                f"error_on_noop=False. Skipped reasons: "
                f"{[s.get('reason') for s in skipped]}"
            )

        skipped_log_json = json.dumps(skipped)
        print(
            f"{_PREFIX} Applied {applied_count} seeds, skipped {len(skipped)}. "
            f"New total prompts: {len(getattr(state_cur, 'prompts', []))}"
        )

        return (state_cur, applied_count, skipped_log_json)


NODE_CLASS_MAPPINGS = {
    "SAM3MultiFrameAddPrompt": SAM3MultiFrameAddPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3MultiFrameAddPrompt": "SAM3 Multi-Frame Add Prompt",
}
