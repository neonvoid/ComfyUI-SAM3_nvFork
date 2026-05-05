"""
Tests for the SAM3 cache safety layer in sam3_video_nodes.py.

Covers the three protections introduced to fix the
`Windows fatal exception: access violation` crash on cache-miss:
  1. Bounded LRU caches (size-limited, evict oldest on insert)
  2. Session-scoped auto-flush on session_uuid change
  3. Teardown-callback synchronization with C++ inference state lifecycle

We bypass `nodes/__init__.py` (which transitively requires a fully-set-up
ComfyUI runtime via comfy.model_patcher etc.) by registering `nodes` as a
namespace package and then importing the three relevant submodules directly
via importlib. The cache helpers depend only on stdlib + torch + the local
`video_state` module, so they're isolatable.
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_NODES_DIR = Path(__file__).parent.parent / "nodes"


# `sam3_video_nodes.py` transitively imports `comfy.model_patcher` (via
# `sam3_model_patcher.py`). The conftest mocks `comfy` and `comfy.utils`
# but not `comfy.model_patcher`. Augment here so the helper module loads.
def _augment_comfy_mock():
    if "comfy" not in sys.modules:
        sys.modules["comfy"] = MagicMock()
    if "comfy.model_patcher" not in sys.modules:
        mp = types.ModuleType("comfy.model_patcher")
        mp.ModelPatcher = MagicMock
        sys.modules["comfy.model_patcher"] = mp


def _bootstrap_nodes_package():
    """Register `nodes` as a real package in sys.modules and load only the
    submodules we need (video_state, inference_reconstructor, sam3_video_nodes)
    via importlib with proper __package__ wiring so relative imports work."""
    if "nodes" not in sys.modules:
        pkg = types.ModuleType("nodes")
        pkg.__path__ = [str(_NODES_DIR)]
        pkg.__package__ = "nodes"
        sys.modules["nodes"] = pkg

    def _load_submodule(name):
        full = f"nodes.{name}"
        if full in sys.modules:
            return sys.modules[full]
        path = _NODES_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(
            full, str(path), submodule_search_locations=None,
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "nodes"
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load in dependency order: video_state (no deps inside nodes) →
    # inference_reconstructor (deps on video_state) → sam3_model_patcher
    # (mocked away if it can't import) → sam3_video_nodes.
    _augment_comfy_mock()
    _load_submodule("video_state")
    ir_mod = _load_submodule("inference_reconstructor")
    # sam3_video_nodes imports sam3_model_patcher; pre-mock it as empty so
    # the import succeeds even if comfy.model_patcher mock isn't enough.
    if "nodes.sam3_model_patcher" not in sys.modules:
        smp = types.ModuleType("nodes.sam3_model_patcher")
        smp.SAM3ModelWrapper = MagicMock
        smp.SAM3ModelPatcher = MagicMock
        smp.__package__ = "nodes"
        sys.modules["nodes.sam3_model_patcher"] = smp
    svn_mod = _load_submodule("sam3_video_nodes")
    return svn_mod, ir_mod


@pytest.fixture(scope="module")
def modules():
    return _bootstrap_nodes_package()


@pytest.fixture(scope="module")
def svn(modules):
    return modules[0]


@pytest.fixture(scope="module")
def ir(modules):
    return modules[1]


# =============================================================================
# _BoundedSessionCache — bounded LRU
# =============================================================================

@pytest.mark.unit
def test_bounded_session_cache_lru_eviction(svn):
    """Insert > max_size entries — oldest must be evicted FIFO."""
    c = svn._BoundedSessionCache("test_lru", max_size=2)
    c["a"] = 1
    c["b"] = 2
    assert "a" in c and "b" in c
    c["c"] = 3
    assert "a" not in c
    assert "b" in c and "c" in c
    assert len(c) == 2


@pytest.mark.unit
def test_bounded_session_cache_getitem_marks_mru(svn):
    """Reading a key marks it most-recently-used; subsequent insert evicts
    the OTHER older key, not the one just read."""
    c = svn._BoundedSessionCache("test_mru", max_size=2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert "a" in c and "c" in c
    assert "b" not in c


@pytest.mark.unit
def test_bounded_session_cache_clear(svn):
    c = svn._BoundedSessionCache("test_clear", max_size=3)
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    assert "a" not in c


@pytest.mark.unit
def test_bounded_session_cache_max_size_validated(svn):
    """max_size < 1 must raise. Defends against accidental zero-size cache."""
    with pytest.raises(ValueError):
        svn._BoundedSessionCache("bad", max_size=0)


# =============================================================================
# Session-change auto-flush
# =============================================================================

@pytest.mark.unit
def test_session_change_flushes_all_caches(svn):
    """Different session_uuid at any entrypoint flushes every registered cache."""
    svn._CURRENT_SESSION_UUID["value"] = None
    seg_cache = svn.SAM3VideoSegmentation._cache
    prop_cache = svn.SAM3Propagate._cache
    seg_cache.clear()
    prop_cache.clear()
    seg_cache["seg_key"] = "seg_value"
    prop_cache["prop_key"] = "prop_value"
    assert len(seg_cache) > 0 and len(prop_cache) > 0

    svn._maybe_evict_on_session_change("session-uuid-A")
    assert len(seg_cache) > 0  # first call records, doesn't flush

    svn._maybe_evict_on_session_change("session-uuid-A")
    assert len(seg_cache) > 0  # same uuid, no flush

    svn._maybe_evict_on_session_change("session-uuid-B")
    assert len(seg_cache) == 0
    assert len(prop_cache) == 0


@pytest.mark.unit
def test_session_change_ignores_falsy_uuid(svn):
    """None/empty uuid must NOT trigger a flush — would cause false flushes
    on partial state."""
    svn._CURRENT_SESSION_UUID["value"] = "session-A"
    cache = svn.SAM3VideoSegmentation._cache
    cache.clear()
    cache["k"] = "v"
    svn._maybe_evict_on_session_change(None)
    svn._maybe_evict_on_session_change("")
    assert len(cache) == 1
    assert svn._CURRENT_SESSION_UUID["value"] == "session-A"


# =============================================================================
# Teardown callbacks (token-based registration, exception isolation)
# =============================================================================

@pytest.mark.unit
def test_teardown_callback_replaces_by_name(ir):
    """Re-register with same name replaces old callback (defends against
    importlib.reload accumulating stale callbacks)."""
    log = []

    def cb_v1(reason):
        log.append(("v1", reason))

    def cb_v2(reason):
        log.append(("v2", reason))

    ir.register_teardown_callback(cb_v1, name="test_replace")
    ir._fire_teardown_callbacks("first")
    assert log == [("v1", "first")]

    ir.register_teardown_callback(cb_v2, name="test_replace")
    log.clear()
    ir._fire_teardown_callbacks("second")
    assert log == [("v2", "second")]

    ir.unregister_teardown_callback("test_replace")
    log.clear()
    ir._fire_teardown_callbacks("third")
    assert log == []


@pytest.mark.unit
def test_teardown_callback_exception_isolated(ir):
    """One bad callback must not break the teardown sequence."""
    fired = []

    def cb_raises(reason):
        raise RuntimeError("intentional")

    def cb_ok(reason):
        fired.append(reason)

    ir.register_teardown_callback(cb_raises, name="test_iso_raises")
    ir.register_teardown_callback(cb_ok, name="test_iso_ok")
    try:
        ir._fire_teardown_callbacks("with_exception")
        assert fired == ["with_exception"]
    finally:
        ir.unregister_teardown_callback("test_iso_raises")
        ir.unregister_teardown_callback("test_iso_ok")


# =============================================================================
# _tensor_signature — collision discrimination
# =============================================================================

@pytest.mark.unit
def test_tensor_signature_discriminates_localized_edits(svn):
    """Mass + sample combo must distinguish a tiny localized edit from
    the original. This is the primary defense against false-hit cache
    collisions in SAM3MaskRefine (which has no session_uuid backstop)."""
    import torch
    a = torch.zeros(1, 256, 256, 3, dtype=torch.float32)
    sig_a = svn._tensor_signature(a)
    b = a.clone()
    b[0, 10:15, 10:15, :] = 1.0
    sig_b = svn._tensor_signature(b)
    assert sig_a != sig_b


@pytest.mark.unit
def test_tensor_signature_discriminates_layout_shift(svn):
    """Two tensors with same total mass but different spatial layout must
    have different signatures — sum() alone collides here."""
    import torch
    a = torch.zeros(1, 64, 64, 3)
    a[0, :32, :, :] = 1.0  # top half white
    b = torch.zeros(1, 64, 64, 3)
    b[0, 32:, :, :] = 1.0  # bottom half white (same total mass)
    assert svn._tensor_signature(a) != svn._tensor_signature(b)


@pytest.mark.unit
def test_tensor_signature_stable_across_reruns(svn):
    """Same content → same signature. Required for legitimate cache hits."""
    import torch
    torch.manual_seed(42)
    a = torch.rand(1, 64, 64, 3)
    sig1 = svn._tensor_signature(a)
    sig2 = svn._tensor_signature(a)
    sig3 = svn._tensor_signature(a.clone())
    assert sig1 == sig2 == sig3


@pytest.mark.unit
def test_tensor_signature_handles_empty_and_none(svn):
    import torch
    assert svn._tensor_signature(None) == ("none",)
    e = torch.zeros(0)
    sig = svn._tensor_signature(e)
    assert sig[3] == 0.0
    assert sig[4] == ()


@pytest.mark.unit
@pytest.mark.parametrize("dtype", ["bool", "uint8", "float16", "float32", "float64"])
def test_tensor_signature_handles_dtypes(svn, dtype):
    """Mass calculation must work for every SAM3 input/output dtype.
    Earlier `flat.float().sum()` form would silently allocate a fp32 copy
    of fp16 inputs (~2× VRAM spike); the current `sum(dtype=fp64)` form
    upcasts inside the kernel accumulator."""
    import torch
    t = torch.ones(64, dtype=getattr(torch, dtype))
    sig = svn._tensor_signature(t)
    assert not (isinstance(sig[-1], str) and sig[-1].startswith("err:")), \
        f"_tensor_signature failed for dtype {dtype}: {sig}"
    assert sig[3] == 64.0  # mass = sum of 64 ones


# =============================================================================
# _masks_signature & _scalar_or_str
# =============================================================================

@pytest.mark.unit
def test_masks_signature_empty(svn):
    assert svn._masks_signature(None) == ("empty",)
    assert svn._masks_signature({}) == ("empty",)


@pytest.mark.unit
def test_masks_signature_distinguishes_content(svn):
    """Two masks dicts with different content must produce different
    signatures (combined with the session_uuid + prompt repr in the
    actual cache key, this gives layered protection)."""
    import torch
    a = {0: torch.zeros(64, 64), 1: torch.zeros(64, 64)}
    b = {0: torch.zeros(64, 64), 1: torch.ones(64, 64)}  # different content
    assert svn._masks_signature(a) != svn._masks_signature(b)


@pytest.mark.unit
def test_scalar_or_str_no_truncation(svn):
    """Earlier versions truncated repr() at 128 chars, hiding trailing
    edits in long point prompts. Must NOT truncate now."""
    long_prompt = {"points": [[float(i), float(i + 1)] for i in range(50)]}
    repr_value = svn._scalar_or_str(long_prompt)
    assert len(repr_value) > 200
    assert repr(long_prompt) == repr_value
