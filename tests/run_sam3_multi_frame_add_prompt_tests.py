"""Standalone test runner for SAM3MultiFrameAddPrompt (D-354).

Run from this tests/ dir:
  python run_sam3_multi_frame_add_prompt_tests.py
"""

import inspect
import sys
import traceback

sys.path.insert(0, ".")

import test_sam3_multi_frame_add_prompt as t


def _run(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception:
        print(f"FAIL: {name}")
        traceback.print_exc()
        return False


def _is_skipped(fn):
    marks = getattr(fn, "pytestmark", []) or []
    return any(getattr(m, "name", None) == "skip" for m in marks)


def main():
    tests = [
        (name, fn) for name, fn in inspect.getmembers(t, inspect.isfunction)
        if name.startswith("test_") and fn.__module__ == t.__name__
    ]
    tests.sort(key=lambda nf: inspect.getsourcelines(nf[1])[1])

    passed = 0
    failed = 0
    skipped = 0
    for name, fn in tests:
        if _is_skipped(fn):
            print(f"SKIP: {name}")
            skipped += 1
            continue
        if _run(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}\n  {passed} passed, {failed} failed, {skipped} skipped\n{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
