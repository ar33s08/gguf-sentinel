#!/usr/bin/env python3
"""Run every tests/test_*.py without pytest: import, call each test_*, report.
These plain-assert functions also run under CI's pytest unchanged."""
import importlib
import os
import sys
import traceback

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
for _p in (ROOT, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    passed = failed = 0
    for fname in sorted(os.listdir(TESTS)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        mod = importlib.import_module(fname[:-3])
        for name in sorted(dir(mod)):
            fn = getattr(mod, name, None)
            if not (name.startswith("test_") and callable(fn)):
                continue
            try:
                fn()
                passed += 1
            except BaseException:
                failed += 1
                print(f"FAIL {fname}::{name}")
                traceback.print_last()
    print(f"tests: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
