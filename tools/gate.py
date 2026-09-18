#!/usr/bin/env python3
"""Compiler-based gate for this repo.

The channel that writes files into this environment is lossy (an occasional
character drops mid-word: `values` -> `valus`). Semantic damage does not
change line counts, so line- and grep-based checks miss it; the compiler and
the runtime do not. This gate therefore trusts only executable evidence:

  stage 1: py_compile every repo python file (syntax)
  stage 2: import every sentinel module + run the generator/parser round-trip
  stage 3: full pytest suite

Anything non-zero stops the build.
"""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def stage_compile() -> list:
    fails = []
    for base, dirs, files in os.walk(ROOT):
        if "vendor" in base or ".git" in base:
            continue
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, "rb") as handle:
                    compile(handle.read(), path, "exec")
            except SyntaxError as exc:
                fails.append(f"SYNTAX {path}:{exc.lineno}: {exc.msg}")
    return fails


def stage_smoke() -> list:
    fails = []
    mods = ["sentinel", "sentinel.findings", "sentinel.reader", "sentinel.registry",
            "sentinel.parser", "sentinel.rules", "sentinel.report", "sentinel.generate",
            "sentinel.fuzz", "sentinel.cli"]
    for dotted in mods:
        try:
            importlib.util.import_module(dotted)
        except Exception as exc:
            fails.append(f"IMPORT {dotted}: {type(exc).__name__}: {exc}")
    if fails:
        return fails
    # behavioral: build a model, re-read it, check agreement (round-trip gate)
    try:
        from sentinel.generate import build_model, sample_config
        from sentinel.parser import parse_gguf
        from sentinel import rules as rulez
        buf = build_model(sample_config())
        model = parse_gguf(buf, filename="mem")
        report = rulez.analyze_model(model, buf)
        hard = [f for f in report if f.severity == "error"]
        if hard:
            fails.append("ROUNDTRIP: clean synthetic model produced errors: "
                         + ", ".join(sorted({f.code for f in hard})))
    except Exception as exc:
        fails.append(f"ROUNDTRIP: {type(exc).__name__}: {exc}")
    return fails


def stage_pytest() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py", top_level_dir=ROOT)
    return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful else 1


def main() -> int:
    fails = stage_compile()
    for line in fails:
        print(line)
    if fails:
        print(f"COMPILE FAIL ({len(fails)})")
        return 1
    print(f"compile: ok")
    fails = stage_smoke()
    for line in fails:
        print(line)
    if fails:
        print(f"SMOKE FAIL ({len(fails)})")
        return 1
    print("smoke: ok")
    rc = stage_pytest()
    if rc:
        print("PYTEST FAIL")
    else:
        print("pytest: ok")
    return rc


if __name__ == "__main__":
    sys.exit(main())
