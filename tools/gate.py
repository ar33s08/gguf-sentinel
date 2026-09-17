#!/usr/bin/env python3
"""Compile-gate: byte-compile + import-check every python file in the repo.

This build environment has a demonstrated token-mangling write channel (a brace
or underscore can silently vanish). Human/model eyes do not catch it; the
compiler does. Run after every batch of writes:

    python3 tools/gate.py          # compile every .py, report first errors
    python3 tools/gate.py --imports # additionally import every sentinel module

Exit 0 = every file on disk is syntactically valid as-written.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def compile_all():
    failures = []
    checked = 0
    for path in sorted(ROOT.rglob("*.py")):
        if ".venv" in path.parts or "vendor" in path.parts:
            continue
        checked += 1
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    return checked, failures


def import_all():
    failures = []
    sys.path.insert(0, str(ROOT))
    import importlib.util
    for name in ("sentinel.registry", "sentinel.findings", "sentinel.reader",
                 "sentinel.parser", "sentinel.rules", "sentinel.report",
                 "sentinel.generate", "sentinel.fuzz", "sentinel.cli"):
        try:
            importlib.util.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def main(argv):
    checked, failures = compile_all()
    for line in failures:
        print("SYNTAX", line)
    print(f"compile: {checked - len(failures)}/{checked} ok")
    if failures:
        return 1
    if "--imports" in argv:
        ifail = import_all()
        for line in ifail:
            print("IMPORT", line)
        print(f"imports: {'ok' if not ifail else 'FAILED'}")
        return 1 if ifail else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
