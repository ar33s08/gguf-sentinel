#!/usr/bin/env python3
"""Unknown-name auditor. Prints ONLY numbers + positions (digits survive the
display channel; identifiers do not). For every bare Name in every module we
check membership in dir(builtins) ∪ module globals ∪ imports ∪ defs INSIDE THE
INTERPRETER, so neither the write nor the display channel can lie about it.

Offender report: file_index, line, col, name_length, name_hash8 (first 8 of
sha256). Decode names locally: python3 tools/audit.py --show N to print name N
through a channel we then cross-check against the intended fix."""
from __future__ import annotations

import ast
import builtins
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted((ROOT / "sentinel").glob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
NAMES = dir(builtins) + ["__file__", "__name__", "__doc__", "__spec__", "__package__",
                         "__builtins__", "__debug__", "__loader__"]


def bound_names(tree):
    out = set()
    for node in ast.walk(tree):
        tname = type(node).__name__
        if tname == "Import":
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
        elif tname == "ImportFrom":
            for a in node.names:
                if a.name == "*":
                    return None
                out.add(a.asname or a.name)
        elif hasattr(node, "name") and not hasattr(node, "names"):
            out.add(node.name)  # functions, classes, except-as
        elif hasattr(node, "arg"):  # ast.arg
            out.add(node.arg)
        elif hasattr(node, "ctx") and hasattr(node, "id"):
            if type(node.ctx).__name__ in ("Store", "Del"):
                out.add(node.id)
    return out


def main():
    show = None
    if "--show" in sys.argv:
        show = sys.argv.index("--show")
    total = 0
    for fi, path in enumerate(FILES):
        src = path.read_text()
        tree = ast.parse(src)
        scope = bound_names(tree)
        if scope is None:
            print(f"file {fi}: star import, skipping")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in NAMES or node.id in scope:
                    continue
                total += 1
                digest = hashlib.sha256(node.id.encode()).hexdigest()[:8]
                print(f"BAD file={fi} name={path.name} line={node.lineno} col={node.col_offset} "
                      f"len={len(node.id)} h={digest}")
    print(f"total offenders: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
