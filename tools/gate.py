#!/usr/bin/env python3
"""Compile + import + behaviour gate for the whole package.

The file-writing channel in this workspace can drop or duplicate a single
character inside a word ("range" -> "range"). A syntax check finds half of
that damage; an import finds more; behavioural probes find the rest. This
script runs all three layers and exits non-zero on anything, so CI and I use
the exact same gate. Run: python3 tools/gate.py
"""
from __future__ import annotations

import importlib
import pathlib
import py_compile
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULES = ("_type_table", "findings", "registry", "reader", "keys", "parser",
           "generate", "rules", "report", "fuzz", "cli")


def fail(label, exc):
    print(f"GATE FAIL [{label}]: {type(exc).__name__}: {exc}")
    raise SystemExit(1)


def layer_compile():
    for path in sorted((ROOT / "sentinel").glob("*.py")):
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            fail("compile " + path.name, exc)
    print("layer compile: ok")


def layer_import():
    for name in MODULES:
        try:
            importlib.import_module("sentinel." + name)
        except BaseException as exc:   # catch token-damaged LookupError subclasses too
            fail("import " + name, exc)
    print("layer import: ok")


def layer_probe():
    from sentinel import findings as findings_mod
    from sentinel import registry as registry_mod

    # findings
    try:
        findings_mod.rank_of("nope")
        raise AssertionError("rank_of must reject unknown severity")
    except findings_mod.UnknownSeverity:
        pass
    except LookupError as exc:  # NameError rides this base when the raise is damaged
        fail("findings.rank_of raise-path", exc)
    f = findings_mod.make("E_MAGIC", "x", offset=0)
    assert f.severity == "error" and f.key_id()[0] == "E_MAGIC"
    assert findings_mod.counts([f]) == {"error": 1, "warn": 0, "info": 0}

    # registry: upstream-pinned quant math
    assert registry_mod.row_bytes(2, 2048) == (2048 // 32) * 18      # q4_0
    assert registry_mod.row_bytes(2, 33) is None                       # block-misaligned
    q8k = registry_mod.typeinfo_by_name("q8_K")
    assert q8k is not None and q8k.blck == 256 and q8k.size == 292
    assert registry_mod.typeinfo(9999) is None
    print("layer probe: ok")


def layer_roundtrip():
    from sentinel.generate import MUTATIONS, ModelSpec, build_model, mutate
    from sentinel.parser import parse_gguf, tensor_bytes

    spec = ModelSpec(n_block=2, n_head=2, n_kv_head=1, n_embed=64,
                     n_ffn=128, n_vocab=256, head_dim=32)
    buf = build_model(spec)
    doc = parse_gguf(buf)
    if doc.kv_get("general.architecture") != "llama":
        fail("roundtrip arch", AssertionError(str(doc.kv_get("general.architecture"))))
    for t in doc.tensors:
        if tensor_bytes(t) is None:
            fail("roundtrip sizing", AssertionError(t.name))
        if t.offset % 32 != 0:
            fail("roundtrip align", AssertionError(t.name))
    if doc.data_start % doc.alignment != 0:
        fail("roundtrip data align", AssertionError(str(doc.data_start)))

    crashes = []
    per_kind = {}
    for kind in MUTATIONS:
        detected = 0
        for seed in range(40):
            mutated = mutate(buf, kind, random.Random(seed))
            if mutated == buf:
                continue
            try:
                d2 = parse_gguf(mutated)
            except BaseException as exc:
                if type(exc).__name__ == "SentinelError" or getattr(exc, "code", None):
                    detected += 1
                    continue
                crashes.append((kind, seed, type(exc).__name__, str(exc)[:120]))
                break
            from sentinel.rules import analyze
            flags = analyze(d2, mutated)
            if any(f.severity in ("error", "warn") for f in flags):
                detected += 1
        per_kind[kind] = detected
    if crashes:
        fail("mutation crash", crashes[0])
    undetected = [k for k, d in per_kind.items() if d == 0]
    if undetected:
        fail("mutation slipped through every seed", undetected)
    print(f"layer roundtrip: ok (every mutation kind detected; min per-kind "
          f"detections {min(per_kind.values())}/40)")


def main():
    layer_compile()
    layer_import()
    layer_probe()
    layer_roundtrip()
    print("GATE GREEN")


if __name__ == "__main__":
    main()
