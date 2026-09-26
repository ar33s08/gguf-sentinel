#!/usr/bin/env python3
"""Regenerate the demo fixtures byte-deterministically, and self-verify them:

    python3 tools/make_demo.py

Writes fixtures/demo.gguf (must scan clean) and fixtures/demo-broken.gguf
(the same model with token_embd.weight's type id corrupted -> E_BAD_TENSOR_TYPE).
Exits non-zero if either fixture stops behaving as documented, so the README
demo can never silently drift from reality."""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sentinel.generate import ModelSpec, build_model, mutate
from sentinel.cli import scan_bytes
from sentinel.findings import SentinelError

# same shape as the fuzzer's base model: small enough to commit, real enough
# to exercise the tensor-table and quant-table paths
SPEC = ModelSpec(n_block=2, n_head=2, n_kv_head=1, n_embed=64,
                 n_ffn=128, n_vocab=256, head_dim=32)


def main():
    fx = os.path.join(ROOT, "fixtures")
    os.makedirs(fx, exist_ok=True)
    clean = build_model(SPEC)
    broken = mutate(clean, "tensor_type", random.Random(1))

    _doc, fnd = scan_bytes(clean, filename="demo.gguf")
    assert fnd == (), f"clean fixture drifted: {[f.code for f in fnd]}"
    try:
        scan_bytes(broken, filename="demo-broken.gguf")
        codes = {f.code for f in scan_bytes(broken, filename="x")[1]}
    except SentinelError as exc:
        codes = {exc.code}
    assert "E_BAD_TENSOR_TYPE" in codes, f"broken fixture drifted: {codes}"

    open(os.path.join(fx, "demo.gguf"), "wb").write(clean)
    open(os.path.join(fx, "demo-broken.gguf"), "wb").write(broken)
    print(f"wrote fixtures/demo.gguf ({len(clean):,} bytes, scans clean)")
    print(f"wrote fixtures/demo-broken.gguf ({len(broken):,} bytes, E_BAD_TENSOR_TYPE)")


if __name__ == "__main__":
    raise SystemExit(main())
