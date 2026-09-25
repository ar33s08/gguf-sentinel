"""Shared harness for the gguf-sentinel test suite (pytest-free on purpose:
tests/run_all.py drives these same test_* functions, and CI runs them under
pytest as well)."""
import os
import random
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
for _p in (ROOT, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sentinel.generate import ModelSpec, build_model, mutate, MUTATIONS
from sentinel.cli import scan_bytes
from sentinel.findings import SentinelError

FUZZ_SPEC = lambda: ModelSpec(n_block=2, n_head=2, n_kv_head=1, n_embed=64,
                              n_ffn=128, n_vocab=256, head_dim=32)


def clean_model(seed=0xC0FFEE):
    return build_model(FUZZ_SPEC(), seed=seed)


def codes_of(buf):
    """All finding codes for a buffer; a fatal SentinelError folds into the
    set as its code (a coded rejection IS a detection, never a bug)."""
    try:
        _doc, fnd = scan_bytes(buf, filename="test")
    except SentinelError as exc:
        return {exc.code}
    except Exception as exc:
        raise AssertionError(f"unhandled crash {type(exc).__name__}: {exc}") from exc
    return {f.code for f in fnd}


def seed_hits(kind, wanted, blob=None, seeds=40):
    """Seeds (of `seeds`) whose mutation of `blob` produces a code in `wanted`."""
    blob = blob if blob is not None else clean_model()
    hits = 0
    for seed in range(seeds):
        mutated = mutate(blob, kind, random.Random(seed))
        if mutated == blob:
            continue
        if codes_of(mutated) & set(wanted):
            hits += 1
    return hits


# Corruption classes whose detector is proven: each kind MUST land on one of
# these codes (measured deterministic 40/40 behaviour, not hope).
STRONG = {
    "magic": {"E_MAGIC"},
    "version": {"E_VERSION"},
    "truncated_header": {"E_TRUNCATED_HEADER"},
    "tensor_type": {"E_BAD_TENSOR_TYPE"},
    "tensor_offset": {"E_TENSOR_PAST_EOF"},
    "append": {"W_TRAILING_JUNK"},
    "zero": {"E_VERSION", "W_NO_ALIGNMENT"},
    "tensor_dim": {"E_EMBED_ROWS_MISMATCH", "E_TENSOR_OVERLAP",
                   "E_TENSOR_PAST_EOF"},
}
