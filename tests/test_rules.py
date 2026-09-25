"""Rule-level detection tests.

Every expected code string below was copied verbatim from sentinel/rules.py or
the CODE_SEVERITIES registry in sentinel/findings.py, and the mutation matrix
behaviour was measured (40 seeds per class), not assumed. Hand-built byte
streams come from tests/_bytes.py, typed straight from the vendored gguf.h."""
import struct

import _bytes as B
from _shared import STRONG, clean_model, codes_of, seed_hits


def _arch():
    return B.kv_str("general.architecture", "llama")


def test_clean_generated_model_is_silent():
    assert codes_of(clean_model()) == set()


def test_strong_mutation_classes_fire_their_codes():
    for kind, wanted in STRONG.items():
        assert seed_hits(kind, wanted) > 0, f"{kind} never produced {sorted(wanted)}"


def test_count_shrink_leaves_records_in_padding():
    m = bytearray(clean_model())
    m[8:16] = struct.pack("<Q", 1)  # declared tensor count shrunk, records remain
    assert "E_PADDING_NOT_ZERO" in codes_of(bytes(m))


def test_tokenizer_parallel_arrays_must_match():
    kvs = (_arch()
           + B.kv_arr_str("tokenizer.ggml.tokens", ["a", "b", "c"])
           + B.kv_arr_i32("tokenizer.ggml.scores", [1, 2]))
    got = codes_of(B.build(n_kv=3, n_tensors=0, kvs=kvs))
    assert "E_TOKENIZER_COUNT_MISMATCH" in got, got


def test_special_id_out_of_range():
    kvs = (_arch()
           + B.kv_arr_str("tokenizer.ggml.tokens", ["a", "b", "c"])
           + B.kv_u32("tokenizer.ggml.bos_token_id", 9999))
    got = codes_of(B.build(n_kv=3, n_tensors=0, kvs=kvs))
    assert "W_SPECIAL_TOKEN_IDS_OUT_OF_RANGE" in got, got


def test_chat_template_without_specials():
    kvs = _arch() + B.kv_str("tokenizer.chat_template", "hello world")
    got = codes_of(B.build(n_kv=2, n_tensors=0, kvs=kvs))
    assert "W_TEMPLATE_NO_SPECIAL_TOKENS" in got, got
