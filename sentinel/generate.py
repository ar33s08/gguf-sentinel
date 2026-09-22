"""Synthetic GGUF builder + deterministic mutation engine.

This is the substrate for three things that make the tool trustworthy:
  * the pytest suite (build a known-good model, assert clean scan),
  * the fixture pack in fixtures/ (checked-in small-but-real files),
  * sentinel.fuzz (deliberate corruption to prove the parser never crashes).

The writer emits a format-accurate file from first principles (layout pinned
to the vendored gguf.h/ggml.h), independent of tests/_bytes.py on purpose: if
writer and reader shared a misreading of the format, round-trips would hide
it, so two independently typed encoders must agree for tests to pass.
"""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from typing import Optional

from . import registry

MAGIC = b"GGUF"
VERSION = 3
ALIGN = 32

# gguf_type discriminants (from vendored gguf.h enum gguf_type)
GU8, GI8, GU16, GI16, GU32, GI32, GF32, GBOOL, GSTRING, GARRAY, GU64, GI64, GF64 = range(13)


# ---------------------------------------------------------------- encoding
def enc_string(text: str) -> bytes:
    raw = text.encode("utf8") if isinstance(text, str) else text
    return struct.pack("<Q", len(raw)) + raw


def enc_key(key: str) -> bytes:
    return enc_string(key)


def enc_scalar_pair(key: str, gguf_type: int, fmt: str, value) -> bytes:
    return enc_string(key) + struct.pack("<I", gguf_type) + struct.pack(fmt, value)


def enc_bool_pair(key: str, value: bool) -> bytes:
    return enc_string(key) + struct.pack("<I", GBOOL) + struct.pack("<B", 1 if value else 0)


def enc_string_pair(key: str, value: str) -> bytes:
    return enc_string(key) + struct.pack("<I", GSTRING) + enc_string(value)


def enc_array_pair(key: str, elem_type: int, fmt, values) -> bytes:
    body = enc_string(key) + struct.pack("<I", GARRAY) + struct.pack("<I", elem_type)
    body += struct.pack("<Q", len(values))
    if elem_type == GSTRING:
        for v in values:
            body += enc_string(v)
    elif fmt == "e":  # half-floats arrive as raw u16 bit patterns
        for v in values:
            body += struct.pack("<H", v)
    else:
        for v in values:
            body += struct.pack(fmt, v)
    return body


def enc_tensor(name: str, dims, type_id: int, offset: int) -> bytes:
    rec = enc_string(name) + struct.pack("<I", len(dims))
    for d in dims:
        rec += struct.pack("<Q", d)
    rec += struct.pack("<I", type_id) + struct.pack("<Q", offset)
    return rec


def align_up(pos: int, alignment: int = ALIGN) -> int:
    rem = pos % alignment
    return pos if rem == 0 else pos + (alignment - rem)


# ---------------------------------------------------------------- model spec
@dataclass
class ModelSpec:
    arch: str = "llama"
    n_vocab: int = 256
    n_embed: int = 64
    n_head: int = 2
    n_kv_head: int = 1
    n_ffn: int = 128
    n_block: int = 2
    head_dim: int = 32
    ctx: int = 2048
    weight_type: int = 2          # q4_0 for the bulk matrices
    embed_type: int = 1          # f16 for embeddings
    quant_version: int = 2

    def qkv_rows(self) -> int:
        return (self.n_head + 2 * self.n_kv_head) * self.head_dim

    def out_rows(self) -> int:
        return self.n_embed


def _fill(seed: int, n: int) -> bytes:
    """Deterministic pseudorandom weight bytes (never all-zero, never all-one:
    rules flag degenerate weight regions, so clean fixtures must avoid them)."""
    rng = random.Random(seed)
    out = bytearray(rng.randbytes(n))
    if n >= 3 and out[0] == out[1] == out[2]:
        out[0] ^= 0x5A
    return bytes(out)


def _tensor_layout(spec: ModelSpec):
    """Return (name, dims, type_id) tuples in write order."""
    e = spec.n_embed
    tensors = [
        ("token_embd.weight", [spec.n_vocab, e, 1, 1], spec.embed_type),
        ("token_embd_norm.weight", [e, 1, 1, 1], 0),
        ("output.weight", [e, spec.n_vocab, 1, 1], spec.embed_type),
        ("output_norm.weight", [e, 1, 1, 1], 0),
    ]
    for b in range(spec.n_block):
        p = f"blk.{b}."
        tensors += [
            (p + "attn_norm.weight", [e, 1, 1, 1], 0),
            (p + "attn_qkv.weight", [spec.qkv_rows(), e, 1, 1], spec.weight_type),
            (p + "attn_out.weight", [e, spec.n_head * spec.head_dim, 1, 1], spec.weight_type),
            (p + "attn_rot_embd.weight", [spec.head_dim // 2, 1, 1, 1], 0),
            (p + "ffn_norm.weight", [e, 1, 1, 1], 0),
            (p + "ffn_gate_up.weight", [2 * spec.n_ffn, e, 1, 1], spec.weight_type),
            (p + "ffn_down.weight", [e, spec.n_ffn, 1, 1], spec.weight_type),
        ]
    return tensors


def _tensor_data_bytes(dims, type_id: int) -> Optional[int]:
    spec = registry.typeinfo(type_id)
    if spec is None:
        return None
    rows = 1
    for d in dims[1:]:
        rows *= d
    per_row = registry.row_bytes(type_id, dims[0])
    if per_row is None:
        return None
    return per_row * rows


def build_model(spec: Optional[ModelSpec] = None, *, seed: int = 0xC0FFEE) -> bytes:
    """Serialize a complete, internally-consisten small LLM in GGUF v3."""
    spec = spec if spec is not None else ModelSpec()
    arch = spec.arch

    # ---- KV section ----
    kvs = []
    kvs.append(enc_string_pair("general.architecture", arch))
    kvs.append(enc_scalar_pair("general.quantization_version", GI32, "<i", spec.quant_version))
    kvs.append(enc_scalar_pair("general.alignment", GU32, "<I", ALIGN))
    kvs.append(enc_string_pair("general.name", f"sentinel-synthetic-{arch}"))
    kvs.append(enc_string_pair("general.description", "synthetic fixture for gguf-sentinel tests"))
    kvs.append(enc_string_pair("tokenizer.ggml.model", "llama"))
    specials = ["<unk>", "<s>", "</s>", "<|start|>", "<|end|>"]
    tokens = specials + [f"tok{i}" for i in range(spec.n_vocab - len(specials))]
    kvs.append(enc_array_pair("tokenizer.ggml.tokens", GSTRING, None, tokens))
    token_types = [3] * len(tokens)
    kvs.append(enc_array_pair("tokenizer.ggml.token_type", GI32, "<i", token_types))
    scores = [((i * 2654435761) % 65536) - 32768 for i in range(len(tokens))]
    kvs.append(enc_array_pair("tokenizer.ggml.scores", GF32, "<f", [s / 32768.0 for s in scores]))
    kvs.append(enc_scalar_pair("tokenizer.ggml.bos_token_id", GI32, "<i", 1))
    kvs.append(enc_scalar_pair("tokenizer.ggml.eos_token_id", GI32, "<i", 2))
    kvs.append(enc_scalar_pair("tokenizer.ggml.eot_token_id", GI32, "<i", 3))
    kvs.append(enc_scalar_pair("tokenizer.ggml.unknown_token_id", GI32, "<i", 0))
    kvs.append(enc_scalar_pair("tokenizer.ggml.sep_token_id", GI32, "<i", 0))
    kvs.append(enc_scalar_pair("tokenizer.ggml.pad_token_id", GI32, "<i", 0))
    kvs.append(enc_bool_pair("tokenizer.ggml.add_bos_token", True))
    kvs.append(enc_bool_pair("tokenizer.ggml.add_eos_token", True))
    kvs.append(enc_bool_pair("tokenizer.ggml.add_eot_token", True))
    template = ("{% for message in messages %}<|start|>{{ message['role']}}\n"
                "{{ message['content'] }}<|end|>\n{% endfor %}<|start|>assistant\n")
    kvs.append(enc_string_pair("tokenizer.chat_template", template))
    arch_keys = [
        ("vocab_size", GI32, "<i", spec.n_vocab),
        ("context_length", GI32, "<i", spec.ctx),
        ("embedding_length", GI32, "<i", spec.n_embed),
        ("feed_forward_length", GI32, "<i", spec.n_ffn),
        ("block_count", GI32, "<i", spec.n_block),
        ("attention.head_count", GI32, "<i", spec.n_head),
        ("attention.head_count_kv", GI32, "<i", spec.n_kv_head),
        ("attention.key_length", GI32, "<i", spec.head_dim),
        ("attention.value_length", GI32, "<i", spec.head_dim),
    ]
    for suffix, gt, fmt, val in arch_keys:
        kvs.append(enc_scalar_pair(arch + "." + suffix, gt, fmt, val))
    kv_blob = b"".join(kvs)

    # ---- tensor table + data ----
    layout = _tensor_layout(spec)
    infos = []
    data_parts = []
    cursor = 0
    for idx, (name, dims, type_id) in enumerate(layout):
        size = _tensor_data_bytes(dims, type_id)
        if size is None:
            raise SystemExit(f"cannot size tensor {name}")
        cursor = align_up(cursor) if idx > 0 else 0
        infos.append(enc_tensor(name, dims, type_id, cursor))
        data_parts.append(_fill(seed + idx * 7919, size))
        cursor += size
    if data_parts:
        # pad the tail so data size is a multiple of the alignment
        pad = align_up(cursor) - cursor
        data_parts.append(b"\x00" * pad)
        cursor += pad

    n_kv = len(kvs)
    n_tensors = len(layout)
    head = MAGIC + struct.pack("<I", VERSION)
    head += struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv)
    head += struct.pack("<I", ALIGN)
    meta = head + kv_blob + b"".join(infos)
    meta_padded = meta + b"\x00" * (align_up(len(meta)) - len(meta))
    return meta_padded + b"".join(data_parts)


# ---------------------------------------------------------------- mutation
MUTATIONS = (
    "magic", "version", "count_tensors", "count_kv", "truncate", "truncated_header",
    "flip", "tensor_offset", "tensor_type", "tensor_dim", "alignment", "kv_type",
    "append", "zero",
)


def mutate(buf: bytes, kind: str, rng: random.Random) -> bytes:
    """Apply one named corruption; returns a new byte string (or the same one
    when the kind is inapplicable, so the driver can loop cheaply)."""
    body = bytearray(buf)
    if kind == "magic":
        i = rng.randrange(4)
        body[i] ^= rng.randbytes(1)[0] or 0x21
    elif kind == "version" and len(body) >= 8:
        body[4:8] = struct.pack("<I", rng.choice([0, 4, 99, 999999]))
    elif kind == "count_tensors" and len(body) >= 16:
        body[8:16] = struct.pack("<Q", rng.choice([2 ** 63, 2 ** 40, rng.randrange(4) + 1]))
    elif kind == "count_kv" and len(body) >= 24:
        body[16:24] = struct.pack("<Q", rng.choice([2 ** 63, 2 ** 40, rng.randrange(2) + 1]))
    elif kind == "alignment" and len(body) >= 28:
        body[24:28] = struct.pack("<I", rng.choice([0, 1, 3, 7, 48, 2 ** 20]))
    elif kind == "truncate":
        cut = rng.randrange(max(8, len(body) - 8), )
        body = body[:cut] if cut else body[:1]
    elif kind == "truncated_header":
        body = body[:rng.randrange(1, min(12, max(1, len(body))))]
    elif kind == "flip":
        i = rng.randrange(len(body))
        body[i] ^= rng.choice([1, 2, 0x40, 0x80, 0xFF])
    elif kind == "zero":
        i = rng.randrange(len(body))
        body[i] = 0
    elif kind == "append":
        body += rng.randbytes(rng.randrange(64) + 1)
    elif kind == "tensor_type" or kind == "tensor_offset" or kind == "tensor_dim" or kind == "kv_type":
        i = rng.randrange(max(1, len(body) - 8))
        if kind == "tensor_type":
            body[i:i + 4] = struct.pack("<I", rng.choice([99, 500, 2 ** 31, 0]))
        elif kind == "tensor_offset":
            body[i:i + 8] = struct.pack("<Q", rng.choice([2 ** 63, 2 ** 48, 1]))
        elif kind == "tensor_dim":
            body[i:i + 8] = struct.pack("<Q", rng.choice([0, 2 ** 40, 7]))
        else:
            body[i:i + 4] = struct.pack("<I", rng.choice([99, 13, 2 ** 31]))
    return bytes(body)
