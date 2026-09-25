"""Semantic audit rules over a ParsedModel.

The parser answers "is the container structurally sound?"; this module answers
"will llama.cpp actually load it -- and correctly?". That is where corrupt or
sloppily-converted files hide: a file can be 100% format-valid yet have
attention.head_count disagree with the qkv tensor's row count, which upstream
rejects at load with an opaque abort (llama_load: inconsistent tensor shapes)
-- or worse, loads and produces garbage logits.

Design rules for this file:
  * Each rule is a small pure function returning findings; a broken file
    triggers ALL applicable rules, never just the first.
  * Shape checks are convention-agnostic ("dumbbell" tests): a matrix must
    have one dim equal to its in-dim and the other to its out-dim, either
    transpose. GGUF has no transpose flag; llama.cpp reads ne[0] as rows but
    converters differ in the wild, and flagging a file's layout convention is
    not our job -- flagging a dim that matches NOTHING expected is.
  * Every finding cites expected and actual so triage is one glance, and a
    missing hyperparameter never guesses: the rule silently abstains (absence
    itself is covered by metadata rules), so we never fire false positives on
    arch families we don't know.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from . import registry
from .findings import Finding, make
from .keys import matches_arch_pattern, resolve, try_resolve
from .parser import ParsedModel, TensorInfo, tensor_bytes

# canonical suffixes (llama.cpp naming); checks fire only when present
EMB = "token_embd.weight"
OUT = "output.weight"
NORM = "token_embd_norm.weight"
ATTN_QKV = "attn_qkv.weight"
ATTN_OUT = "attn_out.weight"
FFN_GATE_UP = "ffn_gate_up.weight"
FFN_DOWN = "ffn_down.weight"


# ------------------------------------------------------------------ helpers
def _tensor_with_suffix(doc: ParsedModel, suffix: str) -> Optional[TensorInfo]:
    for t in doc.tensors:
        if t.name.endswith(suffix):
            return t
    return None


def _tensor_count_with_suffix(doc: ParsedModel, suffix: str) -> int:
    n = 0
    for t in doc.tensors:
        if t.name.endswith(suffix):
            n += 1
    return n


def _int_kv(doc: ParsedModel, arch: Optional[str], suffix: str):
    """Read an arch-scoped i32/u32 hyperparam; abstains (None) when absent or
    wrong-typed. Negative values abstain too: -1 means 'unset' upstream."""
    if arch is None:
        return None
    key = try_resolve(arch, suffix)
    if key is None:
        return None
    value, _type = doc.kv_typed(key)
    if value is None or not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 0:
        return None
    return value


def _str_kv(doc: ParsedModel, key: str):
    value, _type = doc.kv_typed(key)
    if isinstance(value, str) and value:
        return value
    return None


def _array_kv(doc: ParsedModel, key: str):
    value, _type = doc.kv_typed(key)
    if isinstance(value, tuple):
        return value
    return None


def _dumbbell(info: TensorInfo, in_dim: int, out_dim: int, *, allow_square: bool = False) -> bool:
    """True when dims[0:2] is exactly (in_dim, out_dim) in either orientation."""
    d0, d1 = info.declared_dims[0], info.declared_dims[1]
    if d0 == in_dim and d1 == out_dim:
        return True
    if d0 == out_dim and d1 == in_dim:
        return True
    if allow_square and in_dim == out_dim and d0 == d1 == in_dim:
        return True
    return False


def _other_dim(info: TensorInfo, anchor: int) -> Optional[int]:
    """Given one of the two leading dims equals `anchor`, return the other one."""
    d0, d1 = info.declared_dims[0], info.declared_dims[1]
    if d0 == anchor and d1 != anchor:
        return d1
    if d1 == anchor and d0 != anchor:
        return d0
    return None


# ------------------------------------------------------------------- rules
# Every rule: fn(doc, ctx) -> iterable of Finding. ctx holds parsed shortcuts.

def r_arch_present(doc: ParsedModel, ctx) -> list:
    out = []
    if ctx["arch"] is None:
        out.append(make("W_NO_ARCH",
                         "general.architecture is missing or empty: every arch-scoped "
                         "check, and llama.cpp's arch dispatch, depend on it"))
    return out


def r_arch_key_mismatch(doc: ParsedModel, ctx) -> list:
    """Keys scoped under a DIFFERENT known arch than the declared one --
    e.g. qwen2.* keys in a file that declares general.architecture=llama.
    Classic merge-the-wrong-config bug from custom conversion scripts."""
    if ctx["arch"] is None:
        return []
    from .keys import ARCH_NAMES
    out = []
    for entry in doc.kv:
        prefix = entry.key.split(".", 1)[0]
        if prefix in ARCH_NAMES and prefix != ctx["arch"]:
            out.append(make("E_ARCH_KEY_MISMATCH",
                             f"key {entry.key!r} is scoped under {prefix!r} but the file "
                             f"declares {ctx['arch']!r} -- the loader will never read it",
                             key=entry.key, offset=entry.entry_offset,
                             expected=ctx["arch"], actual=prefix))
    return out


def r_quant_version(doc: ParsedModel, ctx) -> list:
    out = []
    if not doc.tensors:
        return out
    value, type_id = doc.kv_typed("general.quantization_version")
    if value is None:
        out.append(make("W_NO_QUANT_VERSION",
                         "general.quantization_version missing: block layouts are "
                         "versioned upstream; loaders assume the newest",
                         key="general.quantization_version"))
        return out
    if isinstance(value, int) and not isinstance(value, bool):
        if value > registry.GGML_QNT_VERSION:
            out.append(make("E_QUANT_VERSION_UNSUPPORTED",
                             f"quantization_version {value} is newer than this tool's "
                             f"pinned gguf (supports <= {registry.GGML_QNT_VERSION})",
                             key="general.quantization_version",
                             expected=f"<= {registry.GGML_QNT_VERSION}", actual=value))
    return out


def r_head_geometry(doc: ParsedModel, ctx) -> list:
    """The money rule family. Only runs when the file declares BOTH heads and
    head_dim; abstains (no false alarms) on arch families that omit them."""
    out = []
    arch = ctx["arch"]
    heads = _int_kv(doc, arch, "attention.head_count")
    kv_heads = _int_kv(doc, arch, "attention.head_count_kv")
    key_len = _int_kv(doc, arch, "attention.key_length")
    val_len = _int_kv(doc, arch, "attention.value_length")
    hidden = _int_kv(doc, arch, "embedding_length")

    if key_len is not None and val_len is not None and key_len != val_len:
        out.append(make("E_KV_HEAD_MISMATCH",
                         "attention.key_length != attention.value_length: llama.cpp's "
                         "build path requires they match (d_k == d_v) for every supported arch",
                         key=resolve(arch, "attention.value_length"),
                         expected=key_len, actual=val_len))
    if key_len is not None and key_len % 2 == 1:
        rot = _int_kv(doc, arch, "attention.rot_ndims")
        if rot is None or rot != 0:
            out.append(make("W_HEAD_DIM_ODD",
                             f"attention.key_length {key_len} is odd: rope tables pair "
                             f"dims two-by-two and most kernels want an even head dim",
                             key=resolve(arch, "attention.key_length"), actual=key_len))
    if hidden is not None and heads:
        if hidden % heads != 0:
            out.append(make("W_INCONSISTENT_HYPERPARAM",
                             f"{hidden} % {heads} != 0: with no explicit key_length the "
                             f"loader derives head_dim = hidden/heads and needs it integral",
                             key=resolve(arch, "attention.head_count"),
                             expected="divisible", actual=f"{hidden} % {heads} = {hidden % heads}"))
        elif key_len is not None and heads * key_len != hidden:
            out.append(make("E_HEAD_DIM_MISMATCH",
                             "heads * key_length must equal embedding_length when GQA "
                             "collapses onto the full-width qkv projection",
                             key=resolve(arch, "attention.key_length"),
                             expected=hidden, actual=heads * key_len))
        elif key_len is not None and heads * key_len == hidden:
            ctx["hidden_derived"] = hidden
    if kv_heads is not None and heads is not None and kv_heads > 0 and heads % kv_heads != 0:
        out.append(make("W_INCONSISTENT_HYPERPARAM",
                         f"heads {heads} is not a multiple of kv heads {kv_heads}: grouped-query "
                         f"attention needs heads % kv_heads == 0 so groups divide evenly",
                         key=resolve(arch, "attention.head_count"),
                         expected=f"multiple of {kv_heads}", actual=heads))
    value_len = _int_kv(doc, arch, "attention.value_length")
    if value_len is not None and value_len % 2 == 1:
        out.append(make("W_KV_HEAD_DIM_ODD",
                         f"attention.value_length {value_len} is odd: rope tables slice the "
                         f"value head two-by-two and most kernels want an even kv head width",
                         key=resolve(arch, "attention.value_length"), actual=value_len))
    return out


def r_q4k_even_hidden(doc: ParsedModel, ctx) -> list:
    """Upstream build_arch rejects q4_K with odd hidden_size unless heads are 0.
    Faithful port of the exact guard."""
    if ctx["arch"] is None or not doc.tensors:
        return []
    hidden = ctx.get("hidden")
    heads = _int_kv(doc, ctx["arch"], "attention.head_count")
    kvh = _int_kv(doc, ctx["arch"], "attention.head_count_kv")
    if hidden is None or hidden % 2 == 0:
        return []
    has_q4k = False
    for t in doc.tensors:
        spec = registry.typeinfo(t.type_id)
        if spec is not None and spec.name.lower() == "q4_k":
            has_q4k = True
            break
    if has_q4k and heads != 0 and kvh != 0:
        return [make("E_FFN_DIM_MISMATCH",
                     f"hidden {hidden} is odd with q4_K tensors: llama.cpp aborts "
                     f"\"all models but gemma require even hidden_size\" (heads==0/kvh==0 "
                     f"is the only exemption)",
                     expected="even hidden_size", actual=hidden)]
    return []


def r_block_count(doc: ParsedModel, ctx) -> list:
    out = []
    if ctx["arch"] is None:
        return out
    declared = _int_kv(doc, ctx["arch"], "block_count")
    seen = set()
    import re as _re
    pat = _re.compile(r"^blk\.(\d+)\.")
    for t in doc.tensors:
        m = pat.match(t.name)
        if m:
            seen.add(int(m.group(1)))
    if declared is not None and seen:
        lo, hi = min(seen), max(seen)
        if hi + 1 != declared or lo != 0:
            out.append(make("E_BLOCK_COUNT_MISMATCH",
                             f"declared block_count {declared} but tensors span "
                             f"blk.{lo}..blk.{hi} (contiguous from 0 expected)",
                             key=resolve(ctx["arch"], "block_count"),
                             expected=declared, actual=hi + 1))
        missing = set(range(declared)) - seen
        if 0 < declared <= 4096 and missing and hi + 1 == declared:
            out.append(make("E_BLOCK_COUNT_MISMATCH",
                             f"blocks {sorted(missing)[:6]} have no tensors at all",
                             key=resolve(ctx["arch"], "block_count"),
                             expected=f"{declared} full blocks", actual=f"{len(seen)} present"))
    ctx["block_seen"] = seen
    return out


def r_tensor_shapes(doc: ParsedModel, ctx) -> list:
    """The dumbbell suite: every canonical matrix's leading dims must match
    (in_dim, out_dim) in either orientation. Also derives the qkv split."""
    out = []
    arch, hidden, vocab = ctx["arch"], ctx.get("hidden"), ctx.get("vocab")
    heads, kvh, key_len = ctx.get("heads"), ctx.get("kv_heads"), ctx.get("key_len")

    def need(suffix):
        return _tensor_with_suffix(doc, suffix)

    if vocab is not None and hidden is not None:
        emb = need(EMB)
        if emb is not None and not _dumbbell(emb, hidden, vocab):
            out.append(make("E_EMBED_ROWS_MISMATCH",
                             f"token_embd.weight dims {emb.declared_dims[0:2]} are neither "
                             f"(embedding_length {hidden}, vocab {vocab}) nor transposed",
                             tensor=emb.name, offset=emb.dims_offset,
                             expected=(hidden, vocab), actual=emb.declared_dims[0:2]))
        outw = need(OUT)
        if outw is not None and not _dumbbell(outw, hidden, vocab):
            out.append(make("E_EMBED_ROWS_MISMATCH",
                             f"output.weight dims {outw.declared_dims[0:2]} match neither "
                             f"(embedding_length {hidden}, vocab {vocab}) nor transposed",
                             tensor=outw.name, offset=outw.dims_offset,
                             expected=(hidden, vocab), actual=outw.declared_dims[0:2]))

    if heads and kvh and key_len and hidden is not None:
        qkv = need(ATTN_QKV)
        expected_rows = (heads + 2 * kvh) * key_len
        if qkv is not None:
            if not _dumbbell(qkv, hidden, expected_rows, allow_square=True):
                out.append(make("E_HEAD_DIM_MISMATCH",
                                 f"attn_qkv.weight dims {qkv.declared_dims[0:2]} disagree with "
                                 f"(hidden {hidden}, (heads+2*kvh)*key_length={expected_rows}): "
                                 f"llama.cpp will abort at build_mmap or produce garbage",
                                 tensor=qkv.name, offset=qkv.dims_offset,
                                 expected=(hidden, expected_rows), actual=qkv.declared_dims[0:2]))
            else:
                other = _other_dim(qkv, hidden)
                if other is not None and other != expected_rows and \
                        qkv.declared_dims[0] == qkv.declared_dims[1]:
                    out.append(make("E_HEAD_DIM_MISMATCH",
                                     f"square attn_qkv but hidden*2 == {(heads + 2 * kvh) * key_len}"
                                     f" coincidence check failed",
                                     tensor=qkv.name, offset=qkv.dims_offset,
                                     expected=expected_rows, actual=other))
        aout = need(ATTN_OUT)
        if aout is not None and not _dumbbell(aout, heads * key_len, hidden, allow_square=True):
            out.append(make("E_HEAD_DIM_MISMATCH",
                             f"attn_out.weight dims {aout.declared_dims[0:2]} agree with neither "
                             f"(heads*key_length {heads * key_len}, hidden {hidden})",
                             tensor=aout.name, offset=aout.dims_offset,
                             expected=(heads * key_len, hidden), actual=aout.declared_dims[0:2]))

    ffn = ctx.get("ffn")
    if ffn and hidden is not None:
        gate = need(FFN_GATE_UP)
        if gate is not None and not _dumbbell(gate, hidden, 2 * ffn):
            out.append(make("E_FFN_DIM_MISMATCH",
                             f"ffn_gate_up.weight dims {gate.declared_dims[0:2]} are neither "
                             f"(hidden {hidden}, 2*feed_forward_length {2 * ffn}) nor transposed",
                             tensor=gate.name, offset=gate.dims_offset,
                             expected=(hidden, 2 * ffn), actual=gate.declared_dims[0:2]))
        down = need(FFN_DOWN)
        if down is not None and not _dumbbell(down, ffn, hidden):
            out.append(make("E_FFN_DIM_MISMATCH",
                             f"ffn_down.weight dims {down.declared_dims[0:2]} are neither "
                             f"(feed_forward_length {ffn}, hidden {hidden}) nor transposed",
                             tensor=down.name, offset=down.dims_offset,
                             expected=(ffn, hidden), actual=down.declared_dims[0:2]))
    return out


def r_norm_dims(doc: ParsedModel, ctx) -> list:
    out = []
    hidden = ctx.get("hidden")
    if hidden is None:
        return out
    for t in doc.tensors:
        if t.name.endswith("_norm.weight"):
            if t.declared_dims[0] != hidden:
                out.append(make("W_INCONSISTENT_HYPERPARAM",
                                 f"{t.name} has rows {t.declared_dims[0]}, but norm vectors are "
                                 f"length embedding_length ({hidden})",
                                 tensor=t.name, offset=t.dims_offset,
                                 expected=hidden, actual=t.declared_dims[0]))
    return out


def r_unknown_types(doc: ParsedModel, ctx) -> list:
    out = []
    for t in doc.tensors:
        if registry.typeinfo(t.type_id) is None:
            out.append(make("E_BAD_TENSOR_TYPE",
                             f"tensor {t.name!r} has type id {t.type_id}, which is not in the "
                             f"pinned ggml_type enum (removed id tba or corruption)",
                             tensor=t.name, offset=t.type_offset, actual=t.type_id))
        elif tensor_bytes(t) is None:
            spec = registry.typeinfo(t.type_id)
            if spec.blck > 1 and t.declared_dims[0] % spec.blck != 0:
                out.append(make("E_QUANT_ROW_NOT_BLOCK_ALIGNED",
                                 f"tensor {t.name!r} rows {t.declared_dims[0]} not a multiple of "
                                 f"block size {spec.blck} for {spec.name}: cannot be quantized",
                                 tensor=t.name, offset=t.dims_offset,
                                 expected=f"multiple of {spec.blck}", actual=t.declared_dims[0]))
    return out


def r_offsets_and_overlaps(doc: ParsedModel, ctx) -> list:
    out = []
    sized = []
    for t in doc.tensors:
        nbytes = tensor_bytes(t)
        if nbytes is None:
            continue
        sized.append((t.offset, nbytes, t))
        if doc.alignment > 0 and t.offset % doc.alignment != 0:
            out.append(make("E_DATA_UNALIGNED",
                             f"tensor {t.name!r} data offset {t.offset} not aligned to "
                             f"{doc.alignment} (llama.cpp uses mmap at alignment boundaries)",
                             tensor=t.name, offset=t.offset_offset,
                             expected=f"multiple of {doc.alignment}", actual=t.offset))
    sized.sort(key=lambda row: row[0])
    for i in range(1, len(sized)):
        prev_off, prev_len, prev_t = sized[i - 1]
        cur_off, _cur_len, cur_t = sized[i]
        if cur_off < prev_off + prev_len:
            out.append(make("E_TENSOR_OVERLAP",
                             f"{cur_t.name!r} at {cur_off} starts inside {prev_t.name!r} "
                             f"(ends {prev_off + prev_len}): one of them reads garbage weights",
                             tensor=cur_t.name, offset=cur_t.offset_offset,
                             expected=f">= {prev_off + prev_len}", actual=cur_off))
    return out


def r_past_eof(doc: ParsedModel, ctx) -> list:
    out = []
    file_avail = doc.size - doc.data_start
    for t in doc.tensors:
        nbytes = tensor_bytes(t)
        if nbytes is None:
            continue
        if t.offset + nbytes > file_avail:
            out.append(make("E_TENSOR_PAST_EOF",
                             f"tensor {t.name!r} claims bytes {t.offset}..{t.offset + nbytes} "
                             f"but the data section only has {file_avail} bytes: file is truncated "
                             f"or the tensor was resized after quantization",
                             tensor=t.name, offset=t.offset_offset,
                             expected=f"<= {file_avail}", actual=t.offset + nbytes))
    return out


def r_padding_zeroed(doc: ParsedModel, ctx) -> list:
    """gguf.h: the bytes between the end of metadata and the alignment
    boundary are zero padding. Non-zero bytes there mean the declared counts
    stopped short and real records sit where zeros belong -- the classic
    signature of a shrunk n_tensors/n_kv that every reader trusts blindly."""
    if doc.data_start <= doc.meta_end:
        return []
    buf = ctx.get("buf")
    if buf is None or doc.data_start > len(buf):
        return []
    gap = buf[doc.meta_end:doc.data_start]
    if any(gap):
        return [make("E_PADDING_NOT_ZERO",
                     f"{sum(1 for b in gap if b)} of {len(gap)} padding bytes between "
                     f"metadata end {doc.meta_end} and data start {doc.data_start} are non-zero: "
                     f"the declared counts under-report what is really in the file",
                     offset=doc.meta_end, expected="all zero",
                     actual="non-zero bytes present")]
    return []


def r_trailing_junk(doc: ParsedModel, ctx) -> list:
    if not doc.tensors or doc.data_start >= doc.size:
        return []
    end = 0
    ok = True
    for t in doc.tensors:
        nbytes = tensor_bytes(t)
        if nbytes is None:
            ok = False
            break
        end = max(end, t.offset + nbytes)
    if not ok:
        return []
    junk = (doc.size - doc.data_start) - end
    if junk > max(1024 * 1024, 4 * doc.alignment):
        return [make("W_TRAILING_JUNK",
                     f"{junk} bytes hang past the last tensor: trailing garbage after the "
                     f"data section (harmless to llama.cpp, common after sloppy re-quantization)",
                     expected="<= alignment pad", actual=junk)]
    return []


def r_degenerate_data(doc: ParsedModel, ctx) -> list:
    """All-identical weight regions: a conversion that died mid-write leaves a
    perfectly valid container full of zeros. Detects it before 'inference'."""
    if doc.data_start >= doc.size or not doc.tensors:
        return []
    sample = doc.size - doc.data_start
    if sample <= 64:
        return []
    from .reader import Reader
    rd = Reader(ctx["buf"])
    try:
        rd.seek(doc.data_start)
        probe = rd.read_exact(min(sample, 65536, rd.remaining()))
    except Exception:
        return []
    if len(probe) < 64:
        return []
    from collections import Counter
    first, same = Counter(probe).most_common(1)[0]
    if same >= len(probe) * 0.995:
        return [make("W_SUSPICIOUS_DENSITY",
                     f">99.5% of the sampled data section is byte 0x{first:02x}: weights "
                     f"are (near-)constant, this model cannot have been trained/converted",
                     expected="varied bytes", actual=f"0x{first:02x} x {same}/{len(probe)}")]
    return []


def r_tokenizer_counts(doc: ParsedModel, ctx) -> list:
    out = []
    tokens = _array_kv(doc, "tokenizer.ggml.tokens")
    scores = _array_kv(doc, "tokenizer.ggml.scores")
    types = _array_kv(doc, "tokenizer.ggml.token_type")
    if tokens is None:
        return out
    n = len(tokens)
    if scores is not None and len(scores) != n:
        out.append(make("E_TOKENIZER_COUNT_MISMATCH",
                         f"tokenizer.ggml.scores has {len(scores)} entries, tokens has {n}: "
                         f"the piece loader indexes scores parallel to tokens",
                         key="tokenizer.ggml.scores", expected=n, actual=len(scores)))
    if types is not None and len(types) != n:
        out.append(make("E_TOKENIZER_COUNT_MISMATCH",
                         f"tokenizer.ggml.token_type has {len(types)} entries, tokens has {n}",
                         key="tokenizer.ggml.token_type", expected=n, actual=len(types)))
    if ctx.get("vocab") is not None and ctx["vocab"] != n:
        out.append(make("W_VOCAB_MISMATCH",
                         f"declared vocab_size {ctx['vocab']} but the token array holds {n} "
                         f"(llama.cpp resizes to the array, but conversion metadata disagrees)",
                         key=resolve(ctx["arch"], "vocab_size") if ctx["arch"] else "vocab_size",
                         expected=ctx["vocab"], actual=n))
    if n == 0:
        out.append(make("W_EMBED_TOKENS_ZERO_ROWS",
                         "tokenizer.ggml.tokens is empty: the model has a vocab of zero",
                         key="tokenizer.ggml.tokens", expected="> 0", actual=0))
    return out


def r_special_ids(doc: ParsedModel, ctx) -> list:
    out = []
    tokens = _array_kv(doc, "tokenizer.ggml.tokens")
    n = len(tokens) if tokens is not None else None
    for suffix in ("bos_token_id", "eos_token_id", "eot_token_id", "unk_token_id",
                   "sep_token_id", "pad_token_id"):
        value, _type = doc.kv_typed("tokenizer.ggml." + suffix)
        if value is None or not isinstance(value, int) or isinstance(value, bool):
            continue
        if value >= 0 and n is not None and value >= n:
            out.append(make("W_SPECIAL_TOKEN_IDS_OUT_OF_RANGE",
                             f"tokenizer.ggml.{suffix} = {value} but there are only {n} tokens: "
                             f"template rendering will crash or silently pick nothing",
                             key="tokenizer.ggml." + suffix, expected=f"< {n}", actual=value))
    for add_key in ("add_bos_token", "add_eos_token"):
        add_val, _t = doc.kv_typed("tokenizer.ggml." + add_key)
        if add_val:
            id_val, _t2 = doc.kv_typed("tokenizer.ggml." + add_key.replace("add_", "").replace("_token", "_token_id"))
            if id_val is None or id_val == 0:
                out.append(make("W_BOS_EOS_MISSING",
                                 f"{add_key} is true but {id_val!r}: the wrapper token to add is unset",
                                 key="tokenizer.ggml." + add_key))
    return out


_TEMPLATE_TOKEN = re.compile(r"<\|[a-z0-9A-Z_\-\.]+\|>")


def r_chat_template(doc: ParsedModel, ctx) -> list:
    out = []
    template = _str_kv(doc, "tokenizer.chat_template")
    if template is None:
        out.append(make("W_NO_CHAT_TEMPLATE",
                         "no tokenizer.chat_template: chat models without one fall back to a "
                         "raw prompt and answer with plain text",
                         key="tokenizer.chat_template"))
        return out
    specials = _array_kv(doc, "tokenizer.ggml.added_tokens") or ()
    tokens = _array_kv(doc, "tokenizer.ggml.tokens") or ()
    if "<|" not in template:
        out.append(make("W_TEMPLATE_NO_SPECIAL_TOKENS",
                         "chat_template contains no <|...|> markers: roles are not delimited, "
                         "the model can leak user/assistant boundaries",
                         key="tokenizer.chat_template"))
        return out
    known = set(specials) | {t for t in tokens if isinstance(t, str) and t.startswith("<|")}
    for tok in _TEMPLATE_TOKEN.findall(template):
        if tok not in known:
            out.append(make("E_TEMPLATE_SPECIAL_TOKENS_UNREGISTERED",
                             f"chat_template uses {tok!r} but it is not registered in "
                             f"tokenizer.ggml.added_tokens or tokens: the tokenizer will "
                             f"split it into pieces and chat breaks",
                             key="tokenizer.chat_template", expected="registered special", actual=tok))
    return out


RULES = (
    ("arch_present", r_arch_present),
    ("arch_key_mismatch", r_arch_key_mismatch),
    ("quant_version", r_quant_version),
    ("head_geometry", r_head_geometry),
    ("q4k_even_hidden", r_q4k_even_hidden),
    ("block_count", r_block_count),
    ("tensor_shapes", r_tensor_shapes),
    ("norm_dims", r_norm_dims),
    ("unknown_types", r_unknown_types),
    ("offsets_and_overlaps", r_offsets_and_overlaps),
    ("past_eof", r_past_eof),
    ("padding_zeroed", r_padding_zeroed),
    ("trailing_junk", r_trailing_junk),
    ("degenerate_data", r_degenerate_data),
    ("tokenizer_counts", r_tokenizer_counts),
    ("special_ids", r_special_ids),
    ("chat_template", r_chat_template),
)


def analyze(doc: ParsedModel, buf: bytes) -> tuple:
    """Run every rule over a parsed model. Returns a deduped, severity-ordered
    tuple of Findings (parser findings + rule findings)."""
    arch = _str_kv(doc, "general.architecture")
    ctx = {
        "arch": arch,
        "buf": buf,
        "hidden": _int_kv(doc, arch, "embedding_length") if arch else None,
        "vocab": _int_kv(doc, arch, "vocab_size") if arch else None,
        "ffn": _int_kv(doc, arch, "feed_forward_length") if arch else None,
        "heads": _int_kv(doc, arch, "attention.head_count") if arch else None,
        "kv_heads": _int_kv(doc, arch, "attention.head_count_kv") if arch else None,
        "key_len": _int_kv(doc, arch, "attention.key_length") if arch else None,
    }
    findings = list(doc.findings)
    for _name, rule in RULES:
        findings.extend(rule(doc, ctx))
    seen = set()
    deduped = []
    for f in findings:
        k = f.key_id()
        if k not in seen:
            seen.add(k)
            deduped.append(f)
    from .findings import SEVERITY_RANK
    deduped.sort(key=lambda f: (-SEVERITY_RANK[f.severity], f.code, f.offset or 0))
    return tuple(deduped)
