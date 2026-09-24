"""The ``sentinel`` command line: scan / explain / generate / fuzz / types.

scan    audit one or more .gguf files; exit 1 on errors under --fail-on error
        (the CI gate). --format text|json|github chooses the renderer.
explain print what a finding code means and where it comes from upstream.
generate write a synthetic-but-format-accurate GGUF (demo files, fixtures).
fuzz    run the mutation harness over the synthetic bases.
types   dump the pinned quantization block table (the upstream-derived one).
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .findings import CODE_SEVERITIES, Finding, SentinelError, make
from .parser import ParsedModel, parse_gguf
from .report import (EXIT_FATAL, EXIT_OK, exit_code_for, render_github,
                     render_json, render_text)
from .rules import analyze

MAX_FILE = 256 * 1024 * 1024  # refuse paths larger than this up front


def scan_bytes(buf: bytes, *, filename: Optional[str] = None) -> tuple:
    """Parse + analyze in-memory bytes. Raises SentinelError on unrecoverable
    damage -- callers (fuzz, cli) translate that into a coded fatal finding.
    Returns (ParsedModel or None, tuple of findings)."""
    doc = parse_gguf(buf, filename=filename)
    findings = analyze(doc, buf)
    return doc, findings


def scan_file(path: str) -> tuple:
    """File wrapper around scan_bytes with up-front size screening."""
    import os
    if not os.path.isfile(path):
        raise SentinelError("E_NOT_A_FILE", f"no such file: {path}", offset=None)
    size = os.path.getsize(path)
    if size > MAX_FILE:
        raise SentinelError("E_OVERSIZED",
                             f"{path} is {size} bytes (> {MAX_FILE}): refusing to mmap "
                             f"a path this large; pass a shrunken sample", offset=None)
    with open(path, "rb") as fh:
        buf = fh.read()
    return scan_bytes(buf, filename=path)


# One-sentence definitions so `explain` teaches without the web. The upstream
# citation tells triagers where the check physically lives.
CODE_DOCS = {
    "E_MAGIC": ("header magic is not 'GGUF'", "gguf.c: gguf_open, first four bytes"),
    "E_VERSION": ("unsupported GGUF version field", "gguf.h GGUF_VERSION_MAX"),
    "E_TRUNCATED_HEADER": ("file ends inside the 28-byte header", "structural"),
    "E_TRUNCATED_KV": ("file ends inside the KV section", "structural"),
    "E_TRUNCATED_TENSOR_INFO": ("file ends inside the tensor-info table", "structural"),
    "E_TRUNCATED_KV_VALUE": ("a KV array claims more bytes than exist", "structural"),
    "E_BAD_KV_TYPE": ("a KV pair uses an unknown gguf_type discriminant", "gguf.h enum gguf_type"),
    "E_BAD_STRING": ("a length-prefixed string is not valid utf8", "gguf.c gguf_get_string"),
    "E_BAD_TENSOR_TYPE": ("tensor type id outside the pinned ggml_type enum", "ggml.h enum ggml_type"),
    "E_HUGE_ALLOC": ("a count/length field claims more bytes than the file can hold", "defensive bound"),
    "E_NEGATIVE_DIM": ("tensor dim has the int64 sign bit set", "defensive bound"),
    "E_TOO_MANY_DIMS": ("n_dims outside 1..4", "ggml.h GGML_MAX_DIMS"),
    "E_NEGATIVE_COUNT": ("count/offset field has the int64 sign bit set", "defensive bound"),
    "E_DATA_UNALIGNED": ("tensor data offset not a multiple of general.alignment", "llama_load: mmap requires alignment"),
    "E_TENSOR_PAST_EOF": ("tensor data extends past the end of the file", "conversion truncation"),
    "E_TENSOR_OVERLAP": ("two tensors occupy the same bytes", "quantization tool bug"),
    "W_TENSOR_GAP": ("large gap between consecutive tensors", "wasted space only"),
    "W_DATA_OFFSET_MISMATCH": ("offset field disagrees with sequential layout", "informational"),
    "E_ZERO_FILE": ("empty file", "structural"),
    "E_OVERSIZED": ("file larger than the safety ceiling", "defensive bound"),
    "W_NO_ARCH": ("general.architecture missing", "llama.cpp dispatches on it"),
    "W_NO_QUANT_VERSION": ("general.quantization_version missing", "ggml-common.h"),
    "W_NO_ALIGNMENT": ("alignment missing or not a power of two", "gguf.h GGUF_DEFAULT_ALIGNMENT=32"),
    "E_ARCH_KEY_MISMATCH": ("keys scoped under a foreign arch prefix", "llama_loader_load: prefixes select the arch"),
    "W_INCONSISTENT_HYPERPARAM": ("hyperparams contradict each other", "llama.cpp build_arch checks"),
    "W_EMBED_TOKENS_ZERO_ROWS": ("token array is empty", "tokenizer sanity"),
    "E_HEAD_DIM_MISMATCH": ("qkv/attn_out shapes disagree with heads*key_length", "llama_build_kv / build_mlp"),
    "E_KV_HEAD_MISMATCH": ("key_length != value_length", "llama.cpp requires d_k == d_v"),
    "E_FFN_DIM_MISMATCH": ("ffn gate/down shapes disagree with feed_forward_length", "build_ffn"),
    "E_EMBED_ROWS_MISMATCH": ("token_embd/output shapes disagree with hidden/vocab", "build_rope / build_norm"),
    "E_BLOCK_COUNT_MISMATCH": ("tensor blk.N span contradicts block_count", "tensor existence loops"),
    "W_HEAD_DIM_ODD": ("odd key_length breaks rope pairing", "rotate_half in kernels"),
    "W_KV_HEAD_DIM_ODD": ("odd value_length breaks rope kv slicing", "rotate_half"),
    "E_QUANT_ROW_NOT_BLOCK_ALIGNED": ("rows not a multiple of the quant block size", "ggml_quantize_tensor preconditions"),
    "E_QUANT_SIZE_NOT_BLOCK_MULTIPLE": ("quantized size is not block-multiple", "ggml.c type_traits"),
    "E_QUANT_VERSION_UNSUPPORTED": ("quantization_version newer than pinned spec", "ggml-common.h GGML_QNT_VERSION"),
    "W_VOCAB_MISMATCH": ("vocab_size metadata != token array length", "llama_load resizes to array"),
    "W_SPECIAL_TOKEN_IDS_OUT_OF_RANGE": ("bos/eos id >= token count", "tokenizer rendering"),
    "W_NO_CHAT_TEMPLATE": ("no chat_template", "chat UX quality"),
    "W_BOS_EOS_MISSING": ("add_bos/eos true but id unset", "tokenizer rendering"),
    "E_TOKENIZER_COUNT_MISMATCH": ("scores/token_type length != tokens length", "llama_load asserts parallel arrays"),
    "W_TEMPLATE_NO_SPECIAL_TOKENS": ("template without <|...|> markers", "chat correctness"),
    "E_TEMPLATE_SPECIAL_TOKENS_UNREGISTERED": ("template uses unregistered specials", "vocab cannot match them"),
    "W_TRAILING_JUNK": ("bytes hang past the last tensor", "informational"),
    "W_SUSPICIOUS_DENSITY": ("weights nearly all one byte value", "failed conversion smell"),
    "E_UNHANDLED": ("unexpected internal error (please file it!)", "this tool itself"),
    "E_NOT_A_FILE": ("path is not a file", "cli layer"),
}


def explain(code: str) -> str:
    row = CODE_DOCS.get(code)
    sev = CODE_SEVERITIES.get(code, "?")
    if row is None:
        return f"{code}: no doc entry (severity {sev})"
    what, where = row
    return f"{code} [{sev}] -- {what}\n  upstream origin: {where}"


def _cmd_scan(args) -> int:
    total_exit = EXIT_OK
    emitted_any = False
    for path in args.paths:
        try:
            doc, findings = scan_file(path)
        except SentinelError as exc:
            f = make(exc.code, exc.message, tensor=exc.tensor, offset=exc.offset)
            doc, findings = None, (f,)
            code = EXIT_FATAL if args.fatal_fatal else exit_code_for((f,), fail_on=args.fail_on)
            total_exit = max(total_exit, code)
            if args.format == "json":
                print(render_json(None, findings))
            elif args.format == "github":
                print(render_github(_stub(path), findings))
            else:
                print(f"{path}\n  [FATAL] {f.code}: {f.message}")
            emitted_any = True
            continue
        if args.format == "json":
            print(render_json(doc, findings))
        elif args.format == "github":
            print(render_github(doc, findings))
        else:
            print(render_text(doc, findings))
        total_exit = max(total_exit, exit_code_for(findings, fail_on=args.fail_on))
        emitted_any = True
    if not emitted_any:
        print("nothing to scan", file=sys.stderr)
        return EXIT_FATAL
    return total_exit


def _stub(path: str) -> ParsedModel:
    return ParsedModel(filename=path, size=0, version=0, n_tensors_declared=0,
                        n_kv_declared=0, alignment=0, meta_end=0, data_start=0,
                        kv=(), kv_map={}, tensors=(), findings=())


def _cmd_explain(args) -> int:
    codes = args.codes if args.codes else tuple(CODE_SEVERITIES)
    for c in codes:
        print(explain(c))
    return EXIT_OK


def _cmd_generate(args) -> int:
    from .generate import ModelSpec, build_model
    spec = ModelSpec(arch=args.arch, n_block=args.blocks, n_head=args.heads,
                     n_kv_head=args.kv_heads, n_embed=args.embed, n_ffn=args.ffn,
                     n_vocab=args.vocab, head_dim=args.head_dim,
                     weight_type=args.weight_type, embed_type=args.embed_type)
    blob = build_model(spec, seed=args.seed)
    with open(args.out, "wb") as fh:
        fh.write(blob)
    print(f"wrote {args.out} ({len(blob):,} bytes, {args.arch}, {args.blocks} blocks)")
    return EXIT_OK


def _cmd_fuzz(args) -> int:
    from .fuzz import main as fuzz_main
    return fuzz_main(["--iterations", str(args.iterations), "--seed", str(args.seed),
                      "--save-dir", args.save_dir or ""])


def _cmd_types(args) -> int:
    from . import registry
    from .registry import vendored_digest
    print(f"vendored upstream digest: {vendored_digest()[:16]}")
    print(f"{'id':>3} {'name':<12} {'blck':>5} {'bytes':>6}  bits/w  quant")
    for spec in registry.all_types():
        print(f"{spec.id:>3} {spec.name:<12} {spec.blck:>5} {spec.size:>6}  "
              f"{spec.bits_per_weight:5.2f}  {'y' if spec.quant else 'n'}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sentinel", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("scan", help="audit .gguf files")
    sp.add_argument("paths", nargs="+")
    sp.add_argument("--format", choices=("text", "json", "github"), default="text")
    sp.add_argument("--fail-on", choices=("error", "warn", "info"), default="error",
                     help="severity that maps to exit code 1 (CI gate)")
    sp.add_argument("--fatal-fatal", action="store_true",
                     help="unparseable files exit 3 instead of mapping through --fail-on")

    xp = sub.add_parser("explain", help="what does a finding code mean")
    xp.add_argument("codes", nargs="*")

    gp = sub.add_parser("generate", help="write a synthetic GGUF file")
    gp.add_argument("--out", required=True)
    gp.add_argument("--arch", default="llama")
    gp.add_argument("--blocks", type=int, default=4)
    gp.add_argument("--heads", type=int, default=8)
    gp.add_argument("--kv-heads", type=int, default=4)
    gp.add_argument("--embed", type=int, default=256)
    gp.add_argument("--ffn", type=int, default=512)
    gp.add_argument("--vocab", type=int, default=256)
    gp.add_argument("--head-dim", type=int, default=32)
    gp.add_argument("--weight-type", type=int, default=2)
    gp.add_argument("--embed-type", type=int, default=1)
    gp.add_argument("--seed", type=int, default=0xC0FFEE)

    fp = sub.add_parser("fuzz", help="mutation fuzz the parser")
    fp.add_argument("--iterations", type=int, default=200)
    fp.add_argument("--seed", type=int, default=0)
    fp.add_argument("--save-dir", default="crashers")

    sub.add_parser("types", help="dump the pinned quantization block table")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        handler = {"scan": _cmd_scan, "explain": _cmd_explain, "generate": _cmd_generate,
                   "fuzz": _cmd_fuzz, "types": _cmd_types}[args.command]
    except LookupError:
        print(f"unknown command {args.command!r}", file=sys.stderr)
        return EXIT_FATAL
    try:
        return handler(args)
    except SentinelError as exc:  # anything escaping the handler still exits coded
        print(f"[FATAL] {exc.code}: {exc.message}", file=sys.stderr)
        return EXIT_FATAL
    except Exception as exc:
        print(f"[E_UNHANDLED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
