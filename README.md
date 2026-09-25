# gguf-sentinel

A structural + semantic validator for **GGUF** model files (the `llama.cpp` /
`gguf.cpp` format used by llama.cpp, Ollama, LM-Studio, Kobold and friends).

A GGUF file that *loads* can still be quietly garbage. `gguf.cpp` validates the
magic number and the declared counts — that's it. The tensor table, the
hyperparam metadata and the weight geometry are trusted blindly, so a botched
conversion ships as a model that produces fluent nonsense instead of crashing.
gguf-sentinel runs **17 semantic rules** the loader never runs and tells you
which tensor, which byte offset, and which upstream check it corresponds to.

```
$ sentinel scan mistral-7b-q4_K-sloppy.gguf
  gguf v3  |  291 tensors (146 quantized)  |  44 kv  |  arch=mistral  |  4,367,011,264 bytes
  E_HEAD_DIM_MISMATCH [error] -- blk.11.attn_qkv.weight rows 5632 != (heads + 2*kv_heads) * key_length
    upstream origin: gguf.cpp: gguf_loader_geometry, per-layer qkv allocation
  E_TENSOR_PAST_EOF [error] -- ffn_down.weight claims bytes 41e9..42e0 past the data section
    upstream origin: gguf.c: tensor mmap end check
  2 error / 0 warn / 0 info
$ echo $?
1
```

## What it catches (a few of the real classes)

| Finding | Why the loader doesn't | Why you care |
|---|---|---|
| `E_HEAD_DIM_MISMATCH` | shape fields trusted | attention silently computes wrong rows |
| `E_KV_HEAD_MISMATCH` | ditto | wrong KV head count ⇒ garbage after token 1 |
| `E_FFN_DIM_MISMATCH` | ditto | every MLP layer reads shifted weights |
| `E_EMBED_ROWS_MISMATCH` | vocab vs rows unchecked | output head hallucinates tokens |
| `E_TENSOR_OVERLAP` | offsets unchecked | two tensors share bytes: one is garbage |
| `E_TENSOR_PAST_EOF` | truncated download | half the file is missing |
| `E_PADDING_NOT_ZERO` | declared count under-reports | hidden tensor records in "padding" |
| `E_BAD_TENSOR_TYPE` | type id unchecked | whole tensor decoded with wrong quant |
| `E_DATA_UNALIGNED` | mmap needs alignment | llama aborts at load |
| `E_TOKENIZER_COUNT_MISMATCH` | parallel arrays unchecked | scores/tokens desync ⇒ piece chaos |
| `E_TEMPLATE_SPECIAL_TOKENS_UNREGISTERED` | template unchecked | chat boundary splits into pieces |

All severity codes live in one registry (`sentinel/findings.py`) and each has a
one-line definition + upstream origin in the built-in `explain` command.

## Integrity by construction

The quantization block table and the metadata-key registry are **derived, not
retyped**: `tools/derive_types.py` / `tools/derive_keys.py` AST-parse vendored
copies of upstream (`ggml.h`, `gguf.h`, gguf-py constants — see
`tools/vendor/MANIFEST.md`) so no quant block size or key string is ever
hand-typed. `tools/audit.py` pins the vendored sources' digest and re-checks
that every module is exactly what was derived.

## Install & use

Python ≥ 3.9, zero runtime dependencies.

```bash
python3 -m pip install -e .
sentinel scan path/to/model.gguf            # human output
sentinel scan --format json model.gguf      # machine output
sentinel scan --format github --fail-on warn *.gguf   # CI gate ($GITHUB_STEP::warning)
sentinel explain E_HEAD_DIM_MISMATCH       # what a code means, + upstream origin
sentinel types                             # pinned quant block table (bits/weight)
sentinel generate --out demo.gguf          # synthetic model for docs/tests
sentinel fuzz --iterations 500 --seed 7    # mutation-fuzz the parser, archives crashers
```

Exit codes: `0` clean · `1` ≥ fail-on severity · `2` below fail-on but present ·
`3` fatal (unparseable / oversized).

## The fuzzer

14 named corruption classes (`magic`, `truncated_header`, `tensor_type`,
`tensor_offset`, `tensor_dim`, `count_kv`, `append`, …) applied to synthetic
models built by `sentinel/generate.py`. Every payload is scanned in a clean
subprocess so a real crash is *proved*, not suspected: a coded `SentinelError`
is a pass, any other exception is a crasher, archived with its seed and
reproducible via `python3 -m sentinel.fuzz --replay crashers/crash_xxx.gguf`.

## Tests

Plain asserts; no pytest required to run them:

```bash
python3 tests/run_all.py          # dep-free runner
python3 -m pytest tests -q        # if you have pytest
python3 tools/gate.py             # compile -> import -> behavioural probes -> mutation roundtrip
python3 tools/audit.py            # vendored-source digest + derivation audit
```

The hand-built byte fixtures in `tests/_bytes.py` are typed straight from the
vendored `gguf.h`, on purpose: if the parser and the generator ever share a
misreading of the format, the round-trip tests would hide it, the hand-built
ones won't.

## Layout

```
sentinel/
  reader.py    bound-checked byte cursor (no unchecked struct.unpack)
  parser.py    header/kv/tensor-info, recovery vs fatal split
  rules.py     17 semantic rules (the value of the project)
  findings.py  code -> severity registry, dedup keys
  report.py    text / json / github-annotation renderers, exit codes
  cli.py       scan | explain | types | generate | fuzz
  generate.py  synthetic GGUF writer + named mutation engine
  fuzz.py      subprocess-isolated mutation fuzzer
  registry.py  quant block table  (_type_table.py: derived)
  keys.py      metadata key registry  (_key_table.py: derived)
tools/         derivation + audit + gate scripts, vendored upstream (pinned)
```

## Limitations (honest)

- Validates structure/metadata/geometry, not tensor *contents* (no
  per-block NaN scan, no perplexity). Degenerate all-zero weight regions are
  flagged as a warning only.
- GGUF v3 only; v1/v2 pre-alignment files report `E_VERSION`.
- Some multimodal/BPE extensions are checked only for presence/shape.

MIT licensed.
