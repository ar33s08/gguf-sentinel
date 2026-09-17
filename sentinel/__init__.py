"""gguf-sentinel -- structural validator and triage tool for GGUF model files.

GGUF is the single-file container format used by the local-LLM ecosystem
(llama.cpp and everything built on it: Ollama, LM Studio, Jan, Kobold CPP,
...). The file carries model hyperparameters, tokenizer data, chat template
and quantized weights as a KV section plus a tensor table. There is no
standard "linter" for it: files downloaded from hubs or produced by custom
conversion/quantization pipelines routinely arrive subtly broken (inconsisten
tensor shapes vs declared hyperparameters, truncated tails, bad alignment,
or outright tampering), and users discover it at llama.cpp load time via a
crash or, worse, silently wrong inference.

This package parses and audits GGUF files **from first principles with zero
runtime dependencies**, using block-size tables derived programmatically from
the vendored upstream C sources under ``tools/vendor/``.

Modules
-------
registry    Authoritative ggml_type table (derived + self-digested).
reader      Struct-safe binary reader with hard byte budget.
parser      Format parser: header, KV pairs, tensor infos, alignment.
rules       Semantic lint rules beyond raw format parsing.
findings    Finding/severity vocabulary (stable, machine-readable codes).
report      Text / JSON / GitHub-annotation rendering + exit-code ladder.
generate    Synthetic GGUF builder + mutation engine (tests, fixtures, fuzz).
fuzz        Mutation fuzzing harness with subprocess isolation.
cli         The ``sentinel`` command line.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
