"""Report assembly: three renderers + the exit-code contract.

The exit-code ladder is what lets CI gate on this tool:

    0  clean or info-only
    1  at least one error   -> the file must not be trusted
    2  warn-only            -> review before shipping
    3  the tool could not parse the file at all (SentinelError)

CI usage: `sentinel scan --fail-on error models/*.gguf` returns 1 the moment a
contributed model is structurally broken, so a quantization pipeline can refuse
to publish a bad artifact automatically.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

from .findings import Finding, counts
from .parser import ParsedModel

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_WARN = 2
EXIT_FATAL = 3

SEVERITY_ORDER = ("error", "warn", "info")
_BADGE = {"error": "ERROR", "warn": "WARN ", "info": "INFO "}


def worst_rank(findings: Sequence[Finding]) -> int:
    if not findings:
        return 0
    ranked = 0
    for f in findings:
        if f.severity == "error" and ranked < 2:
            ranked = 2
        elif f.severity == "warn" and ranked < 1:
            ranked = 1
    return ranked


def exit_code_for(findings: Sequence[Finding], *, fail_on: str = "error",
                  fatal: bool = False) -> int:
    """Map findings to the CLI exit code under the chosen strictness."""
    if fatal:
        return EXIT_FATAL
    threshold = {"error": 2, "warn": 1, "info": 0}.get(fail_on, 2)
    rank = worst_rank(findings)
    if rank >= threshold:
        return EXIT_ERROR if threshold == 2 else (EXIT_ERROR if rank == 2 else EXIT_WARN)
    return EXIT_OK


def summarize(doc: ParsedModel, findings: Sequence[Finding]) -> dict:
    c = counts(findings)
    quant = 0
    for t in doc.tensors:
        from . import registry
        spec = registry.typeinfo(t.type_id)
        if spec is not None and spec.quant:
            quant += 1
    return {
        "file": doc.filename,
        "size_bytes": doc.size,
        "version": doc.version,
        "architecture": doc.kv_get("general.architecture"),
        "n_tensors": doc.n_tensors_declared,
        "n_kv": doc.n_kv_declared,
        "quantized_tensors": quant,
        "alignment": doc.alignment,
        "data_start": doc.data_start,
        "counts": c,
    }


def render_text(doc: ParsedModel, findings: Sequence[Finding], *, color: bool = False) -> str:
    s = summarize(doc, findings)
    name = s["file"] or "<buffer>"
    lines = [f"{name}"]
    meta = (f"  gguf v{s['version']}  |  {s['n_tensors']} tensors "
            f"({s['quantized_tensors']} quantized)  |  {s['n_kv']} kv  |  "
            f"arch={s['architecture'] or '-'}  |  {s['size_bytes']:,} bytes")
    lines.append(meta)
    if not findings:
        lines.append("  clean: no findings")
    for f in sorted(findings, key=_sort_key):
        loc = _location(f)
        badge = _BADGE.get(f.severity, f.severity.upper())
        lines.append(f"  [{badge}] {f.code}: {f.message}{loc}")
    tally = f"  {s['counts']['error']} error / {s['counts']['warn']} warn / {s['counts']['info']} info"
    lines.append(tally)
    return "\n".join(lines)


def _sort_key(f: Finding) -> tuple:
    order = {"error": 0, "warn": 1, "info": 2}
    return (order.get(f.severity, 3), f.code, f.offset if f.offset is not None else 0)


def _location(f: Finding) -> str:
    bits = []
    if f.tensor:
        bits.append(f"tensor={f.tensor}")
    if f.key:
        bits.append(f"key={f.key}")
    if f.offset is not None:
        bits.append(f"@{f.offset}")
    if f.expected is not None and f.actual is not None:
        bits.append(f"expected={f.expected} actual={f.actual}")
    return ("  (" + ", ".join(bits) + ")") if bits else ""


def render_json(doc: Optional[ParsedModel], findings: Sequence[Finding], *,
                include_summary: bool = True) -> str:
    payload = {"findings": [f.as_dict() for f in sorted(findings, key=_sort_key)]}
    if include_summary and doc is not None:
        payload["summary"] = summarize(doc, findings)
    return json.dumps(payload, indent=2, sort_keys=False)


def render_github(doc: ParsedModel, findings: Sequence[Finding]) -> str:
    """GitHub Actions ::error/::warning annotations for one file.

    GGUF has no line numbers, so we annotate line 1 and carry the byte offset
    in the message -- enough for the Actions UI to group findings per file."""
    name = doc.filename or "<buffer>"
    out = []
    for f in sorted(findings, key=_sort_key):
        if f.severity == "info":
            continue
        level = "error" if f.severity == "error" else "warning"
        detail = _location(f).strip(" ()")
        out.append(f"::{level} file={name},line=1::{f.code} {f.message} [{detail}]")
    return "\n".join(out)
