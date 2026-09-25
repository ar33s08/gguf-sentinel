"""Finding vocabulary: stable machine-readable codes + severities.

The code namespace is this tool's stable public contract: scripts and CI gates
match on ``code``, so codes are never renamed, only added. A unit test pins
the full set (tests/test_contract.py).

Severity ladder, mapped to the CLI exit-code contract in report.exit_rank_for:

    error -> 1   the file must not be trusted or shipped
    warn  -> 2   suspicious; review before use
    info  -> 0   informational only

Severities are plain strings (never an Enum member's value accessor) so the
generated table and the code namespace round-trip through JSON unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- severity vocabulary (plain strings + rank map; JSON-stable) ------------
SEV_INFO = "info"
SEV_WARN = "warn"
SEV_ERROR = "error"

SEVERITY_RANK = {SEV_INFO: 0, SEV_WARN: 1, SEV_ERROR: 2}
_ALL_SEVERITIES = (SEV_INFO, SEV_WARN, SEV_ERROR)


def rank_of(severity: str) -> int:
    try:
        return SEVERITY_RANK[severity]
    except LookupError:
        raise UnknownSeverity(severity)


class SentinelError(Exception):
    """Raised when analysis cannot continue (unrecoverable structure).

    Carries the finding code + byte offset so the CLI can render a single
    fatal finding with the same vocabulary as recovered ones."""

    def __init__(self, code: str, message: str, offset: int = None, tensor: str = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.offset = offset
        self.tensor = tensor


class UnknownSeverity(SentinelError):
    def __init__(self, severity):
        super().__init__("E_UNKNOWN_SEVERITY",
                         f"unknown severity: {severity!r} (expected one of {_ALL_SEVERITIES})")
        self.severity = severity


class UnknownCode(SentinelError):
    def __init__(self, code):
        super().__init__("E_UNKNOWN_CODE",
                         f"unregistered finding code: {code!r} (add it to CODE_SEVERITIES)")
        self.code = code


# --- code -> severity registry (single source of truth; add-only) ----------
CODE_SEVERITIES: dict[str, str] = {
    # structural / format (parser)
    "E_MAGIC": SEV_ERROR,
    "E_VERSION": SEV_ERROR,
    "E_TRUNCATED_HEADER": SEV_ERROR,
    "E_TRUNCATED_KV": SEV_ERROR,
    "E_TRUNCATED_TENSOR_INFO": SEV_ERROR,
    "E_TRUNCATED_KV_VALUE": SEV_ERROR,
    "E_BAD_KV_TYPE": SEV_ERROR,
    "E_BAD_STRING": SEV_ERROR,
    "E_BAD_TENSOR_TYPE": SEV_ERROR,
    "E_HUGE_ALLOC": SEV_ERROR,
    "E_NEGATIVE_DIM": SEV_ERROR,
    "E_TOO_MANY_DIMS": SEV_ERROR,
    "E_NEGATIVE_COUNT": SEV_ERROR,
    # offsets / sizes
    "E_DATA_UNALIGNED": SEV_ERROR,
    "E_TENSOR_PAST_EOF": SEV_ERROR,
    "E_TENSOR_OVERLAP": SEV_ERROR,
    "E_PADDING_NOT_ZERO": SEV_ERROR,
    "W_TENSOR_GAP": SEV_WARN,
    "W_DATA_OFFSET_MISMATCH": SEV_WARN,
    "E_ZERO_FILE": SEV_ERROR,
    "E_OVERSIZED": SEV_ERROR,
    # metadata semantics
    "W_NO_ARCH": SEV_WARN,
    "W_NO_QUANT_VERSION": SEV_WARN,
    "W_NO_ALIGNMENT": SEV_WARN,
    "E_ARCH_KEY_MISMATCH": SEV_ERROR,
    "W_INCONSISTENT_HYPERPARAM": SEV_WARN,
    "W_EMBED_TOKENS_ZERO_ROWS": SEV_WARN,
    # shape vs hyperparam (the money rules)
    "E_HEAD_DIM_MISMATCH": SEV_ERROR,
    "E_KV_HEAD_MISMATCH": SEV_ERROR,
    "E_FFN_DIM_MISMATCH": SEV_ERROR,
    "E_EMBED_ROWS_MISMATCH": SEV_ERROR,
    "E_BLOCK_COUNT_MISMATCH": SEV_ERROR,
    "W_HEAD_DIM_ODD": SEV_WARN,
    "W_KV_HEAD_DIM_ODD": SEV_WARN,
    # quant math
    "E_QUANT_ROW_NOT_BLOCK_ALIGNED": SEV_ERROR,
    "E_QUANT_SIZE_NOT_BLOCK_MULTIPLE": SEV_ERROR,
    "E_QUANT_VERSION_UNSUPPORTED": SEV_ERROR,
    # tokenizer sanity
    "W_VOCAB_MISMATCH": SEV_WARN,
    "W_SPECIAL_TOKEN_IDS_OUT_OF_RANGE": SEV_WARN,
    "W_NO_CHAT_TEMPLATE": SEV_WARN,
    "W_BOS_EOS_MISSING": SEV_WARN,
    "E_TOKENIZER_COUNT_MISMATCH": SEV_ERROR,
    # chat template
    "W_TEMPLATE_NO_SPECIAL_TOKENS": SEV_WARN,
    "E_TEMPLATE_SPECIAL_TOKENS_UNREGISTERED": SEV_ERROR,
    # file level
    "W_TRAILING_JUNK": SEV_WARN,
    "W_SUSPICIOUS_DENSITY": SEV_WARN,
    # fuzz/crash classification support
    "E_UNHANDLED": SEV_ERROR,
}


@dataclass(frozen=True)
class Finding:
    """One concrete problem found in one file. Immutable + hashable: the
    parser/rules layers may emit duplicates and report.py dedups on key_id()."""

    code: str
    message: str
    severity: str = SEV_WARN
    tensor: Optional[str] = None
    key: Optional[str] = None
    offset: Optional[int] = None
    expected: Optional[object] = None
    actual: Optional[object] = None

    def key_id(self) -> tuple:
        """Dedup key: same (code, tensor, key, offset) surfaces once."""
        return (self.code, self.tensor, self.key, self.offset)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "tensor": self.tensor,
            "key": self.key,
            "offset": self.offset,
            "expected": self.expected,
            "actual": self.actual,
        }


def make(code: str, message: str, *, tensor: Optional[str] = None,
         key: Optional[str] = None, offset: Optional[int] = None,
         expected: Optional[object] = None, actual: Optional[object] = None,
         severity: Optional[str] = None) -> Finding:
    """Build a Finding carrying the code's registered severity (or an override).

    Raises UnknownCode for unregistered codes on purpose: a typo in a rule
    should fail loudly at test time, not silently degrade the report.
    """
    if code not in CODE_SEVERITIES:
        raise UnknownCode(code)
    chosen = severity if severity is not None else CODE_SEVERITIES[code]
    if chosen not in SEVERITY_RANK:
        raise UnknownSeverity(chosen)
    return Finding(code=code, message=message, severity=chosen, tensor=tensor,
                   key=key, offset=offset, expected=expected, actual=actual)


def counts(findings) -> dict:
    """Severity histogram {'error': n, 'warn': n, 'info': n} (always all keys)."""
    out = {SEV_ERROR: 0, SEV_WARN: 0, SEV_INFO: 0}
    for f in findings:
        out[f.severity] += 1
    return out
