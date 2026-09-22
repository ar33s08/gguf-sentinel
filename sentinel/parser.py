"""GGUF format parser: header, KV pairs, tensor-info table, alignment layout.

Design: recovery-first. Every field read happens at a recorded offset, every
length field is bounds-checked before it is trusted (counts against remaining
bytes, int64 sign screening, sane-count ceiling), and structural damage that
permits continuation yields Findings rather than aborting -- a file with three
broken tensors still gets its healthy metadata audited, which is what triage
users actually need.

Layout (pinned to the vendored gguf.h):
    magic 'GGUF' | u32 version | u64 n_tensors | u64 n_kv | u32 alignment
    n_kv      x KV: string key | u32 gguf_type | value
    n_tensors x TI: string name | u32 n_dims | u64 dims | u32 type | u64 offset
    zero pad to alignment, then tensor data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from . import registry
from .findings import SentinelError, make
from .reader import BadString, Eof, OverAlloc, Reader

MAGIC = b"GGUF"
GGUF_VERSION_MAX = 3
INT64_SIGN_BIT = 1 << 63
COUNT_SANITY = 50_000_000

# gguf_type enum discriminants (vendored gguf.h enum gguf_type)
GGUF_TYPE_NAMES = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
    7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64",
}
_SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
               10: "<Q", 11: "<q", 12: "<d"}


@dataclass(frozen=True)
class KvEntry:
    key: str
    gguf_type: int
    value: Any
    type_name: str
    entry_offset: int
    value_offset: int


@dataclass(frozen=True)
class TensorInfo:
    name: str
    n_dims: int
    dims: tuple            # exactly 4, trailing padded with 1
    declared_dims: tuple   # as stored on disk
    type_id: int
    offset: int            # offset into the data section (u64 as stored)
    entry_offset: int
    dims_offset: int
    type_offset: int
    offset_offset: int


@dataclass
class ParsedModel:
    filename: Optional[str]
    size: int
    version: int
    n_tensors_declared: int
    n_kv_declared: int
    alignment: int
    meta_end: int          # end of tensor-info section, before alignment pad
    data_start: int        # first byte of the tensor data section
    kv: tuple              # tuple[KvEntry]
    kv_map: dict           # first-occurrence semantics, like gguf_find_key
    tensors: tuple         # tuple[TensorInfo]
    findings: tuple = ()   # recovered problems discovered while parsing

    def kv_get(self, key, default=None):
        row = self.kv_map.get(key)
        return default if row is None else row.value

    def kv_typed(self, key):
        """Return (value, gguf_type) with first-occurrence semantics."""
        row = self.kv_map.get(key)
        if row is None:
            return None, None
        return row.value, row.gguf_type


def _align_up(pos: int, alignment: int) -> int:
    if alignment <= 0:
        return pos
    rem = pos % alignment
    return pos if rem == 0 else pos + (alignment - rem)


def _read_count(rd, what: str, remaining_bytes: int, min_each: int,
                findings: list) -> int:
    raw = rd.u64()
    if raw >= INT64_SIGN_BIT:
        raise SentinelError("E_NEGATIVE_COUNT",
                            f"{what} count {raw} has the int64 sign bit set "
                            f"(stored as 2**63 - n)",
                            offset=rd.pos - 8)
    if raw > COUNT_SANITY:
        findings.append(make("E_HUGE_ALLOC",
                             f"{what} count {raw} exceeds the sanity ceiling "
                             f"{COUNT_SANITY}", key=what, offset=rd.pos - 8))
        raise SentinelError("E_HUGE_ALLOC",
                            f"{what} count {raw} exceeds sanity ceiling")
    if raw * min_each > remaining_bytes:
        findings.append(make("E_HUGE_ALLOC",
                             f"{what} count {raw} cannot fit: needs at least "
                             f"{raw * min_each} bytes, only {remaining_bytes} left",
                             key=what, offset=rd.pos - 8))
        raise SentinelError("E_HUGE_ALLOC",
                             f"{what} count {raw} exceeds remaining file size")
    return int(raw)


def _read_string(rd) -> str:
    len_off = rd.pos
    raw_len = rd.u64()
    if raw_len >= INT64_SIGN_BIT:
        raise SentinelError("E_NEGATIVE_COUNT", "string length has the int64 sign bit set",
                            offset=len_off)
    return rd.string_from(raw_len)


def _read_value(rd, type_id: int, findings: list, keyname: str, offset_of_value: int):
    name = GGUF_TYPE_NAMES.get(type_id)
    if name is None:
        raise SentinelError("E_BAD_KV_TYPE",
                            f"unknown gguf_type {type_id} for key {keyname!r}",
                            offset=rd.pos - 4)
    fmt = _SCALAR_FMT.get(type_id)
    if fmt is not None:
        return rd.scalar(fmt)
    if type_id == 7:  # bool: one byte, non-zero is true (upstream gguf_get_bool)
        return rd.u8() != 0
    if type_id == 8:
        return _read_string(rd)
    if type_id == 9:  # array: u32 elem_type | u64 count | elements
        elem_type = rd.u32()
        count = rd.u64()
        if elem_type == 9 or elem_type not in GGUF_TYPE_NAMES:
            raise SentinelError("E_BAD_KV_TYPE",
                                 f"bad array element type {elem_type} in {keyname!r}",
                                 offset=rd.pos - 12)
        if count >= INT64_SIGN_BIT:
            raise SentinelError("E_NEGATIVE_COUNT",
                                f"array length has the int64 sign bit in {keyname!r}",
                                offset=rd.pos - 8)
        if count > COUNT_SANITY:
            raise SentinelError("E_HUGE_ALLOC",
                                f"array length {count} in {keyname!r} exceeds sanity ceiling",
                                offset=rd.pos - 8)
        elem_min = 4 if elem_type == 8 else (1 if elem_type in (0, 1, 7) else 2)
        if count * elem_min > rd.remaining():
            raise SentinelError("E_TRUNCATED_KV_VALUE",
                                 f"array {keyname!r} claims {count} x >= {elem_min}B "
                                 f"= {count * elem_min} bytes, only {rd.remaining()} left",
                                 offset=offset_of_value)
        out = []
        for _ in range(int(count)):
            if elem_type == 7:
                out.append(rd.u8() != 0)
            elif elem_type == 8:
                out.append(_read_string(rd))
            else:
                out.append(rd.scalar(_SCALAR_FMT[elem_type]))
        return tuple(out)
    raise SentinelError("E_BAD_KV_TYPE", f"unhandled gguf_type {type_id}")


def parse_gguf(buf: bytes, *, filename: Optional[str] = None) -> ParsedModel:
    """Parse metadata sections. Raises SentinelError on unrecoverable damage,
    returns a ParsedModel carrying recovered Findings otherwise."""
    size = len(buf)
    if size == 0:
        raise SentinelError("E_ZERO_FILE", "file is empty")
    if size < 4:
        raise SentinelError("E_TRUNCATED_HEADER",
                             f"file shorter than the 4-byte magic ({size} bytes)")
    rd = Reader(buf)
    findings: list = []

    if rd.read_exact(4) != MAGIC:
        raise SentinelError("E_MAGIC", "file does not start with the GGUF magic")
    try:
        version = rd.u32()
        n_tensors = _read_count(rd, "n_tensors", rd.remaining(), 16, findings)
        n_kv = _read_count(rd, "n_kv", rd.remaining(), 13, findings)
        alignment = rd.u32()
    except Eof as exc:
        raise SentinelError("E_TRUNCATED_HEADER",
                             f"file ends inside the 28-byte header: {exc}",
                             offset=exc.offset) from exc
    except OverAlloc as exc:
        raise SentinelError("E_HUGE_ALLOC", f"header field too large: {exc}",
                             offset=exc.offset) from exc
    if alignment == 0 or (alignment & (alignment - 1)) != 0:
        findings.append(make("W_NO_ALIGNMENT",
                              f"general.alignment is {alignment}, not a positive power of two",
                              key="general.alignment", offset=rd.pos - 4,
                              expected="power of two", actual=alignment))

    kv_entries = []
    kv_map = {}
    try:
        for _ in range(int(n_kv)):
            entry_off = rd.pos
            key = _read_string(rd)
            type_id = rd.u32()
            value_off = rd.pos
            value = _read_value(rd, type_id, findings, key, value_off)
            entry = KvEntry(key=key, gguf_type=type_id, value=value,
                            type_name=GGUF_TYPE_NAMES.get(type_id, "?"),
                            entry_offset=entry_off, value_offset=value_off)
            kv_entries.append(entry)
            if key not in kv_map:  # gguf_find_key returns the FIRST match
                kv_map[key] = entry
    except Eof as exc:
        raise SentinelError("E_TRUNCATED_KV", f"kv section truncated: {exc}",
                             offset=exc.offset) from exc
    except OverAlloc as exc:
        raise SentinelError("E_HUGE_ALLOC", f"kv read too large: {exc}",
                             offset=exc.offset) from exc
    except BadString as exc:
        raise SentinelError("E_BAD_STRING", f"invalid utf8 in kv: {exc}",
                             offset=exc.offset) from exc

    tensors = []
    try:
        for _ in range(int(n_tensors)):
            entry_off = rd.pos
            name = _read_string(rd)
            n_dims = rd.u32()
            if n_dims == 0 or n_dims > registry.GGML_MAX_DIMS:
                raise SentinelError("E_TOO_MANY_DIMS",
                                     f"tensor {name!r} declares n_dims={n_dims} "
                                     f"(valid 1..{registry.GGML_MAX_DIMS})",
                                     tensor=name, offset=entry_off)
            dims_off = rd.pos
            declared = tuple(rd.u64() for _ in range(int(n_dims)))
            for d in declared:
                if d >= INT64_SIGN_BIT:
                    raise SentinelError("E_NEGATIVE_DIM",
                                         f"tensor {name!r} dim has the int64 sign bit set",
                                         tensor=name, offset=dims_off)
            type_off = rd.pos
            type_id = rd.u32()
            off_off = rd.pos
            tensor_off = rd.u64()
            if tensor_off >= INT64_SIGN_BIT:
                raise SentinelError("E_NEGATIVE_COUNT",
                                     f"tensor {name!r} has a negative data offset",
                                     tensor=name, offset=off_off)
            full = declared + (1,) * (registry.GGML_MAX_DIMS - len(declared))
            tensors.append(TensorInfo(
                name=name, n_dims=int(n_dims), dims=tuple(full),
                declared_dims=declared, type_id=int(type_id),
                offset=int(tensor_off), entry_offset=entry_off,
                dims_offset=dims_off, type_offset=type_off, offset_offset=off_off))
    except Eof as exc:
        raise SentinelError("E_TRUNCATED_TENSOR_INFO",
                             f"tensor-info table truncated: {exc}",
                             offset=exc.offset) from exc
    except OverAlloc as exc:
        raise SentinelError("E_HUGE_ALLOC", f"tensor-info read too large: {exc}",
                             offset=exc.offset) from exc
    except BadString as exc:
        raise SentinelError("E_BAD_STRING", f"invalid utf8 in tensor name: {exc}",
                             offset=exc.offset) from exc

    meta_end = rd.pos
    data_start = _align_up(meta_end, alignment) if n_tensors > 0 else meta_end
    return ParsedModel(
        filename=filename, size=size, version=version,
        n_tensors_declared=int(n_tensors), n_kv_declared=int(n_kv),
        alignment=alignment, meta_end=meta_end, data_start=data_start,
        kv=tuple(kv_entries), kv_map=kv_map, tensors=tuple(tensors),
        findings=tuple(findings))


def tensor_bytes(info: TensorInfo) -> Optional[int]:
    """Exact on-disk byte size of a tensor's data, or None when its geometry
    is unusable (non-positive dim, unknown/removed type, block-misaligned)."""
    if info.n_dims == 0 or any(d <= 0 for d in info.declared_dims):
        return None
    spec = registry.typeinfo(info.type_id)
    if spec is None:
        return None
    rows = 1
    for d in info.dims[1:]:
        rows *= d
    per_row = registry.row_bytes(info.type_id, info.dims[0])
    if per_row is None:
        return None
    return per_row * rows
