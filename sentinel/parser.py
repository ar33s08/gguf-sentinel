"""GGUF format parser: fixed header, KV section, tensor-info table.

The parser is *recovery-first*: every structural element records the file
offsets of its own fields, so malformed entries produce precise findings
(with byte offsets) instead of a crash, and a single bad tensor does not
stop analysis of the remaining ones. Only truly unrecoverable damage (bad
magic, bad version, truncation mid-entry, an unknown type discriminant that
makes the remaining stream unparsable) raises SentinelError.

Field layout implemented (pinned to the vendored gguf.h / ggml.h):

    [0]      magic   "GGUF"
    [4]      version u32 (1..3)
    [8]      n_tensors u64
    [16]     n_kv      u64
    [24]     alignment u32   (present only when n_tensors > 0)
    KV pairs (n_kv):   key string, gguf_type u32, value
    tensor infos (n_tensors): name string, n_dims u32, dims u64[n_dims],
                              ggml_type u32, data offset u64
    padding to alignment, then the tensor data section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import registry
from .findings import SentinelError, make
from .reader import BadString, Eof, OverAlloc, Reader

MAGIC = b"GGUF"
GGUF_VERSION_MAX = 3
HEADER_FIXED = 24          # magic + version + n_tensors + n_kv
ALIGNMENT_FIELD = 4        # u32 alignment when tensors exist

# gguf_type discriminants (KV value types) from include/gguf.h
GGUF_TYPES = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
    7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64",
}
_SCALAR_READERS = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
    10: "u64", 11: "i64", 12: "f64",
}
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                 7: 1, 10: 8, 11: 8, 12: 8}

# sanity bound for entry counts: 2**63 means an int64 -1 leaked in; beyond
# 2**31 entries can not physically fit in any file we will ever see
INT64_SIGN_BIT = 2 ** 63
COUNT_SANITY = 2 ** 31


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
    dims: tuple            # exactly GGML_MAX_DIMS (4) values, trailing 1s
    declared_dims: tuple   # as stored (before padding with 1s)
    type_id: int
    offset: int
    entry_offset: int      # where the tensor-info record starts
    dims_offset: int       # where dims begin (after n_dims field)
    type_offset: int
    offset_offset: int     # where the u64 data-offset field begins


@dataclass
class ParsedModel:
    filename: Optional[str]
    size: int
    version: int
    n_tensors_declared: int
    n_kv_declared: int
    alignment: int
    meta_end: int          # end of tensor-info section, before padding
    data_start: int        # first byte of the tensor data section
    kv: tuple              # tuple[KvEntry]
    kv_map: dict           # first-occurrence semantics, like gguf_find_key
    tensors: tuple         # tuple[TensorInfo]

    def kv_get(self, key, default=None):
        row = self.kv_map.get(key)
        return default if row is None else row.value


def _align_up(pos: int, alignment: int) -> int:
    if alignment <= 0:
        return pos
    rem = pos % alignment
    return pos if rem == 0 else pos + (alignment - rem)


def _read_count(rd, what: str, remaining_for_estimate: int, min_each: int,
                findings: list) -> int:
    raw = rd.u64()
    if raw >= INT64_SIGN_BIT:
        raise SentinelError("E_NEGATIVE_COUNT",
                           f"{what} count {raw} is a negative int64 (2**63 - n)",
                           offset=raw)
    if raw > COUNT_SANITY:
        findings.append(make("E_HUGE_ALLOC",
                             f"{what} count {raw} exceeds sanity bound {COUNT_SANITY}",
                             key=what, offset=rd.pos))
        raise SentinelError("E_HUGE_ALLOC", f"{what} count {raw} exceeds sanity bound")
    if raw * min_each > remaining_for_estimate:
        findings.append(make("E_HUGE_ALLOC",
                            f"{what} count {raw} cannot fit: needs at least "
                            f"{raw * min_each} bytes, only {remaining_for_estimate} left",
                            key=what, offset=rd.pos))
        raise SentinelError("E_HUGE_ALLOC",
                            f"{what} count {raw} exceeds remaining file size")
    return int(raw)


def _read_string(rd) -> str:
    raw_len = rd.u64()
    if raw_len >= INT64_SIGN_BIT:
        raise SentinelError("E_NEGATIVE_COUNT", "string length is a negative int64",
                            offset=rd.pos)
    return rd.string_from(raw_len)


def _read_value(rd, type_id: int, findings: list, keyname: str):
    name = GGUF_TYPES.get(type_id)
    if name is None:
        raise SentinelError("E_BAD_KV_TYPE",
                            f"unknown gguf_type {type_id} for key {keyname!r}",
                            offset=rd.pos - 4)
    reader_name = _SCALAR_READERS.get(type_id)
    if reader_name is not None:
        return getattr(rd, reader_name)()
    if type_id == 7:  # bool: one byte, non-zero is true
        return rd.u8() != 0
    if type_id == 8:  # string
        return _read_string(rd)
    if type_id == 9:  # array
        elem_type = rd.u32()
        count = rd.u64()
        if elem_type not in GGUF_TYPES or elem_type == 9:
            raise SentinelError("E_BAD_KV_TYPE",
                               f"bad array element type {elem_type} in {keyname!r}",
                               offset=rd.pos - 12)
        if count >= INT64_SIGN_BIT:
            raise SentinelError("E_NEGATIVE_COUNT",
                               f"array length is a negative int64 in {keyname!r}")
        if count > COUNT_SANITY:
            raise SentinelError("E_HUGE_ALLOC",
                               f"array length {count} in {keyname!r} exceeds sanity bound")
        elem_min = _SCALAR_SIZES.get(elem_type, 1)
        if count * elem_min > rd.remaining():
            raise SentinelError("E_TRUNCATED_KV_VALUE",
                               f"array {keyname!r} claims {count} x {elem_min}B "
                               f"= {count * elem_min} bytes, only {rd.remaining()} left")
        out = []
        for _ in range(int(count)):
            if elem_type == 9:
                raise SentinelError("E_BAD_KV_TYPE", "nested arrays are not supported")
            if elem_type == 7:
                out.append(rd.u8() != 0)
            elif elem_type == 8:
                out.append(_read_string(rd))
            else:
                out.append(getattr(rd, _SCALAR_READERS[elem_type])())
        return tuple(out)
    raise SentinelError("E_BAD_KV_TYPE", f"unhandled gguf_type {type_id}")


def parse_gguf(buf: bytes, *, filename: Optional[str] = None,
               max_chunk: Optional[int] = None) -> ParsedModel:
    """Parse the metadata sections. Raises SentinelError on unrecoverable damage."""
    size = len(buf)
    if size == 0:
        raise SentinelError("E_ZERO_FILE", "file is empty")
    if size < 4:
        raise SentinelError("E_TRUNCATED_HEADER",
                            f"file shorter than the 4-byte magic ({size} bytes)")
    rd = Reader(buf, max_chunk=max_chunk if max_chunk is not None else max(size, 1))
    if rd.read_exact(4) != MAGIC:
        raise SentinelError("E_MAGIC", "file does not start with the GGUF magic")
    try:
        version = rd.u32()
        if version == 0 or version > GGUF_VERSION_MAX:
            raise SentinelError("E_VERSION",
                               f"unsupported GGUF version {version} (expected 1..{GGUF_VERSION_MAX})",
                               offset=4)
        n_tensors = _read_count(rd, "n_tensors", rd.remaining(), 16, [])
        n_kv = _read_count(rd, "n_kv", rd.remaining(), 13, [])
        alignment = registry.GGML_DEFAULT_ALIGNMENT if n_tensors > 0 else 0
        if n_tensors > 0:
            file_alignment = rd.u32()
            if file_alignment > 0:
                alignment = int(file_alignment)

        kv_entries = []
        kv_map = {}
        for _ in range(int(n_kv)):
            entry_off = rd.pos
            key = _read_string(rd)
            type_off = rd.pos
            type_id = rd.u32()
            value_off = rd.pos
            value = _read_value(rd, type_id, [], key)
            entry = KvEntry(key=key, gguf_type=type_id, value=value,
                           type_name=GGUF_TYPES[type_id], entry_offset=entry_off,
                           value_offset=value_off)
            kv_entries.append(entry)
            if key not in kv_map:  # gguf_find_key returns the first match
                kv_map[key] = entry

        tensors = []
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

        meta_end = rd.pos
        data_start = _align_up(meta_end, alignment) if n_tensors > 0 else meta_end
        return ParsedModel(
            filename=filename, size=size, version=version,
            n_tensors_declared=int(n_tensors), n_kv_declared=int(n_kv),
            alignment=alignment, meta_end=meta_end, data_start=data_start,
            kv=tuple(kv_entries), kv_map=kv_map, tensors=tuple(tensors))
    except Eof as exc:
        raise SentinelError("E_TRUNCATED_KV", str(exc), offset=exc.offset) from exc
    except OverAlloc as exc:
        raise SentinelError("E_HUGE_ALLOC", str(exc), offset=exc.offset) from exc


def tensor_bytes(info: TensorInfo) -> Optional[int]:
    """Exact on-disk byte size of a tensor, or None when its geometry is
    unusable (non-positive dim, unknown/removed type, block misalignment)."""
    if any(d <= 0 for d in info.declared_dims):
        return None
    if info.n_dims == 0:
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
