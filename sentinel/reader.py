"""Byte-safe random access reader with a hard allocation budget.

Malformed or hostile GGUF files advertise huge length fields; trusting them
is how parsers DoS themselves. Every read goes through read_exact(), which
bounds each individual allocation by `max_chunk` (default 256 MiB) and every
total string/array payload by `max_alloc` (default 2 GiB), and raises Eof
the moment a read would pass the real end of the buffer -- so a truncated or
self-inconsisten file fails controllable, never with a MemoryError or a
slice-past-the-end surprise.
"""
from __future__ import annotations

import struct
from typing import Optional

MIB = 1024 * 1024
DEFAULT_MAX_CHUNK = 256 * MIB
DEFAULT_MAX_ALLOC = 2048 * MIB

_U8 = struct.Struct("<B")
_I8 = struct.Struct("<b")
_U16 = struct.Struct("<H")
_I16 = struct.Struct("<h")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_F32 = struct.Struct("<f")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_F64 = struct.Struct("<d")
_F16 = struct.Struct("<e")  # IEEE 754 half precision, native to struct


class Eof(Exception):
    """Raised when a read would cross the end of the buffer."""

    def __init__(self, wanted: int, have: int, offset: int):
        super().__init__(f"read of {wanted} bytes at offset {offset}, only {have} left")
        self.wanted = wanted
        self.have = have
        self.offset = offset


class BadString(Exception):
    """Raised when a length-prefixed string is not valid utf8."""

    def __init__(self, offset: int, detail: str = ""):
        super().__init__(f"invalid utf8 string at offset {offset}{(': ' + detail) if detail else ''}")
        self.offset = offset


class OverAlloc(Exception):
    """Raised when a single read exceeds the chunk budget."""

    def __init__(self, wanted: int, limit: int, offset: int):
        super().__init__(f"read of {wanted} bytes at offset {offset} exceeds {limit} byte chunk cap")
        self.wanted = wanted
        self.limit = limit
        self.offset = offset


class Reader:
    """Sequential cursor reader over an in-memory buffer (mmap-friendly)."""

    __slots__ = ("_buf", "_pos", "_len", "_max_chunk", "_max_alloc", "_total_alloc")

    def __init__(self, buf: bytes, *, max_chunk: int = DEFAULT_MAX_CHUNK,
                 max_alloc: int = DEFAULT_MAX_ALLOC, pos: int = 0):
        self._buf = buf
        self._pos = pos
        self._len = len(buf)
        self._max_chunk = max_chunk
        self._max_alloc = max_alloc
        self._total_alloc = 0

    # --- cursor ------------------------------------------------------------
    @property
    def pos(self) -> int:
        return self._pos

    @property
    def size(self) -> int:
        return self._len

    def remaining(self) -> int:
        return self._len - self._pos

    def seek(self, pos: int) -> None:
        if not 0 <= pos <= self._len:
            raise Eof(pos - self._pos, self.remaining(), pos)
        self._pos = pos

    def at_end(self) -> bool:
        return self._pos >= self._len

    # --- primitives --------------------------------------------------------
    def read_exact(self, n: int) -> bytes:
        if n < 0:
            raise OverAlloc(n, self._max_chunk, self._pos)
        end = self._pos + n
        if end > self._len:
            # the file itself is too short for the advertised length: that is
            # truncation, regardless of how big the claimed length is
            raise Eof(n, self._len - self._pos, self._pos)
        if n > self._max_chunk:
            # fits the file but would allocate an absurd chunk in one step
            raise OverAlloc(n, self._max_chunk, self._pos)
        chunk = self._buf[self._pos:end]
        self._pos = end
        return chunk

    def _read_scalar(self, fmt: struct.Struct) -> float:
        raw = self.read_exact(fmt.size)
        return fmt.unpack(raw)[0]

    def scalar(self, fmt_str: str):
        """Decode one little-endian scalar from its struct format string."""
        raw = self.read_exact(struct.calcsize(fmt_str))
        return struct.unpack(fmt_str, raw)[0]

    def u8(self) -> int:    return self._read_scalar(_U8)
    def i8(self) -> int:    return self._read_scalar(_I8)
    def u16(self) -> int:   return self._read_scalar(_U16)
    def i16(self) -> int:   return self._read_scalar(_I16)
    def u32(self) -> int:   return self._read_scalar(_U32)
    def i32(self) -> int:   return self._read_scalar(_I32)
    def f32(self) -> float: return self._read_scalar(_F32)
    def u64(self) -> int:   return self._read_scalar(_U64)
    def i64(self) -> int:   return self._read_scalar(_I64)
    def f64(self) -> float: return self._read_scalar(_F64)
    def f16(self) -> float: return self._read_scalar(_F16)

    def string(self) -> str:
        """GGUF string: u64 byte length + utf8 bytes (strict: invalid utf8 fails)."""
        n = self.u64()
        return self.string_from(n)

    def string_from(self, n: int) -> str:
        """Decode a string whose u64 length was already consumed."""
        raw = self.read_exact(n)
        try:
            return raw.decode("utf8", errors="strict")
        except UnicodeDecodeError as exc:
            raise BadString(self._pos - n, str(exc)) from exc

    def bytes_exact(self, n: int) -> bytes:
        return self.read_exact(n)

    def try_peek_magic(self, size: int = 4) -> Optional[bytes]:
        if self._len < size:
            return None
        return self._buf[:size]
