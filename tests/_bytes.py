"""Hand-built GGUF byte stream, INDEPENDENT of sentinel.generate.

Kept separate on purpose: if the parser and the writer share a systematic
misreading of the format, a round-trip through generate.py would hide it,
while a byte layout typed straight from the vendored gguf.h catches it.
"""
import struct

def s(b):            # u64 length + bytes
    if isinstance(b, str):
        b = b.encode("utf8")
    return struct.pack("<Q", len(b)) + b

def kv_str(key, val):
    return s(key) + struct.pack("<I", 8) + s(val)

def kv_u32(key, val):
    return s(key) + struct.pack("<I", 5) + struct.pack("<i", val)

def kv_arr_i32(key, vals):
    body = s(key) + struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", len(vals))
    body += b"".join(struct.pack("<i", v) for v in vals)
    return body

def tensor(name, dims, type_id, offset):
    rec = s(name) + struct.pack("<I", len(dims))
    for d in dims:
        rec += struct.pack("<Q", d)
    rec += struct.pack("<I", type_id) + struct.pack("<Q", offset)
    return rec

def build(n_kv, n_tensors, kvs, tensors, meta_tail=b"", data=b"", alignment=32):
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv)
    if n_tensors > 0:
        head += struct.pack("<I", alignment)
    meta = head + kvs + tensors + meta_tail
    pad = (-len(meta)) % alignment if n_tensors > 0 else 0
    return meta + b"\x00" * pad + data
