"""Parser contract: unrecoverable damage raises a coded SentinelError (never a
raw exception), and every header field has a bound. Codes measured, not hoped."""
import struct

from _shared import codes_of


def test_empty_file_is_zero_file():
    assert codes_of(b"") == {"E_ZERO_FILE"}


def test_magic_corruption_is_fatal_coded():
    m = bytearray(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 200)
    m[1] ^= 0x21
    assert codes_of(bytes(m)) == {"E_MAGIC"}


def test_short_header_is_truncated_not_crash():
    assert codes_of(b"GGUF\x03\x00\x00\x00") == {"E_TRUNCATED_HEADER"}


def test_huge_count_is_alloc_bounded():
    m = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                   + struct.pack("<Q", 2 ** 40) + struct.pack("<I", 32) + b"\x00" * 64)
    assert codes_of(bytes(m)) == {"E_HUGE_ALLOC"}


def test_negative_count_is_screened():
    m = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                   + struct.pack("<Q", 2 ** 63) + struct.pack("<I", 32) + b"\x00" * 64)
    assert codes_of(bytes(m)) == {"E_NEGATIVE_COUNT"}


def test_truncation_never_crashes_raw():
    from sentinel.generate import mutate
    import random
    from _shared import clean_model
    blob = clean_model()
    for seed in range(20):
        codes_of(mutate(blob, "truncate", random.Random(seed)))  # AssertionError on raw crash
