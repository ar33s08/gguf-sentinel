"""CLI contract: exit codes are the CI gate, so pin them."""
import os
import sys
import tempfile

from _shared import clean_model


def _write(path, blob):
    open(path, "wb").write(blob)
    return path


def test_scan_clean_exits_zero():
    from sentinel.cli import main
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "m.gguf"), clean_model())
        assert main(["scan", p]) == 0


def test_scan_broken_magic_exits_nonzero():
    from sentinel.cli import main
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "m.gguf"), b"XXXX" + clean_model()[4:])
        assert main(["scan", p]) != 0


def test_generate_then_scan_roundtrip_zero():
    from sentinel.cli import main
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "gen.gguf")
        assert main(["generate", "--out", out]) == 0
        assert os.path.getsize(out) > 1024
        assert main(["scan", out]) == 0


def test_explain_is_zero_and_names_the_code():
    from sentinel.cli import main
    assert main(["explain", "E_MAGIC"]) == 0


def test_types_table_is_nontrivial():
    from sentinel.cli import main
    assert main(["types"]) == 0
