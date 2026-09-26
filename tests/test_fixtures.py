"""The committed fixtures ARE the README console block, so the suite pins them
too, not just CI: clean model -> zero findings, broken model -> the exact code
the README shows. A refactor that silently changes either fails here loudly."""
import os

from sentinel.cli import scan_bytes
from sentinel.findings import SentinelError

_FX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def _codes(name):
    with open(os.path.join(_FX, name), "rb") as fh:
        buf = fh.read()
    try:
        _doc, fnd = scan_bytes(buf, filename=name)
    except SentinelError as exc:
        return {exc.code}
    return {f.code for f in fnd}


def test_demo_fixture_scans_clean():
    assert _codes("demo.gguf") == set()


def test_demo_broken_fixture_is_bad_tensor_type():
    assert "E_BAD_TENSOR_TYPE" in _codes("demo-broken.gguf")
