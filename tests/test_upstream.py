"""The upstream-pinned corpus: the only tests that exercise the parser against
bytes this project did not write.

Four genuine GGUF files from the very repository the type table in
tools/vendor/ is derived from, pinned to an exact commit AND to a sha256 each.
The digest check is the point: without it a "passes on real files" claim is
unfalsifiable, because the URL could be serving different bytes tomorrow. A
silent upstream edit therefore fails this test loudly instead of quietly
turning the claim into a fiction.

Network is a legitimate dependency of a real-world corpus test, so an
unreachable upstream skips -- an unreachable network is not a verdict about
the parser. A digest mismatch or a fatal scan is a hard failure.
"""
import hashlib
import urllib.error
import urllib.request

from _shared import ROOT  # noqa: F401  (sys.path wiring)
from sentinel.cli import scan_bytes
from sentinel.findings import SentinelError

COMMIT = "95887577ab5fead779581a7030a83c7752ff3234"
REPO = "ggml-org/llama.cpp"
CORPUS = (
    ("models/ggml-vocab-bert-bge.gguf",
     "fbcbe22278fb302694d5f4a41bfe48c5f90e8e3554eab1c0435387dff654a854"),
    ("models/ggml-vocab-llama-spm.gguf",
     "16c3724582d59aa8bf84711894e833f916ee46a31d80e21312759c48bf8d0e69"),
    ("models/ggml-vocab-phi-3.gguf",
     "967d7190d11c4842eab697079d98d56c2116e10eb617be355a2733bfc132e326"),
    ("models/ggml-vocab-refact.gguf",
     "ac3ceda902fed91ccf74312b305d9b86c37e4f8e35fa9cc6ef3ce34fca7d4678"),
)


def _fetch(path):
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "gguf-sentinel test"})
    return urllib.request.urlopen(req, timeout=90).read()


def test_upstream_corpus_is_complete_and_pinned():
    assert len(CORPUS) == 4
    assert all(len(d) == 64 and all(c in "0123456789abcdef" for c in d)
               for _, d in CORPUS)
    assert len(COMMIT) == 40


def test_validator_passes_on_upstream_authored_files():
    for path, digest in CORPUS:
        try:
            blob = _fetch(path)
        except (urllib.error.URLError, OSError) as exc:
            print(f"SKIP (network: {type(exc).__name__}) -- {path}")
            return
        name = path.rsplit("/", 1)[-1]
        assert hashlib.sha256(blob).hexdigest() == digest, (
            f"{name}: upstream bytes changed under the pin -- the "
            f"'passes on real files' claim is no longer true")
        try:
            doc, findings = scan_bytes(blob, filename=name)
        except SentinelError as exc:
            raise AssertionError(
                f"{name}: upstream-authored file was rejected as fatal "
                f"{exc.code} -- a parser bug, not a broken download") from exc
        errors = [f.code for f in findings if f.severity == "error"]
        assert not errors, f"{name}: error findings on a valid file: {errors}"
        # metadata really did parse -- an empty document would pass vacuously
        assert doc.kv, f"{name}: parsed zero KV pairs"
        assert doc.version == 3, f"{name}: version {doc.version}"
        assert doc.alignment > 0, f"{name}: alignment {doc.alignment}"
        print(f"  ok {name}: {len(doc.kv)} kv, {len(doc.tensors)} tensors, "
              f"align={doc.alignment}, {len(findings)} warn/info")
