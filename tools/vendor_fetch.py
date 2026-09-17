import re, urllib.request, hashlib, os

TAG = "v0.24.0"
DEST = os.path.expanduser("~/gguf-sentinel/tools/vendor")
os.makedirs(DEST, exist_ok=True)

def raw(url, accept=None):
    h = {"User-Agent": "gguf-sentinel-vendor"}
    if accept: h["Accept"] = accept
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=120).read()

FILES = {
    "ggml.h": (f"https://raw.githubusercontent.com/ggml-org/ggml/{TAG}/include/ggml.h", None),
    "ggml.c": (f"https://raw.githubusercontent.com/ggml-org/ggml/{TAG}/src/ggml.c", None),
    "ggml-common.h": (f"https://raw.githubusercontent.com/ggml-org/ggml/{TAG}/src/ggml-common.h", None),
    "gguf.h": (f"https://raw.githubusercontent.com/ggml-org/ggml/{TAG}/include/gguf.h", None),
    "gguf-py_constants.py.reference": (
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py", None),
}

lines = ["# Vendored upstream sources (pinned)", ""]
for name, (url, acc) in FILES.items():
    data = raw(url, acc)
    path = os.path.join(DEST, name)
    old = open(path, "rb").read() if os.path.exists(path) else b""
    status = "unchanged" if old == data else ("updated" if old else "new")
    open(path, "wb").write(data)
    digest = hashlib.sha256(data).hexdigest()
    lines.append(f"- `{name}` -- sha256 `{digest}` -- [{url}]({url})")
    print(name, len(data), digest[:12], status)

# sanity: the tokens derive_types.py depends on must be present
chk = open(os.path.join(DEST, "ggml.c"), "rb").read()
assert b"type_traits[GGML_TYPE_COUNT]" in chk
chk2 = open(os.path.join(DEST, "ggml-common.h"), "rb").read()
assert b"static_assert(sizeof(block_q4_0)" in chk2
assert b"GGML_TYPE_NVFP4" in open(os.path.join(DEST, "ggml.h"), "rb").read()
print("sanity tokens present")

lines.insert(1, "Pinned upstream copies the type table is *derived from* — see tools/derive_types.py.")
open(os.path.join(DEST, "MANIFEST.md"), "w").write("\n".join(lines) + "\n")
print("wrote MANIFEST.md")
