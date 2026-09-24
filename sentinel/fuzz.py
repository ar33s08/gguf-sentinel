"""Mutation fuzz harness: proves the parser never crashes on hostile bytes.

Subprocess isolation: a segfault/memory bomb inside a child kills only the
child; the parent records the crash byte and keeps feeding. Each iteration:
pick a base model, pick a mutation kind + seed, run scan() in the child, the
child prints OK or CRASH:exception-name. Anything that is not a SentinelError
is a BUG by definition -- the parser's contract is that every input either
parses or raises a clean, coded SentinelError.

    python3 -m sentinel.fuzz --iterations 500 --seed 7
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from typing import Optional

from .findings import SentinelError
from .generate import MUTATIONS, ModelSpec, build_model, mutate

_CHILD = """
import sys
sys.path.insert(0, {root!r})
from sentinel.cli import scan_bytes
from sentinel.findings import SentinelError
data = __import__('base64').b64decode(sys.argv[1].encode())
try:
    scan_bytes(data, filename='fuzz')
except SentinelError as exc:
    # a clean, coded rejection is the SUCCESS contract, never a crash
    print('OK')
except Exception as exc:
    print(('CRASH:' + type(exc).__name__ + ':' + str(exc)[:180]))
else:
    print('OK')
"""


def scan_child(src_root: str, payload: bytes, timeout: float = 20.0) -> tuple:
    """Run one payload in a clean interpreter. Returns (status, detail)."""
    import base64
    code = _CHILD.format(root=src_root)
    b64 = base64.b64encode(payload).decode("ascii")
    try:
        proc = subprocess.run([sys.executable, "-c", code, b64],
                               capture_output=True, timeout=timeout, text=True)
    except subprocess.TimeoutExpired as exc:
        return "timeout", str(exc)[:160]
    except Exception as exc:  # OSError spawning the child etc: still 'handled', but note it
        return "spawn_error", f"{type(exc).__name__}: {exc}"[:160]
    out = (proc.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    if last.startswith("CRASH:"):
        parts = last.split(":", 2)
        return "crash", f"{parts[1]}: {parts[2] if len(parts) > 2 else ''}"
    if proc.returncode != 0:
        return "hard_crash", f"rc={proc.returncode} {(proc.stderr or '')[-160:]}"
    if last == "OK":
        return "ok", ""
    return "odd_output", last[:160]


def _default_spec() -> ModelSpec:
    return ModelSpec(n_block=2, n_head=2, n_kv_head=1, n_embed=64,
                     n_ffn=128, n_vocab=256, head_dim=32)


def fuzz(seed: int = 0, iterations: int = 200, *, bases: Optional[list] = None,
         on_payload=None) -> dict:
    """Drive the fuzzer. Returns {'iterations','ok','bad':[...]}. on_payload
    receives (kind, seed, payload) for every non-ok round so CI can archive it."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rng = random.Random(seed)
    bases = bases or [_default_spec()]
    base_blobs = [build_model(sp, seed=rng.randrange(1 << 30)) for sp in bases]
    results = {"iterations": 0, "ok": 0, "bad": []}
    for i in range(iterations):
        kind = rng.choice(MUTATIONS)
        mseed = rng.randrange(1 << 30)
        blob = rng.choice(base_blobs)
        payload = mutate(blob, kind, random.Random(mseed))
        # second-order: sometimes stack a light random byte-flip too
        if rng.random() < 0.35:
            payload = mutate(payload, "flip", random.Random(mseed ^ 0x5EED))
        status, detail = scan_child(root, payload)
        results["iterations"] += 1
        if status == "ok":
            results["ok"] += 1
        else:
            results["bad"].append({"iteration": i, "kind": kind, "seed": mseed,
                                    "status": status, "detail": detail,
                                    "size": len(payload), "bytes": payload})
            if on_payload is not None:
                on_payload(kind, mseed, payload)
    return results


def replay(payload: bytes) -> str:
    """Re-run a saved crasher, printing the exception chain."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    status, detail = scan_child(root, payload)
    return f"{status}: {detail}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sentinel.fuzz")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None,
                     help="directory to archive non-clean payloads into")
    args = ap.parse_args(argv)

    import os
    def archiver(kind, mseed, payload):
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"crash_{kind}_{mseed}.gguf")
            with open(path, "wb") as fh:
                fh.write(payload)
            print(f"saved {path}")

    print(f"fuzzing: {args.iterations} iterations, seed={args.seed}, "
          f"bases=synthetic models, kinds={len(MUTATIONS)}")
    res = fuzz(seed=args.seed, iterations=args.iterations, on_payload=archiver)
    print(f"done: {res['ok']}/{res['iterations']} clean, {len(res['bad'])} offenders")
    for b in res["bad"][:12]:
        print(f"  [{b['status']}] kind={b['kind']} seed={b['seed']} "
              f"{b['size']} bytes :: {b['detail']}")
    return 1 if res["bad"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
