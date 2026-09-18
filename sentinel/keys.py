"""Official GGUF metadata-key registry (runtime view over the derived table).

The key spellings and arch ids in sentinel._key_table are extracted from the
vendored llama.cpp gguf-py constants (tools/derive_keys.py) -- never hand
typed -- so rules can check key existence and {arch}-prefix consistency
against the same vocabulary llama.cpp itself ships.
"""
from __future__ import annotations

import re
from typing import Optional

from . import _key_table

KEY_PATTERNS: tuple = tuple(_key_table.KEY_PATTERNS)
ARCH_NAMES: frozenset = frozenset(_key_table.ARCH_NAMES)
KEYS_VENDORED_SHA = _key_table.KEYS_VENDORED_SHA

ARCH_PLACEHOLDER = "{arch}"

# exact (non-templated) keys, for membership tests
_EXACT_KEYS = frozenset(k for k in KEY_PATTERNS if "{" not in k)
# (prefix, remainder) pairs for the "{arch}.foo.bar" style keys
_TEMPLATE_KEYS = tuple(k for k in KEY_PATTERNS if ARCH_PLACEHOLDER in k)


def is_known_key(key: str, arch: Optional[str] = None) -> bool:
    """True when `key` is an official key, either exactly or via {arch}=arch."""
    if key in _EXACT_KEYS:
        return True
    if arch:
        templ = key.replace(ARCH_PLACEHOLDER, arch, 1)
        for pattern in _TEMPLATE_KEYS:
            if pattern.replace(ARCH_PLACEHOLDER, arch, 1) == templ:
                return True
    return False


def matches_arch_pattern(key: str, arch: str) -> bool:
    """Does `key` equal some official {arch}-template with {arch}=arch?"""
    target = ARCH_PLACEHOLDER + "."
    if not key.startswith(target):
        return False
    remainder = key[len(target):]
    for pattern in _TEMPLATE_KEYS:
        if pattern.split(ARCH_PLACEHOLDER + ".", 1)[-1] == remainder:
            if pattern.startswith(target):
                return True
    return False


def arch_scoped_keys(kv_keys) -> tuple:
    """Keys that LOOK arch-scoped: first dotted segment is an official arch id.

    Such keys are only official when written '<arch>.'; llama.cpp expects the
    literal '{arch}' placeholder expanded -- i.e. these keys are almost always
    a conversion bug (someone forgot to substitute the placeholder)."""
    hits = []
    for key in kv_keys:
        head = key.split(".", 1)[0]
        if head in ARCH_NAMES:
            hits.append(key)
    return tuple(hits)


def resolve(arch: str, suffix: str) -> str:
    """Build the concrete key for a template suffix, e.g.
    resolve('llama', 'attention.head_count') -> 'llama.attention.head_count'.

    Raises KeyError when the suffix is not an official {arch}-template tail."""
    template = ARCH_PLACEHOLDER + "." + suffix
    if template in _TEMPLATE_KEYS:
        return arch + "." + suffix
    raise KeyError(f"not an official arch-scoped key suffix: {suffix!r}")


def try_resolve(arch: str, suffix: str) -> Optional[str]:
    try:
        return resolve(arch, suffix)
    except KeyError:
        return None


def known_tensor_suffixes() -> frozenset:
    """Official per-block tensor name suffixes, derived from the upstream
    MODEL_TENSOR enum (lowercased). Used only for advisory checks -- an
    unknown suffix is suspicious, never fatal, because new architectures add
    tensors constantly."""
    return _TENSOR_SUFFIXES

# built at import time from the vendored constants file text (already shipped
# in _key_table via a second derived block); fall back to empty set so rules
# simply skip the advisory check if the harvest produced nothing.
_TENSOR_SUFFIXES = frozenset(getattr(_key_table, "TENSOR_SUFFIXES", ()))
