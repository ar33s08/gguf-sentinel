"""Typed view over the generated ggml_type table (sentinel._type_table).

This is the ONLY module other code imports the table through: every consumer
goes via typeinfo()/iter_types() so the raw dict keys (stringified ints from
the generator) never leak and never get re-typed by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from . import _type_table

GGML_MAX_DIMS = 4
GGML_DEFAULT_ALIGNMENT = 32  # GGUF_DEFAULT_ALIGNMENT from the vendored gguf.h
GGML_QNT_VERSION = 2  # GGML_QNT_VERSION from the vendored ggml-common.h


@dataclass(frozen=True)
class TypeSpec:
    id: int
    enum: str          # e.g. "GGML_TYPE_Q4_K"
    name: str          # e.g. "q4_K"
    blck: int          # elements per quantization block
    size: int          # bytes per block
    quant: bool

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.size / self.blck


_BY_ID = {int(k): _type_table.TYPE_TABLE[str(k)] for k in _type_table.TYPE_TABLE}
_BY_NAME = {_type_table.TYPE_TABLE[k]["name"].lower(): _type_table.TYPE_TABLE[k]
            for k in _type_table.TYPE_TABLE}


def typeinfo(type_id: int) -> Optional[TypeSpec]:
    """Lookup by ggml_type id; None when the id is outside the pinned enum."""
    row = _BY_ID.get(int(type_id))
    if row is None:
        return None
    return TypeSpec(id=row["id"], enum=row["enum"], name=row["name"],
                    blck=row["blck"], size=row["size"], quant=row["quant"])


def typeinfo_by_name(name: str) -> Optional[TypeSpec]:
    """Lookup by canonical lowercase name (e.g. 'q4_k'); None if unknown."""
    row = _BY_NAME.get(name.lower())
    if row is None:
        return None
    return TypeSpec(id=row["id"], enum=row["enum"], name=row["name"],
                    blck=row["blck"], size=row["size"], quant=row["quant"])


def known_ids() -> frozenset:
    return frozenset(_BY_ID)


def all_types() -> Iterator[TypeSpec]:
    for type_id in sorted(_BY_ID):
        row = _BY_ID[type_id]
        yield TypeSpec(id=row["id"], enum=row["enum"], name=row["name"],
                       blck=row["blck"], size=row["size"], quant=row["quant"])


def row_bytes(type_id: int, n_elements: int) -> Optional[int]:
    """Exact on-disk bytes for a quantized tensor row of n_elements elements.

    For non-quantized types this is just n * element_size (blck == 1).
    Returns None when the element count is not a whole multiple of the block
    size (callers must report E_QUANT_ROW_NOT_BLOCK_ALIGNED) or the type id is
    unknown.
    """
    spec = typeinfo(type_id)
    if spec is None:
        return None
    if spec.blck <= 0 or spec.size <= 0:
        return None
    if n_elements % spec.blck != 0:
        return None
    return (n_elements // spec.blck) * spec.size


def vendored_digest() -> str:
    """Content hash of the vendored upstream sources the table came from.

    Pinned into a unit test: a vendored-source edit without a matching
    tools/derive_types.py regeneration fails CI.
    """
    return _type_table.VENDORED_SHA
