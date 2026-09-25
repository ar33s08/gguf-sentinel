"""Write -> read roundtrip: what the builder emits, the reader must accept
silently -- and no named corruption class may be inert."""
import random

from _shared import MUTATIONS, STRONG, clean_model, codes_of
from sentinel.generate import build_model, mutate


def test_generated_model_needs_no_recovery():
    assert codes_of(clean_model()) == set()


def test_read_is_deterministic():
    blob = clean_model()
    assert codes_of(blob) == codes_of(blob)


def test_no_mutation_kind_is_inert():
    # every corruption class must be detectable by at least one of 40 seeds;
    # the strict codes themselves are pinned in test_rules
    blob = build_model()
    for kind in MUTATIONS:
        hits = 0
        for seed in range(40):
            mutated = mutate(blob, kind, random.Random(seed))
            if mutated == blob:
                continue
            if codes_of(mutated):
                hits += 1
        assert hits >= 1, f"{kind} never detected"


def test_strict_table_covers_most_of_the_registry():
    assert len(STRONG) >= len(MUTATIONS) // 2
