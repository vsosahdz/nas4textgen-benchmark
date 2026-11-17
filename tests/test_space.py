"""Invariants of the search space, encoding, and cost model.

These are the checks the prior work did not have, and each corresponds to a defect it
shipped: an encoding that admitted ``e = 8.0017``, a cost model that disagreed with
the models it described, and a parameter objective that depended on one gene out of
seven.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.space.cost import (
    embedding_share,
    kv_cache_bytes,
    parameter_count,
)
from nas4textgen.space.genotype import (
    DISTILGPT2,
    GENES,
    LEVELS,
    Genotype,
    decode,
    encode,
    enumerate_space,
    repair,
    valid_kv,
)

#: The real DistilGPT2 checkpoint's parameter count. The anchor must land on it
#: exactly, which is the strongest available test of the closed-form cost model.
DISTILGPT2_PARAMETERS = 81_912_576


@pytest.fixture(scope="module")
def space() -> list[Genotype]:
    return list(enumerate_space())


def test_anchor_reproduces_the_real_checkpoint_exactly():
    assert parameter_count(DISTILGPT2) == DISTILGPT2_PARAMETERS


def test_enumeration_is_deterministic():
    assert [g.as_tuple() for g in enumerate_space()] == [
        g.as_tuple() for g in enumerate_space()
    ]


def test_every_enumerated_genotype_is_valid(space):
    assert all(g.is_valid() for g in space)


def test_genotypes_are_unique(space):
    assert len({g.as_tuple() for g in space}) == len(space)


def test_encoding_is_injective_over_the_whole_space(space):
    """``decode(encode(g)) == g`` for every valid genotype, with no exceptions."""
    assert [decode(encode(g)) for g in space] == space


def test_structural_genes_are_never_fractional(space):
    """The defect that produced ``e = 8.0017`` in the prior work."""
    for g in space:
        for gene in GENES:
            value = getattr(g, gene)
            assert isinstance(value, int)
        assert isinstance(g.e, int)
        assert isinstance(g.intermediate, int)


def test_model_dimension_is_divisible_by_head_count(space):
    """Holds by construction because ``e`` is derived, not searched."""
    assert all(g.e % g.h == 0 for g in space)


def test_head_dimension_floor(space):
    """No architecture may have degenerate heads; the prior search collapsed to 1."""
    assert min(g.d_head for g in space) >= 32
    assert min(g.e for g in space) >= 64


def test_kv_divides_h(space):
    assert all(g.h % g.kv == 0 and g.kv <= g.h for g in space)


def test_multi_query_attention_is_reachable(space):
    """``kv = 1`` is the extreme of the memory axis and must exist for every ``h``."""
    assert all(1 in valid_kv(h) for h in LEVELS["h"])
    assert any(g.kv == 1 for g in space)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_repair_makes_any_random_vector_valid(seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        g = decode(rng.random(len(GENES)))
        assert g.is_valid()
        assert decode(encode(g)) == g, "repair must be idempotent on valid genotypes"


def test_repair_is_a_noop_on_valid_genotypes(space):
    assert all(repair(g) is g or repair(g) == g for g in space)


def test_repair_never_increases_memory_cost():
    """On a tie, repair prefers the smaller kv, so it cannot silently cost memory."""
    g = Genotype(h=8, d_head=64, n=4, ffn=2, kv=6)  # 6 does not divide 8
    assert not g.is_valid()
    assert repair(g).kv == 4


def test_embedding_no_longer_dominates_the_parameter_count(space):
    """The prior objective was 98.6% token embedding, so it had one effective gene."""
    shares = sorted(embedding_share(g) for g in space)
    median = shares[len(shares) // 2]
    assert median < 0.70, f"median embedding share {median:.1%} is still dominant"


def test_kv_cache_scales_as_declared():
    """Linear in layers, context, and kv heads; independent of the feed-forward."""
    base = Genotype(h=16, d_head=64, n=4, ffn=2, kv=4)
    assert kv_cache_bytes(base, context=512) * 2 == kv_cache_bytes(base, context=1024)
    wider_ffn = Genotype(h=16, d_head=64, n=4, ffn=4, kv=4)
    assert kv_cache_bytes(base) == kv_cache_bytes(wider_ffn)
    fewer_kv = Genotype(h=16, d_head=64, n=4, ffn=2, kv=2)
    assert kv_cache_bytes(fewer_kv) * 2 == kv_cache_bytes(base)


def test_context_length_must_be_explicit_in_reports():
    """A KV-cache figure without its context length is uninterpretable."""
    g = Genotype(h=8, d_head=64, n=2, ffn=2, kv=2)
    assert kv_cache_bytes(g, context=128) != kv_cache_bytes(g, context=256)


def test_objectives_conflict_at_matched_parameter_count(space):
    """The measurement that justifies the multi-objective formulation.

    Cost objectives do not conflict among themselves — everything shrinks as the model
    shrinks. The conflict lives at matched parameter count, and if it vanished the
    second objective would be redundant.
    """
    params = np.array([parameter_count(g) for g in space])
    kv = np.array([kv_cache_bytes(g) for g in space])
    band = (params >= 4.5e7) & (params < 5.5e7)
    assert band.sum() > 10
    assert kv[band].max() / kv[band].min() > 10.0


def test_cost_objectives_are_not_the_same_objective(space):
    """Rank correlation must stay clear of the gate's 0.90 threshold."""
    from scipy.stats import spearmanr

    params = [parameter_count(g) for g in space]
    kv = [kv_cache_bytes(g) for g in space]
    rho = spearmanr(params, kv).statistic
    assert abs(rho) < 0.90, f"params and kv_bytes are near-duplicates (rho={rho:.3f})"
