"""The gate, the exact front, and the sampling designs.

These run on CPU in seconds and are the core of the ``verify`` replication tier. Their
outputs must be bit-identical on any machine, which is asserted here rather than
claimed in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.space.designs import (
    fill_distance,
    latin_hypercube_design,
    maximin_order,
    uniform_random_design,
    waves,
)
from nas4textgen.space.enumerate import (
    PARAMETER_CEILING,
    enumerate_with_ceiling,
    non_dominated,
    reference_point,
    reference_span,
)
from nas4textgen.space.gate import (
    MAX_OBJECTIVE_CORRELATION,
    MAX_SOBOL_INDEX,
    run_gate,
)

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def report():
    return run_gate()


@pytest.fixture(scope="module")
def enumeration():
    return enumerate_with_ceiling()


def test_gate_passes(report):
    failed = [c.name for c in report.checks if not c.passed]
    assert not failed, f"failing checks: {failed}"


def test_gate_covers_all_eight_checks(report):
    assert len(report.checks) == 8


def test_gate_is_deterministic():
    """Same space, same report — the verify tier's bit-exactness claim."""
    first, second = run_gate().as_dict(), run_gate().as_dict()
    first.pop("provenance", None)
    second.pop("provenance", None)
    assert first == second


def test_no_gene_dominates_an_objective(report):
    check = next(c for c in report.checks if "sobol_first_order" in c.values)
    for objective, per_gene in check.values["sobol_first_order"].items():
        worst = max(per_gene.values())
        assert worst <= MAX_SOBOL_INDEX, f"{objective} is dominated by one gene: {per_gene}"


def test_no_gene_is_inert(report):
    check = next(c for c in report.checks if "median_relative_effect" in c.values)
    assert not check.values["inert_genes"]
    for gene, effects in check.values["median_relative_effect"].items():
        assert max(effects.values()) > 0.0, f"{gene} moves no objective"


def test_objectives_are_distinct(report):
    check = next(c for c in report.checks if "spearman" in c.values)
    assert abs(check.values["spearman"]) < MAX_OBJECTIVE_CORRELATION


def test_reference_point_dominates_the_whole_space(enumeration):
    """The prior [1e9, 0.0] left 12.3% of the space outside its own reference box."""
    objectives = enumeration.objectives
    reference = reference_point(objectives)
    assert np.all(np.all(objectives <= reference, axis=1))
    span = reference_span(objectives, reference)
    assert np.all(span > 0.01) and np.all(span < 0.99)


def test_ceiling_exclusion_is_recorded(enumeration):
    assert enumeration.ceiling == PARAMETER_CEILING
    assert enumeration.excluded > 0
    assert 0.0 < enumeration.excluded_fraction < 1.0
    assert enumeration.total_before_ceiling == len(enumeration) + enumeration.excluded


def test_cost_only_front_is_a_single_point(enumeration):
    """Cost objectives never conflict among themselves: everything shrinks together.

    If this ever returns more than one point, either a cost objective stopped being
    monotone in model size or the enumeration is wrong. The conflict that justifies the
    formulation lives at matched parameter count, not here.
    """
    assert len(non_dominated(enumeration.objectives)) == 1


def test_non_dominated_keeps_ties():
    """Two architectures with identical objectives are both on the front."""
    objectives = np.array([[10, 10], [10, 10], [20, 5], [30, 30]])
    front = non_dominated(objectives)
    assert set(front.tolist()) == {0, 1, 2}


def test_maximin_order_is_reproducible(enumeration):
    space = enumeration.genotypes
    assert np.array_equal(maximin_order(space, seed=0), maximin_order(space, seed=0))


def test_maximin_order_covers_the_space_exactly(enumeration):
    order = maximin_order(enumeration.genotypes, seed=0)
    assert sorted(order.tolist()) == list(range(len(enumeration)))


def test_waves_nest(enumeration):
    order = maximin_order(enumeration.genotypes, seed=0)
    sizes = [50, 150, 300, 600]
    produced = waves(order, sizes)
    for smaller, larger in zip(produced[:-1], produced[1:], strict=True):
        assert set(smaller.tolist()) <= set(larger.tolist())


def test_every_prefix_is_more_space_filling_than_random(enumeration):
    """The property that makes an interrupted sweep a valid sample.

    Also the empirical justification for choosing maximin over Latin hypercube: on a
    discrete grid with as few as three levels per gene, LHS stratification is vacuous
    and provides no advantage over uniform sampling.
    """
    space = enumeration.genotypes
    order = maximin_order(space, seed=0)
    for budget in (150, 300, 600):
        maximin = fill_distance(space, order[:budget])
        random = fill_distance(space, uniform_random_design(space, budget, seed=0))
        lhs, _ = latin_hypercube_design(space, budget, seed=0)
        assert maximin < random, f"maximin lost to random at {budget}"
        assert maximin < fill_distance(space, lhs), f"maximin lost to LHS at {budget}"


def test_lhs_records_how_stratification_was_resolved(enumeration):
    """A gene with fewer levels than the sample size cannot be properly stratified.

    The benchmark-artifact spec requires that limitation to live in the artifact rather
    than a footnote, so the design reports it per gene.
    """
    _, notes = latin_hypercube_design(enumeration.genotypes, 300, seed=0)
    assert "_realisation" in notes
    recycled = [gene for gene, note in notes.items() if "recycled" in note]
    assert recycled, "no gene reported recycled strata, which cannot be right at n=300"
