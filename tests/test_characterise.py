"""Claims about which architectures are good, tested rather than repeated.

Constructed data throughout, so each test knows the answer it should get. The interesting
part is the banding: without it, a comparison of deep against wide in this space is mostly a
comparison of small against large, because depth adds only block parameters while the
embedding dominates the total.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.characterise import (
    fronts_differ,
    gene_effects_on_quality,
    iso_parameter_depth_vs_width,
)

# Imported under another name: pytest collects any module-level callable whose name starts
# with test_, so importing it directly makes the production function look like a test case
# and it is then called with a fixture it does not take.
from nas4textgen.characterise import test_iso_parameter as run_protocol


def synthetic(n: int = 400, width_wins: bool = True, seed: int = 0):
    """Architectures with a shape effect planted independently of size.

    Built as an iso-parameter design rather than a random cloud: eight size levels, each
    holding architectures of nearly identical parameter count but varying depth-to-width
    ratio. That is what the analysis expects, and a random cloud does not provide it — the
    first version drew depth and width independently, and the residual size variation inside
    each band swamped the shape effect it was supposed to detect.

    Separating shape from size is the point. A term written in width alone would also be a
    term in size, since the embedding makes parameter count track width, and the test would
    then pass for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    per_level = max(9, n // 8)

    params, perplexity, depth, width = [], [], [], []
    for level in range(8):
        target = 5e6 * (2.0 ** level)
        base = 300.0 / (1.0 + level)          # size effect, constant within a level
        for _ in range(per_level):
            # log2(depth / width). Higher means deeper relative to its width.
            ratio = rng.uniform(-6, 0)
            w = rng.choice([64, 128, 256, 512, 1024])
            d = max(2.0, float(w) * (2.0 ** ratio))
            # Parameter count is pinned to the level, so the only thing varying is shape.
            params.append(target * rng.uniform(0.98, 1.02))
            depth.append(d)
            width.append(float(w))

            # Written without a double negative, which is how the first version planted the
            # opposite of what it claimed. Perplexity rising with the ratio means deeper is
            # worse, which is width winning.
            slope = 25.0 if width_wins else -25.0
            perplexity.append(base + slope * ratio + rng.normal(0, 0.5))

    return (
        np.array(params), np.array(perplexity), np.array(depth), np.array(width)
    )


class TestIsoParameter:
    def test_a_planted_width_advantage_is_recovered(self):
        params, perplexity, depth, width = synthetic(width_wins=True)
        result = iso_parameter_depth_vs_width(params, perplexity, depth, width, bands=6)
        assert result.bands_won_by_width >= result.bands - 1

    def test_a_planted_depth_advantage_is_recovered(self):
        """The test that shows the method is not simply always answering 'wide'."""
        params, perplexity, depth, width = synthetic(width_wins=False)
        result = iso_parameter_depth_vs_width(params, perplexity, depth, width, bands=6)
        assert result.bands_won_by_width <= 1

    def test_banding_is_what_separates_shape_from_size(self):
        """Unbanded, deep architectures in this space are also smaller on average, so a raw
        comparison would answer a question about size."""
        params, perplexity, depth, width = synthetic(width_wins=True)
        shape = depth / width

        deep_all = params[shape >= np.quantile(shape, 2 / 3)]
        wide_all = params[shape <= np.quantile(shape, 1 / 3)]
        assert deep_all.mean() < wide_all.mean(), (
            "the confound the banding exists for: deep architectures really are smaller here"
        )

    def test_the_extremes_must_leave_a_middle(self):
        params, perplexity, depth, width = synthetic()
        with pytest.raises(ValueError, match="leave a middle"):
            iso_parameter_depth_vs_width(params, perplexity, depth, width, extreme=0.5)

    def test_too_few_bands_is_refused(self):
        """Each band is one paired observation, and a signed-rank test over one pair says
        nothing."""
        tiny = np.array([1e6, 1.1e6, 1.2e6, 1.3e6])
        with pytest.raises(ValueError, match="paired observation"):
            iso_parameter_depth_vs_width(
                tiny, np.array([100.0, 99.0, 98.0, 97.0]),
                np.array([2.0, 4.0, 8.0, 16.0]), np.array([64.0, 64.0, 64.0, 64.0]),
                bands=10,
            )

    def test_the_verdict_goes_through_the_declared_protocol(self):
        params, perplexity, depth, width = synthetic(width_wins=True)
        result = iso_parameter_depth_vs_width(params, perplexity, depth, width, bands=8)
        out = run_protocol(result, rope=0.65, rope_source="pilot seed replicates")

        assert out["declaration"]["blocks"] == [f"band{i + 1}" for i in range(result.bands)]
        assert out["declaration"]["rope_source"]
        assert "wide" in out["bayesian"]["verdict"]

    def test_a_rope_is_required_for_the_verdict(self):
        """Practical equivalence has no meaning without a measured noise floor."""
        from nas4textgen.statistics import MalformedComparison

        params, perplexity, depth, width = synthetic()
        result = iso_parameter_depth_vs_width(params, perplexity, depth, width, bands=6)
        with pytest.raises(MalformedComparison):
            run_protocol(result, rope=0.65, rope_source="")


class TestFronts:
    def test_two_objectives_that_agree_give_one_front(self):
        """The degenerate case the project's premise denies. If cost and memory picked the
        same architectures, the second objective would be a relabelled axis."""
        params = np.array([1.0, 2.0, 3.0, 4.0])
        kv = params * 2          # perfectly correlated
        quality = 100 / params

        result = fronts_differ(params, kv, quality)
        assert result.jaccard == pytest.approx(1.0)

    def test_objectives_that_trade_off_give_different_fronts(self):
        params = np.array([1.0, 2.0, 3.0, 4.0])
        kv = np.array([4.0, 3.0, 2.0, 1.0])   # anti-correlated
        quality = np.array([10.0, 9.0, 8.0, 7.0])

        result = fronts_differ(params, kv, quality)
        assert result.jaccard < 1.0
        assert result.params_front != result.memory_front

    def test_the_joint_front_contains_at_least_the_union_of_neither(self):
        """A tri-objective front is not the union of the two bi-objective ones, and
        asserting the relationship keeps a later refactor honest."""
        rng = np.random.default_rng(0)
        params, kv, quality = (rng.random(80) for _ in range(3))
        result = fronts_differ(params, kv, quality)
        assert len(result.joint_front) >= len(result.params_front)
        assert len(result.joint_front) >= len(result.memory_front)


class TestGeneEffects:
    def test_a_gene_that_drives_quality_dominates_the_decomposition(self):
        rng = np.random.default_rng(0)
        genes = rng.integers(1, 5, size=(400, 3)).astype(float)
        quality = 50 * genes[:, 1] + rng.normal(0, 1, 400)

        effects = gene_effects_on_quality(genes, quality, ("a", "b", "c"))
        indices = effects["first_order_indices"]
        assert indices["b"] > 0.9
        assert indices["a"] < 0.05 and indices["c"] < 0.05

    def test_an_inert_gene_scores_near_zero(self):
        rng = np.random.default_rng(1)
        genes = rng.integers(1, 5, size=(400, 2)).astype(float)
        quality = 30 * genes[:, 0] + rng.normal(0, 1, 400)

        effects = gene_effects_on_quality(genes, quality, ("live", "inert"))
        assert effects["first_order_indices"]["inert"] < 0.02

    def test_the_shortfall_from_one_is_where_interactions_live(self):
        """A pure interaction has no first-order effect, and the estimator says so rather
        than attributing it to a main effect."""
        rng = np.random.default_rng(2)
        genes = rng.integers(0, 2, size=(600, 2)).astype(float)
        quality = 40 * (genes[:, 0] == genes[:, 1]) + rng.normal(0, 1, 600)

        effects = gene_effects_on_quality(genes, quality, ("a", "b"))
        assert effects["sum_of_first_order"] < 0.15, (
            "an XOR has almost no first-order structure, and the sum should show it"
        )

    def test_constant_quality_is_an_error_not_a_decomposition(self):
        genes = np.ones((20, 2))
        with pytest.raises(ValueError, match="nothing to decompose"):
            gene_effects_on_quality(genes, np.ones(20), ("a", "b"))

    def test_the_result_says_it_is_not_the_gate_measurement(self):
        """The gate measured each gene's effect on cost, which is closed-form. This is the
        effect on measured quality, and a gene can be nearly inert in one and dominant in
        the other."""
        rng = np.random.default_rng(3)
        genes = rng.integers(1, 5, size=(200, 2)).astype(float)
        quality = 10 * genes[:, 0] + rng.normal(0, 1, 200)

        effects = gene_effects_on_quality(genes, quality, ("a", "b"))
        assert "cost" in effects["distinct_from_the_gate"]
        assert "interactions" in effects["estimator"]


class TestGroupedQueryFrontier:
    """What sharing key-value heads costs, measured where it can be attributed."""

    def matched(self, elasticity: float = 0.0, seed: int = 0):
        """Groups identical in every gene but kv, with a planted cost per unit of cache."""
        from nas4textgen.characterise import grouped_query_frontier

        rng = np.random.default_rng(seed)
        heads, depth, width, ffn, kv, kv_bytes, ppl = [], [], [], [], [], [], []
        for h, n, e, f, base in [
            (4, 2, 256, 2, 300.0), (8, 6, 512, 3, 120.0),
            (12, 12, 768, 4, 70.0), (16, 24, 1024, 2, 55.0),
        ]:
            full_cache = float(n * e * h * 1024)
            for share in (1, 2, h):
                cache = full_cache * share / h
                removed = 1.0 - cache / full_cache
                heads.append(float(h))
                depth.append(float(n))
                width.append(float(e))
                ffn.append(float(f))
                kv.append(float(share))
                kv_bytes.append(cache)
                ppl.append(base * (1.0 + elasticity * removed) + rng.normal(0, 0.05))
        return grouped_query_frontier(
            *(np.array(v) for v in (heads, depth, width, ffn, kv, kv_bytes, ppl))
        )

    def test_a_planted_cost_is_recovered_as_an_elasticity(self):
        result = self.matched(elasticity=0.10)
        assert result.as_dict()["elasticity"]["median"] == pytest.approx(0.10, abs=0.02)

    def test_a_free_saving_reads_as_zero(self):
        result = self.matched(elasticity=0.0)
        assert abs(result.as_dict()["elasticity"]["median"]) < 0.02

    def test_matching_is_on_every_gene_but_kv(self):
        """Leaving the feed-forward ratio free was enough to bury the effect entirely: a
        group like h=2, n=2, e=64 then mixed three ffn values whose perplexities differ by
        more than the kv effect being measured."""
        import inspect

        from nas4textgen.characterise import grouped_query_frontier

        source = inspect.getsource(grouped_query_frontier)
        assert "int(ffn[index])" in source, "ffn must be part of the group signature"
        assert inspect.getdoc(grouped_query_frontier).count("ffn") >= 1

    def test_the_elasticity_is_reported_as_a_distribution(self):
        """A pooled slope in perplexity-per-megabyte is dominated by the largest
        architectures, because least squares weights by the square of the saving and the
        same six perplexity points are one per cent of 509 and ten per cent of 60."""
        payload = self.matched(elasticity=0.05).as_dict()
        elasticity = payload["elasticity"]
        assert elasticity["quartiles"] is not None
        assert elasticity["share_costing_quality"] is not None
        assert "should not be read as the answer" in elasticity["definition"]

    def test_too_few_matched_pairs_is_refused(self):
        from nas4textgen.characterise import grouped_query_frontier

        one = np.array([4.0, 4.0])
        with pytest.raises(ValueError, match="matched pair"):
            grouped_query_frontier(
                one, one, one * 64, one, np.array([1.0, 4.0]),
                np.array([1e6, 4e6]), np.array([100.0, 99.0]),
            )


class TestDesignRulesMustBeatTheirBaseRate:
    """The trap every mined design rule walks into.

    "92 per cent of Pareto-optimal architectures have fewer than 12 layers" sounds like a
    finding until you notice 89 per cent of the whole space does too. A rule is only a rule
    where the front's rate under it exceeds the front's rate overall.
    """

    @staticmethod
    def _rule(matched, matched_on_front, total=1000, total_on_front=200):
        from nas4textgen.characterise import DesignRule

        return DesignRule(gene="n", predicate="n <= 8", matched=matched,
                          matched_on_front=matched_on_front, total=total,
                          total_on_front=total_on_front)

    def test_a_rule_that_only_restates_the_base_rate_is_refused(self):
        """900 of 1000 architectures match and 180 of them are on the front — exactly the
        20 per cent the whole space shows. The rule has told you nothing."""
        rule = self._rule(matched=900, matched_on_front=180)
        assert rule.precision == pytest.approx(0.2)
        assert rule.lift == pytest.approx(1.0)
        assert not rule.holds
        assert "describes rather than predicts" in rule.verdict

    def test_a_rule_that_doubles_the_rate_holds(self):
        rule = self._rule(matched=200, matched_on_front=80)
        assert rule.lift == pytest.approx(2.0)
        assert rule.holds
        assert "holds" in rule.verdict

    def test_a_narrow_rule_with_huge_lift_is_still_refused(self):
        """Four architectures can have enormous lift and describe nothing about the front."""
        rule = self._rule(matched=4, matched_on_front=4)
        assert rule.lift == pytest.approx(5.0)
        assert rule.coverage == pytest.approx(0.02)
        assert not rule.holds
        assert "too narrow" in rule.verdict

    def test_only_single_gene_thresholds_are_mined(self):
        """Conjunctions over five genes give hundreds of candidates over 1,099
        architectures, and the best of hundreds looks impressive by construction."""
        import numpy as np

        from nas4textgen.characterise import design_rules
        from nas4textgen.space.genotype import GENES

        genes = np.array([[2, 32, 2, 2, 1], [16, 128, 24, 4, 16]] * 40)
        on_front = np.array([True, False] * 40)
        rules = design_rules(genes, on_front, GENES)
        assert all(r.predicate.count(" ") == 2 for r in rules), "one gene, one threshold"
        assert {r.gene for r in rules} <= set(GENES)

    def test_the_rules_hold_on_the_real_front(self):
        """Regression on the finding itself: sharing key-value heads survives removing the
        objective that would make it true by construction."""
        import json
        from pathlib import Path

        path = (Path(__file__).resolve().parents[1]
                / "artifacts" / "characterisation" / "design_rules.json")
        if not path.exists():
            pytest.skip("run: python -m nas4textgen.cli characterise")
        payload = json.loads(path.read_text(encoding="utf-8"))

        without_cache = {r["rule"]: r for r in payload["parameters_only"]["holding"]}
        assert "kv <= 1" in without_cache, (
            "the key-value rule must survive the parameters-only front, where cache size "
            "is not an objective and the rule cannot be tautological"
        )
        assert without_cache["kv <= 1"]["lift"] > 1.5
