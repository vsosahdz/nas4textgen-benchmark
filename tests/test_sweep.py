"""Waves, and the property that makes stopping early cheap rather than wasteful."""

from __future__ import annotations

import pytest

from nas4textgen.azure.sweep import (
    WAVE_SIZES,
    ExecutionOrderMissing,
    OrderNotNested,
    load_order,
    waves,
)
from nas4textgen.space.enumerate import enumerate_with_ceiling


@pytest.fixture(scope="module")
def space():
    return enumerate_with_ceiling().genotypes


class TestWaves:
    def test_each_wave_contains_the_one_before(self, space):
        """The property the whole design rests on. A hundred and fifty architectures spread
        over the space support a surrogate study and an interim front; the first hundred and
        fifty in enumeration order would all be small and support neither."""
        plan = waves(list(range(len(space))), space)
        for earlier, later in zip(plan, plan[1:], strict=False):
            assert set(g.as_tuple() for g in earlier.genotypes) <= set(
                g.as_tuple() for g in later.genotypes
            )

    def test_only_the_new_architectures_are_submitted(self, space):
        """Otherwise wave two pays again for wave one."""
        plan = waves(list(range(len(space))), space)
        assert len(plan[0].new) == WAVE_SIZES[0]
        for earlier, later in zip(plan, plan[1:], strict=False):
            assert len(later.new) == later.size - earlier.size

    def test_the_waves_cover_the_whole_space(self, space):
        plan = waves(list(range(len(space))), space)
        assert plan[-1].size == len(space)
        submitted = {g.as_tuple() for wave in plan for g in wave.new}
        assert submitted == {g.as_tuple() for g in space}

    def test_no_architecture_is_submitted_twice(self, space):
        plan = waves(list(range(len(space))), space)
        submitted = [g.as_tuple() for wave in plan for g in wave.new]
        assert len(submitted) == len(set(submitted)) == len(space)

    def test_an_order_that_does_not_nest_is_refused(self, space):
        """Nesting is checked rather than assumed, because the sampling code that produces
        the order is not the code that consumes it."""
        broken = list(range(len(space)))
        # Move an element from inside wave 1 to beyond it.
        broken[10], broken[900] = broken[900], broken[10]
        # Still nests: swapping within the prefix is fine. Break it properly by rotating
        # the prefix out of the second wave's range is impossible with prefixes, so assert
        # the guard fires on a hand-built violation instead.
        with pytest.raises(OrderNotNested):
            _force_non_nesting(space)

    def test_a_mismatched_order_is_refused(self, space):
        """An order from a different revision of the search space would silently measure
        the wrong architectures."""
        with pytest.raises(ExecutionOrderMissing, match="different revision"):
            waves(list(range(len(space) - 5)), space)


def _force_non_nesting(space):
    """Call waves() with sizes that cannot nest, to prove the check is live."""
    return waves(list(range(len(space))), space, sizes=(300, 150))


class TestRecordedOrder:
    def test_the_order_is_read_rather_than_recomputed(self):
        """Recomputing at submission time would let a change in the sampling code reorder a
        campaign already half done, and the waves would stop nesting with what has run."""
        import inspect

        from nas4textgen.azure import sweep

        source = inspect.getsource(sweep.load_order)
        assert "maximin_order" in source
        for recompute in ("maximin_order(", "latin_hypercube", "default_rng"):
            assert recompute not in source

    def test_a_missing_order_says_what_to_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NAS4TG_ARTIFACTS", str(tmp_path))
        with pytest.raises(ExecutionOrderMissing, match="designs"):
            load_order()

    def test_the_recorded_order_matches_the_space(self, space):
        order = load_order()
        assert len(order) == len(space)
        assert sorted(order) == list(range(len(space)))


class TestCost:
    def test_the_waves_sum_to_the_whole_sweep(self, space):
        from nas4textgen.azure.sweep import cost
        from nas4textgen.azure.throughput import load

        model = load()
        if model is None:
            pytest.skip("no throughput model recorded")

        plan = waves(load_order(), space)
        total = sum(cost(w, model, 100_000_000, 5).node_hours for w in plan)
        whole = sum(model.seconds_for(g, 100_000_000) for g in space) / 3600
        assert total == pytest.approx(whole, rel=1e-6)

    def test_the_price_is_the_measured_one(self):
        from nas4textgen.azure.sweep import LOW_PRIORITY_USD_PER_HOUR

        # Azure retail prices API, Standard_NC4as_T4_v3 Low Priority, southcentralus.
        assert pytest.approx(0.1260) == LOW_PRIORITY_USD_PER_HOUR


def test_the_first_wave_is_small_enough_to_be_a_decision(space):
    """Submitting 1,099 jobs at once removes the decision point, and the failures it
    removes it for are the expensive kind — an architecture class that reliably runs out of
    memory, a limit too tight at the large end. Every one of those was found at canary
    scale already."""
    assert WAVE_SIZES[0] < len(space) / 5
    plan = waves(load_order(), space)
    assert len(plan) >= 4, "enough waves that a problem is caught before most of the cost"
