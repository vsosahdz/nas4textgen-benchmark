"""The pilot's design, which is what makes its twenty runs worth more than twenty random ones.

Two numbers the campaign depends on are currently guesses: how long an unmeasured
architecture takes, and how much a perplexity moves when nothing but the seed changes. The
first makes every timeout fall back to the slowest rate ever seen; the second blocks the
entire statistical layer, because the spec requires a ROPE derived from measured noise.
"""

from __future__ import annotations

import numpy as np

from nas4textgen.azure.pilot import (
    CALIBRATION_SIZE,
    VARIANCE_SEEDS,
    calibration_set,
    plan,
    summary,
    variance_architecture,
)
from nas4textgen.space.cost import forward_flops_per_token
from nas4textgen.space.enumerate import enumerate_with_ceiling


class TestCalibrationSet:
    def test_it_spreads_over_compute_rather_than_over_the_space(self):
        """Sampling the space samples its density, and this space is dense where
        architectures are cheap — the fit would then be well-determined exactly where it is
        already known and unconstrained at the end that sets the timeouts.
        """
        chosen = calibration_set()
        flops = np.array(sorted(forward_flops_per_token(g) for g in chosen), dtype=float)

        space_flops = np.array(
            sorted(forward_flops_per_token(g) for g in enumerate_with_ceiling().genotypes),
            dtype=float,
        )
        # The chosen set reaches both ends of the real range.
        assert flops[0] <= space_flops[0] * 1.5
        assert flops[-1] >= space_flops[-1] * 0.9

        # Roughly even in log space: no gap more than a few times the median gap.
        gaps = np.diff(np.log10(flops))
        assert gaps.max() < 6 * np.median(gaps), "log spacing left a hole in the range"

    def test_linear_spacing_would_have_crowded_the_top_decade(self):
        """Stated as a comparison because it is the reason for the log, and a reader should
        be able to check it rather than take it."""
        space = sorted(enumerate_with_ceiling().genotypes, key=forward_flops_per_token)
        flops = np.array([forward_flops_per_token(g) for g in space], dtype=float)

        linear = np.linspace(flops[0], flops[-1], CALIBRATION_SIZE)
        top_decade = flops[-1] / 10
        assert (linear > top_decade).sum() >= CALIBRATION_SIZE * 0.8, (
            "linear targets really would sit almost entirely in the top decade"
        )

        chosen = np.array([forward_flops_per_token(g) for g in calibration_set()])
        assert (chosen > top_decade).sum() <= CALIBRATION_SIZE * 0.3

    def test_every_architecture_is_distinct(self):
        """A cluster of identical-cost architectures must not collapse several targets onto
        one point and quietly shrink the sample."""
        chosen = calibration_set()
        assert len({g.as_tuple() for g in chosen}) == len(chosen) == CALIBRATION_SIZE

    def test_every_architecture_is_in_the_space(self):
        space = {g.as_tuple() for g in enumerate_with_ceiling().genotypes}
        assert all(g.as_tuple() in space for g in calibration_set())

    def test_the_set_is_reproducible(self):
        assert [g.as_tuple() for g in calibration_set()] == [
            g.as_tuple() for g in calibration_set()
        ]


class TestVarianceStudy:
    def test_the_repeated_architecture_sits_near_the_median_of_compute(self):
        """A ROPE from the smallest architecture would describe a model that barely trains;
        one from the largest would cost five times as much. The median is where most of the
        space's runs sit, so its noise is what most comparisons are made against."""
        space = sorted(enumerate_with_ceiling().genotypes, key=forward_flops_per_token)
        flops = [forward_flops_per_token(g) for g in space]
        chosen = forward_flops_per_token(variance_architecture())

        rank = flops.index(chosen) / len(flops)
        assert 0.35 < rank < 0.65, f"sits at the {rank:.0%} percentile of compute"

    def test_enough_seeds_for_the_rope_helper_to_accept_them(self):
        """Below its minimum the standard deviation is itself so uncertain that a ROPE
        built on it is arbitrary in a different way than simply picking one."""
        from nas4textgen.statistics.rope import MIN_REPLICATES

        assert VARIANCE_SEEDS >= MIN_REPLICATES

    def test_the_seeds_differ_and_nothing_else_does(self):
        variance = [m for m in plan() if m.purpose == "variance"]
        assert len({m.genotype.as_tuple() for m in variance}) == 1
        assert sorted(m.seed for m in variance) == list(range(VARIANCE_SEEDS))


class TestPlan:
    def test_the_pilot_is_a_small_fraction_of_the_sweep(self):
        """A pilot that cost a tenth of the sweep would not be worth running before it."""
        space = len(enumerate_with_ceiling().genotypes)
        assert len(plan()) / space < 0.05

    def test_each_half_states_what_it_unblocks(self):
        report = summary(plan())
        assert "declines to project" in report["calibration"]["unblocks"]
        assert "ROPE" in report["variance"]["unblocks"]
        assert report["calibration"]["decades_covered"] > 2

    def test_every_run_says_what_it_is_for(self):
        for member in plan():
            assert member.purpose in ("calibration", "variance")
            assert len(member.why) > 30

    def test_the_size_is_configurable_without_breaking_the_spread(self):
        for size in (8, 12, 30):
            chosen = calibration_set(size)
            assert len(chosen) == size
            assert len({g.as_tuple() for g in chosen}) == size


def test_a_calibration_of_this_size_makes_the_fit_checkable():
    """The calibration must clear the fit's minimum on its own.

    The minimum already carries the four-samples-per-parameter rule, so the requirement
    here is simply that twenty points is enough — and it is, with room for a run to fail
    without taking the fit down with it. One of the twenty did fail.
    """
    from nas4textgen.azure.throughput import MIN_SAMPLES, MODEL_PARAMETERS

    assert MIN_SAMPLES == 4 * MODEL_PARAMETERS
    assert CALIBRATION_SIZE >= MIN_SAMPLES
    assert CALIBRATION_SIZE >= MIN_SAMPLES + 4, "leave room for a lost run"
