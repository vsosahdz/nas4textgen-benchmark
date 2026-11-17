"""Projecting run time from what the canary measured, rather than from a formula.

Every number here comes from the three canary jobs on a T4 with float16. The point of the
model is that it replaces a guess with a fit whose residuals are reported.
"""

from __future__ import annotations

import pytest

from nas4textgen.azure import throughput as T
from nas4textgen.space.genotype import Genotype

MEASURED = [
    (Genotype(h=2, d_head=32, n=2, ffn=2, kv=1), 45_713.0, 32),
    (Genotype(h=12, d_head=32, n=16, ffn=2, kv=1), 22_609.5, 8),
    (Genotype(h=6, d_head=128, n=24, ffn=2, kv=1), 8_171.3, 16),
]


def samples(device="Tesla T4"):
    return [
        T.ThroughputSample(g, rate, device, "float16", batch, f"job{i}")
        for i, (g, rate, batch) in enumerate(MEASURED)
    ]


def test_too_few_points_are_refused_for_want_of_degrees_of_freedom():
    """A fit that nearly interpolates its own inputs has a meaningless residual.

    The canary's four measurements pass the under-prediction test at 1.07x, which sounds
    excellent and says only that a three-parameter model can reproduce four points. The
    pilot's twenty-seven give the same fit 1.48x, and that number is about the model rather
    than about the arithmetic.
    """
    assert T.MIN_SAMPLES >= 4 * T.MODEL_PARAMETERS
    with pytest.raises(T.NotEnoughSamples):
        T.fit(samples()[:2])
    with pytest.raises(T.NotEnoughSamples):
        T.fit((samples() * 2)[:4])


def _wide_enough():
    return samples() * 4


def test_samples_from_different_devices_are_refused():
    """A throughput model describes one device; mixing them projects a machine that does
    not exist."""
    mixed = _wide_enough()
    mixed[0] = T.ThroughputSample(MEASURED[0][0], 45_713.0, "A100", "bfloat16", 32, "x")
    with pytest.raises(T.NotEnoughSamples, match="one device"):
        T.fit(mixed)


def test_the_fit_reproduces_its_own_measurements_closely():
    model = T.fit(samples() * 4)
    for error in model.relative_error().values():
        assert error < 0.15, f"the affine shape does not describe the data: {error:.1%}"


def test_the_model_is_affine_rather_than_proportional_to_flops():
    """Small architectures are not compute-bound. The smallest measured reached 0.009
    TFLOP/s and the heaviest 2.78 on the same card, so a model proportional to FLOPs would
    predict the smallest finishing in seconds and hand it a limit it could not meet.
    """
    model = T.fit(_wide_enough())
    assert model.intercept > 0, "a fixed per-token overhead must survive the fit"

    smallest, heaviest = MEASURED[0][0], MEASURED[2][0]
    flops_ratio = (
        T.forward_flops_per_token(heaviest) / T.forward_flops_per_token(smallest)
    )
    time_ratio = model.seconds_per_token(heaviest) / model.seconds_per_token(smallest)

    assert flops_ratio > 1000, "the compute span really is three orders of magnitude"
    assert time_ratio < 20, "but the time span is not, because overhead dominates below"


def test_projection_never_returns_a_rate_faster_than_anything_observed():
    """Extrapolating below the smallest sample can send an affine fit negative."""
    model = T.fit(_wide_enough())
    tiny = Genotype(h=2, d_head=32, n=2, ffn=2, kv=1)
    fastest = max(s.tokens_per_second for s in model.samples)
    assert model.seconds_per_token(tiny) >= 1.0 / fastest


def test_the_record_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("NAS4TG_ARTIFACTS", str(tmp_path))
    assert T.load() is None, "no model before the canary has run"

    original = T.fit(_wide_enough())
    T.save(original)
    restored = T.load()

    assert restored is not None
    assert restored.slope == pytest.approx(original.slope)
    assert restored.intercept == pytest.approx(original.intercept)
    assert len(restored.samples) == len(original.samples)
    assert restored.device == "Tesla T4"


def test_each_job_gets_a_limit_at_its_own_scale(tmp_path, monkeypatch):
    """One limit per wave has to cover the slowest architecture, so the fastest carries
    six times what it needs — and a limit that loose does not do what a limit is for."""
    monkeypatch.setenv("NAS4TG_ARTIFACTS", str(tmp_path))
    from nas4textgen.azure.submit import JobSpec, timeout_for

    model = T.fit(_wide_enough())
    T.save(model)

    def limit(genotype):
        return timeout_for(
            JobSpec(genotype=genotype, corpus="wikitext103", seed=0,
                    token_budget=100_000_000),
            model=model,
        )

    small, heavy = limit(MEASURED[0][0]), limit(MEASURED[2][0])
    assert small < heavy
    assert heavy / small > 3, "the limits must actually track the architectures"
    # Both stay inside a working day, so a hung job is caught the same day it hangs.
    assert heavy < 8 * 3600


class TestProjectability:
    """A fit that cannot predict must say so rather than being used as if it could."""

    def wide(self):
        """The canary's four float16 measurements. They span 1,728x in compute."""
        return [
            T.ThroughputSample(Genotype(h=2, d_head=32, n=2, ffn=2, kv=1),
                               86_540.3, "Tesla T4", "float16", 32, "a"),
            T.ThroughputSample(Genotype(h=12, d_head=32, n=16, ffn=2, kv=1),
                               24_338.9, "Tesla T4", "float16", 8, "b"),
            T.ThroughputSample(Genotype(h=12, d_head=128, n=4, ffn=2, kv=6),
                               11_577.8, "Tesla T4", "float16", 8, "c"),
            T.ThroughputSample(Genotype(h=6, d_head=128, n=24, ffn=2, kv=1),
                               9_445.3, "Tesla T4", "float16", 16, "d"),
        ]

    def test_four_points_are_refused_before_their_residual_is_even_consulted(self):
        """The canary could not have produced a projectable model however good it looked."""
        with pytest.raises(T.NotEnoughSamples):
            T.fit(self.wide())

    def test_an_unprojectable_model_falls_back_to_the_slowest_measured_rate(self):
        """A bound rather than a claim. Loose for small architectures, which finish early
        and never reach the limit."""
        model = T.ThroughputModel(
            slope=1e-15, layer_cost=0.0, intercept=1e-9,
            samples=tuple(samples() * 5), device="Tesla T4",
        )
        assert not model.is_projectable
        slowest = min(s.tokens_per_second for s in model.samples)
        unseen = Genotype(h=4, d_head=64, n=8, ffn=3, kv=2)
        assert model.seconds_per_token(unseen) == pytest.approx(1.0 / slowest)

    def test_a_measured_architecture_uses_its_measurement_not_the_fit(self):
        """A measurement beats a fit, including a fit good enough to be projectable."""
        model = T.fit(self.wide() * 3)
        for sample in model.samples:
            projected = model.seconds_for(sample.genotype, 100_000_000)
            assert projected == pytest.approx(100_000_000 / sample.tokens_per_second)

    def test_the_recorded_pilot_model_is_projectable(self):
        """The threshold is reachable: the twenty-seven measured points clear it."""
        recorded = T.load()
        if recorded is None or len(recorded.samples) < T.MIN_SAMPLES:
            pytest.skip("no pilot model recorded in this checkout")
        assert recorded.is_projectable
        assert recorded.worst_underprediction < T.MAX_UNDERPREDICTION

    def test_scoring_the_fit_does_not_consult_the_fallback(self):
        """is_projectable asked relative_error which asked seconds_per_token which asked
        is_projectable. The residual is a property of the fit, so it is computed from the
        fit alone."""
        model = T.fit(self.wide() * 3)
        assert model.relative_error()  # would recurse forever if the paths were shared
        assert model._fitted_seconds_per_token(model.samples[0].genotype) > 0


class TestJudgedOnTheRightLoss:
    """The model sets wall-clock limits, so only under-prediction is dangerous.

    Scoring it with a symmetric residual reported 47% on the pilot's 27 measurements, which
    reads as failure. The same fit bounds every one of them at a factor of 1.48 — and a
    limit derived from too large a number costs nothing unless a job hangs, while one from
    too small a number kills a healthy run at the expensive end and wastes everything it
    computed.
    """

    def test_the_threshold_comes_from_the_head_room_not_from_the_data(self):
        """Setting it to the worst observed value would make the test circular: any sample
        passes by definition, and a worse sample raises the bar to accommodate itself."""
        from nas4textgen.azure.submit import TIMEOUT_SAFETY_FACTOR

        assert pytest.approx(
            TIMEOUT_SAFETY_FACTOR / T.REQUIRED_TIMEOUT_MARGIN
        ) == T.MAX_UNDERPREDICTION
        assert T.MAX_UNDERPREDICTION < TIMEOUT_SAFETY_FACTOR, "no margin left"

    def test_a_model_that_underpredicts_badly_is_not_projectable(self):
        """Built by hand so the failure is certain rather than hoped for."""
        model = T.ThroughputModel(
            slope=1e-15, layer_cost=0.0, intercept=1e-9,
            samples=samples(), device="Tesla T4",
        )
        assert model.worst_underprediction > T.MAX_UNDERPREDICTION
        assert not model.is_projectable

    def test_amortising_depth_matters_because_the_micro_batch_varies(self):
        """The pilot measured the same throughput question at micro-batches 4, 8, 16 and 32.
        Ignoring that spends the residual explaining a confound: adding the term took the
        median error from 26% to 9%.
        """
        model = T.fit(_wide_enough())
        genotype = Genotype(h=6, d_head=64, n=24, ffn=3, kv=1)

        small = model._fitted_seconds_per_token(genotype, micro_batch=4)
        large = model._fitted_seconds_per_token(genotype, micro_batch=32)
        assert small > large, "less amortisation must mean more seconds per token"

    def test_projection_assumes_the_less_amortised_case(self):
        """Erring towards more predicted time, since the probe may settle anywhere."""
        model = T.fit(_wide_enough())
        assert model.projection_micro_batch <= min(s.micro_batch for s in samples())
