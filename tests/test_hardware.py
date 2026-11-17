"""The deployment measurement protocol, and the numbers it refuses to publish.

Latency and energy are measured with an instrument whose own variation is often the size of
the effect being looked for, and every knob — warm-up, sampling rate, absolute versus delta —
moves the answer. Choosing them after seeing results is choosing the result, so the protocol
is frozen and its own idle spread becomes the threshold.
"""

from __future__ import annotations

import pytest

from nas4textgen.hardware import (
    DeploymentMeasurement,
    NotReproducible,
    Protocol,
    ProtocolViolation,
    baseline_from,
    require_reproducible,
)


def protocol(**overrides) -> Protocol:
    base = {"device": "Tesla T4", "power_mode": "default 70W"}
    base.update(overrides)
    return Protocol(**base)


def baseline(sigma: float = 0.5, mean: float = 30.0, n: int = 600):
    import numpy as np

    rng = np.random.default_rng(0)
    return baseline_from(list(rng.normal(mean, sigma, n)), protocol())


class TestProtocol:
    def test_an_unstated_power_mode_is_refused(self):
        """The same model draws different power in each mode, so a figure without it is not
        comparable to anything."""
        with pytest.raises(ProtocolViolation, match="power mode is unstated"):
            Protocol(device="T4", power_mode="")

    def test_a_short_warm_up_is_refused(self):
        """It would measure the device clocking up rather than the model running."""
        with pytest.raises(ProtocolViolation, match="clocking up"):
            protocol(warm_up_seconds=2)

    def test_too_few_repetitions_is_refused(self):
        """The ROPE is derived from their spread, so two cannot support one."""
        with pytest.raises(ProtocolViolation, match="cannot support a spread"):
            protocol(repetitions=2)

    def test_the_default_reports_deltas_and_says_why(self):
        """Absolute power includes whatever else the device is doing; a delta attributes
        only the increment to the model, which is what a deployment question asks."""
        payload = protocol().as_dict()
        assert payload["reported_as"].startswith("delta")
        assert "how much more does this cost" in payload["why_delta"]

    def test_context_and_batch_are_part_of_the_protocol(self):
        """Latency per token depends on both, so a figure without them is not a figure."""
        payload = protocol().as_dict()
        assert payload["context_tokens"] > 0
        assert payload["batch_size"] > 0

    def test_the_device_state_before_the_run_is_recorded(self):
        """A device that has just finished a heavy job is not in the same state as one that
        has been idle."""
        assert "idle" in protocol().as_dict()["device_state_before"]


class TestBaseline:
    def test_the_rope_comes_from_the_idle_spread(self):
        """Derived rather than chosen. A difference below it is the instrument, not the
        architecture."""
        base = baseline(sigma=0.5)
        assert base.rope_watts == pytest.approx(0.5 * base.sigma)
        assert 0.1 < base.rope_watts < 0.5

    def test_too_few_idle_samples_is_refused(self):
        with pytest.raises(ProtocolViolation, match="cannot establish a noise floor"):
            baseline_from([30.0] * 10, protocol())

    def test_a_constant_reading_is_refused(self):
        """Either the instrument is not reading or it is quantised too coarsely to derive a
        threshold from."""
        with pytest.raises(ProtocolViolation, match="not reading"):
            baseline_from([30.0] * 600, protocol())


class TestMeasurement:
    def measurement(self, watts: list[float], rate: float = 50.0):
        return DeploymentMeasurement(
            architecture="h12 d64 n6 kv12",
            protocol=protocol(),
            baseline=baseline(sigma=0.5),
            latency_ms_per_token=[20.0] * len(watts),
            tokens_per_second=[rate] * len(watts),
            peak_memory_bytes=[1_500_000_000] * len(watts),
            watts_under_load=watts,
        )

    def test_energy_is_the_increment_over_idle(self):
        """Absolute power over the token rate would charge the model for the device's
        existence."""
        m = self.measurement([40.0] * 5, rate=50.0)
        expected = (40.0 - m.baseline.mean) / 50.0
        assert m.energy_per_token_joules[0] == pytest.approx(expected)
        assert all(j > 0 for j in m.energy_per_token_joules)

    def test_repetitions_that_agree_are_reproducible(self):
        m = self.measurement([40.0, 40.05, 39.98, 40.02, 40.01])
        ok, why = m.reproducible()
        assert ok and "within a ROPE" in why

    def test_repetitions_that_disagree_are_refused(self):
        """A figure that does not repeat is a property of the run rather than of the
        architecture."""
        m = self.measurement([40.0, 46.0, 38.0, 44.0, 39.0])
        assert not m.reproducible()[0]
        with pytest.raises(NotReproducible, match="not repeatable"):
            require_reproducible(m)

    def test_missing_repetitions_are_refused_rather_than_averaged(self):
        m = self.measurement([40.0, 40.1])
        ok, why = m.reproducible()
        assert not ok
        assert "2 of 5" in why

    def test_latency_and_memory_are_reported_but_not_gated(self):
        """A latency that varies is a fact about the device under load, and gating it on a
        threshold derived from a power instrument would borrow a tolerance from the wrong
        measurement."""
        m = DeploymentMeasurement(
            architecture="x", protocol=protocol(), baseline=baseline(),
            latency_ms_per_token=[20.0, 35.0, 21.0, 22.0, 20.5],
            tokens_per_second=[50.0] * 5,
            peak_memory_bytes=[1_000_000_000] * 5,
            watts_under_load=[40.0, 40.02, 39.99, 40.01, 40.0],
        )
        assert m.reproducible()[0], "power repeats, so the measurement stands"
        spread = m.as_dict()["latency_ms_per_token"]
        assert spread["max"] > spread["median"], "and the latency spread is still visible"

    def test_the_record_carries_the_protocol_it_was_taken_under(self):
        """Two measurements are comparable only if the protocol was the same, so it travels
        with the number rather than being remembered separately."""
        payload = self.measurement([40.0] * 5).as_dict()
        assert payload["protocol"]["power_mode"]
        assert payload["baseline"]["rope_watts"] > 0
        assert "reproducibility_note" in payload
