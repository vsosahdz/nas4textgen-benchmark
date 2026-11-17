"""How a deployment measurement is taken, fixed before anything is measured.

The spec requires the protocol declared in advance, and the reason is specific to this kind
of number. Latency and energy are measured with an instrument whose own variation is often
the same size as the effect being looked for, and every knob — how long the device warms up,
how fast it is sampled, whether the figure is absolute or a delta against idle — moves the
answer. Choosing those after seeing the results is choosing the result.

So the protocol is a frozen object, its own idle variance becomes the ROPE, and a
measurement that has not passed the reproducibility check is not reported.

**Absolute or delta is the decision that matters most.** Absolute power includes whatever
else the device is doing; a delta against a measured idle baseline attributes only the
increment to the model. The second is what a deployment question asks — how much more does
this cost than not running it — and it is only meaningful if the baseline was measured under
the same conditions, which is why the baseline is part of the protocol rather than a
constant someone remembers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

#: Seconds of steady load before any sample counts. A cold device clocks up over the first
#: seconds, so a measurement that starts immediately measures the ramp rather than the
#: model.
WARM_UP_SECONDS = 30

#: Samples per second from the power instrument. Fast enough to see the shape of a decode
#: step, slow enough that the sampling itself does not perturb what it measures.
SAMPLE_HZ = 10

#: Seconds of measurement after warm-up, per repetition.
MEASURE_SECONDS = 60

#: Independent repetitions. Five, because the ROPE derives from their spread and fewer
#: leaves a standard deviation too uncertain to build a threshold on — the same reason the
#: quality ROPE needs five seeds.
REPETITIONS = 5

#: Multiple of the measured idle standard deviation that counts as a real difference.
ROPE_SIGMA_MULTIPLE = 0.5


class ProtocolViolation(RuntimeError):
    """Raised when a measurement was taken under conditions the protocol does not allow."""


class NotReproducible(RuntimeError):
    """Raised when repetitions of one measurement disagree by more than the ROPE."""


@dataclass(frozen=True)
class Protocol:
    """Everything that has to be the same between two comparable measurements."""

    device: str
    power_mode: str
    warm_up_seconds: int = WARM_UP_SECONDS
    sample_hz: int = SAMPLE_HZ
    measure_seconds: int = MEASURE_SECONDS
    repetitions: int = REPETITIONS
    #: Whether reported figures are absolute or increments over a measured idle baseline.
    absolute: bool = False
    #: What the device was doing before the run. Recorded because a device that has just
    #: finished a heavy job is not in the same state as one that has been idle.
    device_state_before: str = "idle for at least 60 seconds, no other load"
    #: Sustained decode context and batch, since latency per token depends on both and a
    #: figure without them is not a figure.
    context_tokens: int = 512
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.warm_up_seconds < 5:
            raise ProtocolViolation(
                "a warm-up under five seconds measures the device clocking up rather than "
                "the model running"
            )
        if self.repetitions < 3:
            raise ProtocolViolation(
                f"{self.repetitions} repetitions cannot support a spread, and the ROPE is "
                "derived from that spread"
            )
        if not self.power_mode:
            raise ProtocolViolation(
                "the power mode is unstated. On a device with selectable modes the same "
                "model draws different power in each, so a figure without it is not "
                "comparable to anything."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "power_mode": self.power_mode,
            "warm_up_seconds": self.warm_up_seconds,
            "sample_hz": self.sample_hz,
            "measure_seconds": self.measure_seconds,
            "repetitions": self.repetitions,
            "reported_as": "absolute" if self.absolute else "delta against measured idle",
            "device_state_before": self.device_state_before,
            "context_tokens": self.context_tokens,
            "batch_size": self.batch_size,
            "why_delta": (
                "Absolute power includes whatever else the device is doing. A delta against "
                "a baseline measured under these same conditions attributes only the "
                "increment to the model, which is what a deployment question asks: how much "
                "more does this cost than not running it."
            ),
        }


@dataclass
class IdleBaseline:
    """What the device draws doing nothing, and how much that number itself wanders."""

    protocol: Protocol
    samples_watts: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples_watts)

    @property
    def sigma(self) -> float:
        return statistics.stdev(self.samples_watts)

    @property
    def rope_watts(self) -> float:
        """The instrument's own noise floor, in watts.

        Derived rather than chosen. A difference smaller than this is not a small
        difference; it is the instrument, and reporting it as an architectural result would
        be the same class of error as the prior work's metric that moved for reasons
        unrelated to quality.
        """
        return ROPE_SIGMA_MULTIPLE * self.sigma

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": len(self.samples_watts),
            "mean_watts": round(self.mean, 4),
            "sigma_watts": round(self.sigma, 4),
            "rope_watts": round(self.rope_watts, 4),
            "sigma_multiple": ROPE_SIGMA_MULTIPLE,
            "why": (
                "The idle spread is what the instrument does when nothing changes, so it is "
                "the floor below which no claim about an architecture can be made."
            ),
        }


def baseline_from(samples: list[float], protocol: Protocol) -> IdleBaseline:
    """Build the baseline, refusing a sample too small to have a spread."""
    expected = protocol.measure_seconds * protocol.sample_hz
    if len(samples) < 30:
        raise ProtocolViolation(
            f"{len(samples)} idle samples cannot establish a noise floor; the protocol "
            f"specifies {expected} at {protocol.sample_hz} Hz for "
            f"{protocol.measure_seconds}s"
        )
    if len(set(samples)) == 1:
        raise ProtocolViolation(
            "every idle sample is identical, so either the instrument is not reading or it "
            "is quantised too coarsely to derive a threshold from"
        )
    return IdleBaseline(protocol=protocol, samples_watts=list(samples))


@dataclass
class DeploymentMeasurement:
    """One architecture's deployment cost, over the protocol's repetitions."""

    architecture: str
    protocol: Protocol
    baseline: IdleBaseline
    #: One entry per repetition.
    latency_ms_per_token: list[float] = field(default_factory=list)
    tokens_per_second: list[float] = field(default_factory=list)
    peak_memory_bytes: list[int] = field(default_factory=list)
    watts_under_load: list[float] = field(default_factory=list)

    @property
    def energy_per_token_joules(self) -> list[float]:
        """Joules attributable to generating one token, per repetition.

        The delta against idle divided by the token rate. Absolute power over the rate
        would charge the model for the device's existence.
        """
        return [
            (watts - self.baseline.mean) / rate
            for watts, rate in zip(self.watts_under_load, self.tokens_per_second, strict=True)
        ]

    def reproducible(self) -> tuple[bool, str]:
        """Whether the repetitions agree within the instrument's own noise.

        Checked on power, because that is the quantity the ROPE is expressed in. Latency and
        memory are reported with their spread but are not gated: a latency that varies is a
        fact about the device under load, and hiding it behind a threshold derived from a
        power instrument would be borrowing a tolerance from the wrong measurement.
        """
        if len(self.watts_under_load) < self.protocol.repetitions:
            return False, (
                f"{len(self.watts_under_load)} of {self.protocol.repetitions} repetitions "
                "recorded"
            )
        spread = max(self.watts_under_load) - min(self.watts_under_load)
        rope = self.baseline.rope_watts
        if spread > 2 * rope:
            return False, (
                f"repetitions span {spread:.4f} W against a ROPE of {rope:.4f} W. The "
                "measurement is not repeatable under its own protocol, so it is not "
                "reported."
            )
        return True, f"repetitions span {spread:.4f} W within a ROPE of {rope:.4f} W"

    def as_dict(self) -> dict[str, object]:
        ok, why = self.reproducible()

        def spread(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            return {
                "median": round(statistics.median(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        return {
            "architecture": self.architecture,
            "protocol": self.protocol.as_dict(),
            "baseline": self.baseline.as_dict(),
            "latency_ms_per_token": spread(self.latency_ms_per_token),
            "tokens_per_second": spread(self.tokens_per_second),
            "peak_memory_bytes": spread([float(v) for v in self.peak_memory_bytes]),
            "energy_per_token_joules": spread(self.energy_per_token_joules),
            "reproducible": ok,
            "reproducibility_note": why,
        }


def require_reproducible(measurement: DeploymentMeasurement) -> None:
    """Refuse to publish a measurement its own repetitions do not agree on."""
    ok, why = measurement.reproducible()
    if not ok:
        raise NotReproducible(
            f"{measurement.architecture}: {why}\n"
            "A deployment figure that does not repeat is a property of the run rather than "
            "of the architecture, and publishing it would invite exactly the objection this "
            "project exists to answer."
        )
