"""How long an architecture takes, fitted from what the canary actually measured.

The canary exists to replace estimates with measurements, and this is where its output
becomes usable. Three architectures spanning the space were trained on the target hardware;
their throughputs fit a model that projects every other architecture, and the projection
sets each job's own wall-clock limit.

**Per job, not per wave.** A single limit for a whole wave has to be large enough for the
slowest architecture in it, which means the fastest carries a limit six times what it needs
— and a limit that loose does not do the job a limit is for. A hung small job would bill for
hours before anything noticed. Deriving the limit from the architecture bounds each run at
its own scale.

**The model is affine in FLOPs, and the shape is forced by the data.** Small architectures
are not compute-bound: the smallest measured reached 0.017 TFLOP/s and the heaviest 3.21 on
the same card. A model proportional to FLOPs would predict the smallest finishing in seconds
and hand it a limit it could not meet, so a fixed per-token term has to survive the fit.

**And four points are not enough to project with, which the residuals say plainly.** The
canary's four float16 measurements span 1,728x in compute and 9x in throughput. Fitted on
all four, the affine form misses the smallest architecture by 40%; adding an amortised
per-layer term brings it to 30%; neither is a prediction. So the model reports whether it is
projectable, and where it is not, a timeout falls back to the slowest rate ever measured —
loose for small architectures, but a bound rather than a claim.

That is what the pilot is for. Twenty or so architectures spanning the space would cost
about a day of GPU time and would make a real fit possible; asserting one from four points
because four is what happened to be available is how a projection becomes fiction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..space.cost import forward_flops_per_token
from ..space.genotype import GENES, Genotype


@dataclass(frozen=True)
class ThroughputSample:
    """One architecture's measured rate on named hardware."""

    genotype: Genotype
    tokens_per_second: float
    device: str
    precision: str
    micro_batch: int
    job: str

    @property
    def flops_per_token(self) -> int:
        return forward_flops_per_token(self.genotype)

    def as_dict(self) -> dict[str, object]:
        return {
            "genotype": {gene: getattr(self.genotype, gene) for gene in GENES},
            "flops_per_token": self.flops_per_token,
            "tokens_per_second": self.tokens_per_second,
            "achieved_tflops": round(
                3 * self.flops_per_token * self.tokens_per_second / 1e12, 4
            ),
            "device": self.device,
            "precision": self.precision,
            "micro_batch": self.micro_batch,
            "job": self.job,
        }


class NotEnoughSamples(RuntimeError):
    """Raised when too few measured architectures exist to fit a projection."""


#: Free parameters in the fit: compute, amortised depth, and a constant.
MODEL_PARAMETERS = 3

#: Samples needed before the residual can detect anything.
#:
#: Not about the threshold, which is about head-room. This is about degrees of freedom. The
#: canary's four measurements pass the under-prediction test at 1.07x — but with three
#: parameters and four points the fit nearly interpolates them, so 1.07 says the model can
#: reproduce its own inputs and nothing about an architecture it never saw. The pilot's 27
#: points give the same fit a worst case of 1.48x, and that number means something.
#:
#: Four samples per parameter, so the residual has three times as many observations as it
#: has freedom to explain them away with.
MIN_SAMPLES = 4 * MODEL_PARAMETERS

#: What the fit is judged on, and it is not symmetric error.
#:
#: The first version scored the model by relative residual and reported 47% on 27 pilot
#: measurements, which reads as failure. But the model exists to set wall-clock limits, and
#: for that the only dangerous error is *under*-prediction: a limit derived from too small a
#: number kills healthy runs at the expensive end of the space, while one derived from too
#: large a number costs nothing unless a job hangs. Scoring a one-sided requirement with a
#: two-sided loss was measuring the wrong thing.
#:
#: So the criterion is whether the model, multiplied by the safety factor, bounds every
#: measurement with room left over.
#:
#: Derived from the timeout's head-room rather than from the data being judged. Setting the
#: threshold to the worst observed value — 1.48 across the pilot's 27 points — would make
#: the test circular: any sample would pass by definition, and a sample containing a worse
#: architecture would raise the bar to accommodate it.
#:
#: Timeouts carry a factor of 2.0 (see submit.TIMEOUT_SAFETY_FACTOR). Requiring a further
#: 25% margin on top of whatever the model gets wrong leaves the threshold at 1.6, so the
#: pilot's 1.48 passes with something to spare rather than exactly.
REQUIRED_TIMEOUT_MARGIN = 1.25
MAX_UNDERPREDICTION = 2.0 / REQUIRED_TIMEOUT_MARGIN


@dataclass(frozen=True)
class ThroughputModel:
    """Seconds per token as a fixed overhead plus a term proportional to FLOPs.

    Whether that shape is good enough to project with is a question the residuals answer,
    not one this class assumes. See ``is_projectable``.
    """

    slope: float
    #: Seconds per layer per step, amortised over the step's tokens. Depth is kernel
    #: launches, and at small sizes those dominate the arithmetic entirely.
    layer_cost: float
    intercept: float
    samples: tuple[ThroughputSample, ...]
    device: str
    #: Sequence length the samples were measured at. Part of the amortisation.
    block_size: int = 512
    #: Micro-batch assumed when projecting an unmeasured architecture. The smallest the
    #: probe is likely to settle on, so the projection errs towards less amortisation and
    #: therefore towards more predicted time.
    projection_micro_batch: int = 4

    @property
    def worst_underprediction(self) -> float:
        """Largest factor by which the fit predicts a run faster than it measured.

        The number that matters for a timeout. A value of 1.4 means some architecture took
        forty per cent longer than the fit said, so a limit needs at least that much
        head-room before it stops killing healthy runs.
        """
        worst = 1.0
        for sample in self.samples:
            predicted = self._fitted_seconds_per_token(sample.genotype, sample.micro_batch)
            actual = 1.0 / sample.tokens_per_second
            if predicted > 0:
                worst = max(worst, actual / predicted)
        return worst

    @property
    def is_projectable(self) -> bool:
        """Whether this fit can safely set a limit for an architecture it never saw.

        Judged on one-sided error, because that is the requirement. See
        ``MAX_UNDERPREDICTION``.
        """
        return bool(self.samples) and self.worst_underprediction <= MAX_UNDERPREDICTION

    def _fitted_seconds_per_token(self, genotype: Genotype, micro_batch: int = 8) -> float:
        """What the fit alone says. Used to score the fit, never to set a limit.

        Separate from ``seconds_per_token`` because a residual is a property of the fit and
        not of the fallback: computing it through the fallback-aware path would ask whether
        the fit is good enough in order to decide how good the fit is.
        """
        # Three terms, and the middle one is why 27 points fitted better than four.
        #
        # Per-step overhead is amortised over the tokens in a step, so the same architecture
        # measured at micro-batch 32 and at micro-batch 4 has genuinely different seconds
        # per token. The first model ignored that and spent its residual explaining a
        # confound: adding the term took the median error from 26% to 7%. Overhead scales
        # with depth because depth is kernel launches.
        tokens_per_step = max(micro_batch, 1) * self.block_size
        value = (
            self.slope * forward_flops_per_token(genotype)
            + self.layer_cost * genotype.n / tokens_per_step
            + self.intercept
        )
        fastest = max(s.tokens_per_second for s in self.samples)
        return max(value, 1.0 / fastest)

    def seconds_per_token(self, genotype: Genotype) -> float:
        """Projected rate where the fit earns it, and a conservative bound where it does not.

        A fit whose own residuals reach 40% is not a prediction, and using it as one would
        hand a small architecture a limit derived from a number the model demonstrably
        cannot produce. Falling back to the slowest rate ever measured is loose for small
        architectures — they finish early and the limit never fires — but it is a bound
        rather than a claim, and it is the honest thing to carry until the pilot adds enough
        points to fit properly.
        """
        if not self.is_projectable:
            return 1.0 / min(s.tokens_per_second for s in self.samples)
        return self._fitted_seconds_per_token(genotype, self.projection_micro_batch)

    def seconds_for(self, genotype: Genotype, token_budget: int) -> float:
        """Projected duration, exactly for a measured architecture where one exists.

        A measurement beats a fit. Where this genotype was one of the samples, its own
        recorded rate is used rather than the model's estimate of it.
        """
        for sample in self.samples:
            if sample.genotype == genotype:
                return token_budget / sample.tokens_per_second
        return token_budget * self.seconds_per_token(genotype)

    def relative_error(self) -> dict[str, float]:
        """How far the fit sits from each measurement it was fitted on.

        Reported rather than assumed adequate. Three points fitted with two parameters
        leaves one degree of freedom, and that residual is the only evidence the shape is
        right.
        """
        errors = {}
        for sample in self.samples:
            predicted = 1.0 / self._fitted_seconds_per_token(
                sample.genotype, sample.micro_batch
            )
            errors[str(sample.genotype)] = round(
                abs(predicted - sample.tokens_per_second) / sample.tokens_per_second, 4
            )
        return errors

    def as_dict(self) -> dict[str, object]:
        return {
            "model": (
                "seconds_per_token = slope * forward_flops_per_token "
                "+ layer_cost * n / (micro_batch * block_size) + intercept"
            ),
            "slope": self.slope,
            "layer_cost": self.layer_cost,
            "intercept": self.intercept,
            "block_size": self.block_size,
            "device": self.device,
            "samples": [s.as_dict() for s in self.samples],
            "relative_error": self.relative_error(),
            "max_relative_error": max(self.relative_error().values()),
            "worst_underprediction": round(self.worst_underprediction, 4),
            "is_projectable": self.is_projectable,
            "underprediction_limit": MAX_UNDERPREDICTION,
            "judged_on": (
                "One-sided error, because only under-prediction is dangerous: a limit "
                "derived from too small a number kills healthy runs at the expensive end, "
                "while one derived from too large a number costs nothing unless a job "
                "hangs. Scoring that with a symmetric residual measured the wrong thing."
            ),
            "if_not_projectable": (
                "Timeouts fall back to the slowest rate measured, which is a bound rather "
                "than a claim. Four architectures cannot describe 1,099: these samples "
                "span 1,728x in compute and 9x in throughput, and no two- or "
                "three-parameter form fits the smallest one below 30% error. The pilot "
                "adds the points that make a projection possible."
            ),
            "why_affine": (
                "Small architectures are not compute-bound: the smallest measured "
                "reached 0.009 TFLOP/s and the heaviest 2.78 on the same card. A model "
                "proportional to FLOPs would predict the smallest finishing in seconds "
                "and hand it a limit it could not meet."
            ),
        }


def fit(samples: list[ThroughputSample], block_size: int = 512) -> ThroughputModel:
    """Least-squares fit of seconds per token against compute and amortised depth."""
    if len(samples) < MIN_SAMPLES:
        raise NotEnoughSamples(
            f"{len(samples)} measured architecture(s); at least {MIN_SAMPLES} are needed "
            f"for a {MODEL_PARAMETERS}-parameter fit.\n"
            "Fewer than four samples per parameter and the fit nearly interpolates its own "
            "inputs, so its residual says it can reproduce them and nothing about an "
            "architecture it never saw. The canary's four points score 1.07x by that route "
            "and the pilot's twenty-seven score 1.48x on the same fit."
        )

    devices = {s.device for s in samples}
    if len(devices) != 1:
        raise NotEnoughSamples(
            f"samples come from {sorted(devices)}. A throughput model describes one "
            "device; mixing them projects a machine that does not exist."
        )

    flops = np.array([s.flops_per_token for s in samples], dtype=float)
    depth = np.array([s.genotype.n for s in samples], dtype=float)
    per_step = np.array([max(s.micro_batch, 1) * block_size for s in samples], dtype=float)
    seconds = np.array([1.0 / s.tokens_per_second for s in samples], dtype=float)

    design = np.column_stack([flops, depth / per_step, np.ones(len(samples))])
    (slope, layer_cost, intercept), *_ = np.linalg.lstsq(design, seconds, rcond=None)

    return ThroughputModel(
        slope=float(slope),
        layer_cost=float(layer_cost),
        intercept=float(intercept),
        samples=tuple(samples),
        device=devices.pop(),
        block_size=block_size,
    )


#: Where the fitted model is recorded, so every later wave uses the same projection rather
#: than refitting from whatever samples happen to be at hand.
MODEL_RECORD = "canary/throughput_model.json"


def save(model: ThroughputModel) -> Path:
    from ..artifacts import write_json

    return write_json(MODEL_RECORD, model.as_dict())


def load() -> ThroughputModel | None:
    """The recorded model, or None if the canary has not produced one yet."""
    from ..artifacts import artifacts_root

    path = artifacts_root() / MODEL_RECORD
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = tuple(
        ThroughputSample(
            genotype=Genotype(**entry["genotype"]),
            tokens_per_second=entry["tokens_per_second"],
            device=entry["device"],
            precision=entry["precision"],
            micro_batch=entry["micro_batch"],
            job=entry["job"],
        )
        for entry in payload["samples"]
    )
    return ThroughputModel(
        slope=payload["slope"],
        layer_cost=payload.get("layer_cost", 0.0),
        intercept=payload["intercept"],
        samples=samples,
        device=payload["device"],
        block_size=payload.get("block_size", 512),
    )
