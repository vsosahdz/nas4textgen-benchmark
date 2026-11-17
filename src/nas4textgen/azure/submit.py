"""Build and submit one training job per architecture. The sweep's only writer.

Three properties matter more than convenience here, and each costs a little code.

**A job is identified by what it measures, not by when it was launched.** Its identity is a
hash of the genotype, corpus, seed and budget. Re-running a wave after an interruption
therefore skips what already succeeded instead of paying for it again, and two rows of the
benchmark can never disagree about which run produced them.

**No job can outlive its usefulness.** Every job carries a wall-clock limit. A spot node
whose job hangs is a node billing until someone notices, and nobody notices at three in the
morning. The limit is set from the canary's measurement rather than guessed.

**Every job records what produced it.** Image digest, corpus vintage, lock hash and
genotype travel as tags, so the workspace itself can answer which code and which data
produced a given number — without trusting a spreadsheet kept alongside.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..space.genotype import GENES, Genotype
from .workspace import (
    Platform,
    image_reference,
    ml_client,
    require_current_image,
    resolve_platform,
)

# Azure ML chooses where it mounts an input and an output, and the path contains the run
# id. Binding expressions are how a command asks for the real one; a fixed path is a
# different directory that happens to exist.
#
# Getting this wrong fails in two very different ways. On the input side the job stops
# immediately with "corpus unavailable", which is loud and cheap. On the output side it does
# not fail at all: /mnt/outputs is writable local disk, so the run trains, evaluates, writes
# its manifest, exits zero — and the results disappear when the spot node is reclaimed.
# A silent success that produces nothing is the worse of the two, and it is the one a fixed
# path invites.
CACHE_MOUNT = "${{inputs.cache}}"
OUTPUT_MOUNT = "${{outputs.outputs}}"

#: Head-room over the projected training time, as a multiplier and a fixed addition.
#:
#: A flat multiplier is the wrong shape. Three times a forty-minute job is two hours, which
#: is reasonable; three times a three-and-a-half-hour job is ten, which protects nothing —
#: a hung job would bill most of a day. The costs a limit has to absorb are of two kinds:
#: proportional ones, like a slower node or a preemption-and-restart, and fixed ones, like
#: pulling a six-gigabyte image and mounting a half-gigabyte corpus. So the allowance has
#: both terms.
#: Derived from the pilot rather than chosen. Across 27 measured architectures the fitted
#: model's worst under-prediction is a factor of 1.48, so 1.5 bounded them with 1.3% to
#: spare — which is not margin, it is luck, on 27 of 1,099 architectures. Two gives 35%
#: head-room over the worst thing actually observed.
#:
#: The asymmetry is the argument: a limit that is too tight kills a healthy run at the
#: expensive end of the space and wastes everything it had computed, while a limit that is
#: too loose costs nothing at all unless a job hangs.
TIMEOUT_SAFETY_FACTOR = 2.0
TIMEOUT_FIXED_OVERHEAD_SECONDS = 900

#: Used only until the canary reports real figures. Deliberately not a round number, so it
#: is recognisable in the workspace as a default that was never replaced.
PROVISIONAL_TIMEOUT_SECONDS = 5400


#: Where the process's own deadlines sit inside the scheduler's limit, as fractions of it.
#:
#: The ordering is the whole point and it was inverted. The defaults in TrainingConfig put
#: the soft stop at 7200s and the hard stop at 8100s while the job limit was 5400s, so Azure
#: ML would have killed every long run before either fired. The graceful path — stop
#: training, still evaluate, still write — was unreachable code, and a run that hit the wall
#: would have lost everything it had computed.
#:
#: So the deadlines are derived from the limit rather than defaulted alongside it. The soft
#: stop leaves a quarter of the budget to evaluate and write; the hard stop is the process
#: releasing its own node; the scheduler's limit is the backstop for a process too wedged to
#: honour either.
SOFT_DEADLINE_FRACTION = 0.70
HARD_DEADLINE_FRACTION = 0.85


def process_deadlines(timeout_seconds: int) -> tuple[int, int]:
    """The soft and hard deadlines a job carries, always inside the scheduler's limit."""
    soft = int(timeout_seconds * SOFT_DEADLINE_FRACTION)
    hard = int(timeout_seconds * HARD_DEADLINE_FRACTION)
    return soft, hard


class TimeoutNotMeasured(RuntimeError):
    """Raised when a sweep wave is submitted before the canary measured a job's duration."""


@dataclass(frozen=True)
class JobSpec:
    """One row of the benchmark, before it exists."""

    genotype: Genotype
    corpus: str
    seed: int
    token_budget: int
    train_split: str = "train"
    eval_split: str = "validation"
    #: Whether the job computes the reporting battery as well as the objective. It is part
    #: of the fingerprint below because a battery run produces a different set of
    #: measurements, so it is a different measurement even though perplexity is unchanged.
    run_battery: bool = False
    extra_tags: dict[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Identity of the measurement, independent of when it was submitted.

        Everything that changes the number belongs in here; nothing else does. Wall-clock
        limits and cluster names are absent on purpose — a run that was given a longer
        timeout measures the same thing, and re-submitting it would be waste.
        """
        parts = [
            *(f"{gene}={getattr(self.genotype, gene)}" for gene in GENES),
            f"corpus={self.corpus}",
            f"train={self.train_split}",
            f"eval={self.eval_split}",
            f"seed={self.seed}",
            f"tokens={self.token_budget}",
        ]
        # Appended only when set, so every fingerprint minted before the battery existed
        # stays what it was. Adding an unconditional term would change the identity of all
        # 3,123 completed runs and offer to measure the census again.
        if self.run_battery:
            parts.append("battery=1")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def name(self, attempt: int = 1) -> str:
        """Azure ML job name: the fingerprint plus which attempt this is.

        The attempt suffix is not decoration. An Azure ML job name is permanent — a job
        that failed still holds its name forever, and resubmitting under it returns
        "Existing properties cannot be modified" rather than running. So the fingerprint
        stays the identity of the *measurement*, and the suffix distinguishes attempts at
        it. Skipping duplicates is decided by looking up what already exists, not by hoping
        a name collision will do it.
        """
        return f"nas-{self.fingerprint}-a{attempt}"

    @property
    def display_name(self) -> str:
        g = self.genotype
        return f"h{g.h}-d{g.d_head}-n{g.n}-f{g.ffn}-kv{g.kv} {self.corpus} s{self.seed}"

    def tags(self, image: str, cache_vintage: str, attempt: int = 1) -> dict[str, str]:
        return {
            **{gene: str(getattr(self.genotype, gene)) for gene in GENES},
            "e": str(self.genotype.e),
            "corpus": self.corpus,
            "seed": str(self.seed),
            "token_budget": str(self.token_budget),
            "fingerprint": self.fingerprint,
            "attempt": str(attempt),
            "image": image,
            "data_vintage": cache_vintage,
            **self.extra_tags,
        }

    def command_arguments(self, timeout_seconds: int) -> str:
        soft, hard = process_deadlines(timeout_seconds)
        genes = " ".join(
            f"--{gene.replace('_', '-')} {getattr(self.genotype, gene)}" for gene in GENES
        )
        return (
            f"{genes} --corpus {self.corpus} --seed {self.seed} "
            f"--token-budget {self.token_budget} "
            f"--train-split {self.train_split} --eval-split {self.eval_split} "
            f"--soft-deadline-seconds {soft} --hard-deadline-seconds {hard} "
            f"--cache {CACHE_MOUNT} --output {OUTPUT_MOUNT}"
            + (" --run-battery" if self.run_battery else "")
        )


def build_job(
    spec: JobSpec,
    platform: Platform,
    image: str,
    cluster: str,
    timeout_seconds: int,
    cache_vintage: str,
    attempt: int = 1,
):
    """Assemble the Azure ML job. No network calls; unit-testable without a subscription."""
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.entities import Environment

    from .data import ARTIFACT_DATASTORE, CACHE_DATASTORE

    # Datastore paths, not https:// URLs. Azure ML treats the http(s) scheme as public
    # storage and refuses it as an output — the job is rejected at validation, before a node
    # is allocated. That is the cheap failure; the expensive version would be an input that
    # silently resolved somewhere unintended.
    cache_uri = f"azureml://datastores/{CACHE_DATASTORE}/paths/corpora"
    output_uri = f"azureml://datastores/{ARTIFACT_DATASTORE}/paths/runs/{spec.fingerprint}"

    return command(
        name=spec.name(attempt),
        display_name=spec.display_name,
        experiment_name=f"sweep-{spec.corpus}",
        description=(
            "One architecture, one corpus, one seed. The unit of the benchmark. "
            "Re-submitting this fingerprint is refused rather than repeated."
        ),
        # Azure ML replaces the image's ENTRYPOINT with the command it is given, so the
        # module is named again here. The build asserts `python -m nas4textgen.train
        # --help` resolves, which is what keeps these two from drifting apart.
        command=f"python -m nas4textgen.train {spec.command_arguments(timeout_seconds)}",
        # The image by digest. Azure ML pulls it from the workspace's own registry using
        # the workspace's managed identity — no registry credential is stored anywhere
        # this repository can see, which is what lets the replication package be public.
        environment=Environment(image=image),
        environment_variables={
            # No NAS4TG_CACHE here: the mount path is only known once Azure ML resolves the
            # binding, so it arrives as --cache on the command line instead.
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        inputs={
            "cache": Input(
                type=AssetTypes.URI_FOLDER,
                path=cache_uri,
                # Read-only, and mounted rather than downloaded: a half-gigabyte corpus
                # copied onto every node would add minutes of billed time per job for no
                # benefit, and a writable corpus is a corpus a job could change.
                mode=InputOutputModes.RO_MOUNT,
            )
        },
        outputs={
            "outputs": Output(
                type=AssetTypes.URI_FOLDER,
                path=output_uri,
                mode=InputOutputModes.RW_MOUNT,
            )
        },
        compute=cluster,
        tags=spec.tags(image, cache_vintage, attempt),
        instance_count=1,
        # The hard stop. Azure ML cancels the job and releases the node; nothing depends on
        # the process noticing its own deadline, because a process wedged in a driver call
        # is exactly the case that would not.
        timeout=timeout_seconds,
    )


def resolve_timeout(measured_seconds: float | None, allow_provisional: bool = False) -> int:
    """Turn a projected training time into the limit a job carries.

    Refusing to guess is the point. A timeout picked by intuition is either so loose it
    protects nothing or so tight it kills healthy runs at the large end of the space, and
    both failures are discovered by paying for them.
    """
    if measured_seconds is None:
        if not allow_provisional:
            raise TimeoutNotMeasured(
                "no measured per-job wall clock. Run the canary first: its architectures "
                "fit the throughput model every sweep job's limit is projected from. Pass "
                "allow_provisional=True only for the canary itself."
            )
        return PROVISIONAL_TIMEOUT_SECONDS
    return int(measured_seconds * TIMEOUT_SAFETY_FACTOR + TIMEOUT_FIXED_OVERHEAD_SECONDS)


def timeout_for(spec: JobSpec, model=None, allow_provisional: bool = False) -> int:
    """The limit this particular architecture carries.

    Per job rather than per wave. One limit for a whole wave has to be large enough for the
    slowest architecture in it, so the fastest carries six times what it needs — and a
    limit that loose does not do what a limit is for. A hung small job would bill for hours
    before anything noticed.
    """
    from . import throughput

    model = model if model is not None else throughput.load()
    if model is None:
        return resolve_timeout(None, allow_provisional=allow_provisional)
    return resolve_timeout(model.seconds_for(spec.genotype, spec.token_budget))


#: Where the canary records what it measured. Written once, read by every wave after.
MEASURED_TIMEOUT_RECORD = "canary/measured_timeout.json"


def record_measured_timeout(
    seconds: float,
    source_role: str,
    genotype: Genotype,
    job_name: str,
) -> str:
    """Persist the canary's wall clock so later waves inherit it instead of guessing.

    Written as a file rather than passed by hand because a wave submitted next week must
    carry the same limit as one submitted today, and a number retyped from a terminal is a
    number that eventually gets retyped wrong.
    """
    from ..artifacts import write_json

    path = write_json(
        MEASURED_TIMEOUT_RECORD,
        {
            "measured_seconds": seconds,
            "safety_factor": TIMEOUT_SAFETY_FACTOR,
            "timeout_seconds": resolve_timeout(seconds),
            "source_role": source_role,
            "source_genotype": {gene: getattr(genotype, gene) for gene in GENES},
            "source_job": job_name,
            "why": (
                "Taken from the architecture with the most forward compute per token, not "
                "the one with the most parameters: parameters here are dominated by the "
                "embedding matrix, so the parameter extreme is merely the widest model and "
                "runs well inside the bound."
            ),
        },
    )
    return str(path)


def measured_timeout() -> float | None:
    """The canary's wall clock, if it has been measured. None means no sweep may go out."""
    import json

    from ..artifacts import artifacts_root

    path = artifacts_root() / MEASURED_TIMEOUT_RECORD
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["measured_seconds"])


@dataclass
class SubmissionOutcome:
    spec: JobSpec
    status: str  # submitted | already_present | failed | dry_run | would_skip
    job_name: str
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.spec.fingerprint,
            "job_name": self.job_name,
            "status": self.status,
            "display_name": self.spec.display_name,
            "detail": self.detail,
        }


#: Azure ML states from which a job will not produce a result, so re-attempting is correct.
RESUBMITTABLE = {"Failed", "Canceled", "NotResponding"}

#: A Completed job whose result is not in the container is resubmittable too, and that
#: needs saying because it sounds like a contradiction.
#:
#: Two hundred and thirteen jobs of wave four were accepted, never scheduled, and marked
#: Completed at the instant they were created — no container, no log, no output. By status
#: they were successes; by result they did not happen. Treating status as the authority
#: would leave those architectures permanently unmeasurable, since nothing about a
#: successful job invites a retry, and the campaign would have ended 213 rows short with
#: every counter reporting completion.
#:
#: So the manifest decides. A job's status is a claim about the job; the manifest is the
#: measurement, and only one of the two can be checked against the benchmark.
COMPLETED_WITHOUT_RESULT_IS_RESUBMITTABLE = True

#: How many attempts one measurement may have **on one image** before it is left alone.
#:
#: Counted per image, not per fingerprint, and the distinction is not pedantic. The guard
#: exists to stop a broken specification from being retried forever across waves; it is not
#: meant to stop a fixed harness from being retried at all. The canary reached four failed
#: attempts on four different images — a corpus path, then a mount binding, then a dtype,
#: then a memory probe — and each fix made the next attempt a genuinely new experiment.
#: Counting those against a single budget would have refused to run the version that works.
MAX_ATTEMPTS_PER_IMAGE = 4

#: Total attempts across all images, as the backstop the per-image count no longer provides.
#: High enough that a real sequence of fixes is never blocked, low enough that a resubmit
#: loop cannot run unattended.
MAX_ATTEMPTS_TOTAL = 16


FINISHED_STATES = {"Completed"}


def written_results(platform: Platform) -> set[str]:
    """Fingerprints whose manifest is in the artifact container.

    Read once per wave rather than per job: it is one listing against the container, and a
    per-job check would turn a 400-spec submission into 400 extra round trips for an answer
    that does not change while the wave is being queued.
    """
    from .data import _blob_service

    container = _blob_service(platform).get_container_client(platform.artifact_container)
    return {
        blob.name.split("/")[1]
        for blob in container.list_blobs(name_starts_with="runs/")
        if blob.name.endswith("manifest.json")
    }


def _next_attempt(
    client, spec: JobSpec, image: str, force: bool = False,
    results: set[str] | None = None,
) -> tuple[int | None, str | None]:
    """Find which attempt to submit next, by asking the workspace what exists.

    Returns ``(attempt, None)`` when a submission should go out, or ``(None, reason)`` when
    it should not. The workspace is the ledger: a local record and the workspace disagree
    the moment a submission is interrupted between the call and the write, and that
    disagreement always resolves towards paying twice.

    Attempts made under a different image do not count against the retry budget. They were
    a different harness, and treating them as the same one means a fix cannot be tried.
    """
    on_this_image = 0
    for attempt in range(1, MAX_ATTEMPTS_TOTAL + 1):
        status, ran_image = _existing_attempt(client, spec.name(attempt))
        if status is None:
            if force:
                # Re-measuring something already measured. Needed when a completed row is
                # not comparable with the rest — the canary produced one under emulated
                # bfloat16 while the others used float16 — since the fingerprint describes
                # what was asked for and not the arithmetic that answered it.
                return attempt, None
            if on_this_image >= MAX_ATTEMPTS_PER_IMAGE and not force:
                return None, (
                    f"{on_this_image} attempts have already failed on this exact image. "
                    "Something about this specification is wrong, and another node will "
                    "not fix it — change the image or the spec before retrying."
                )
            return attempt, None
        finished_but_empty = (
            status in FINISHED_STATES
            and results is not None
            and spec.fingerprint not in results
        )
        if status not in RESUBMITTABLE and not finished_but_empty and not force:
            return None, f"attempt {attempt} exists in state {status}; not resubmitted"
        if finished_but_empty:
            # Not counted against the per-image budget: the job did not run, so this is a
            # first attempt at the measurement rather than a repeat of a failed one.
            continue
        if ran_image == image:
            on_this_image += 1
    return None, (
        f"{MAX_ATTEMPTS_TOTAL} attempts in total have failed across every image tried."
    )


def submit_wave(
    specs: list[JobSpec],
    cluster: str | None = None,
    measured_seconds: float | None = None,
    allow_provisional: bool = False,
    cache_vintage: str = "unknown",
    dry_run: bool = False,
    force_roles: tuple[str, ...] = (),
) -> list[SubmissionOutcome]:
    """Submit every spec that has no live or completed job already.

    The existence check is per fingerprint against the workspace, not against a local
    ledger. A ledger and a workspace disagree the moment a submission is interrupted after
    the call and before the write, and the disagreement always resolves towards paying
    twice.
    """
    from . import throughput

    platform = resolve_platform()
    # Before anything is queued: the image must contain the code in this checkout. Wave one
    # of the sweep went out on an image built before a fix to the memory probe, and six of
    # its first thirty-seven jobs failed with the error that fix addressed.
    require_current_image()
    image = image_reference()
    target = cluster or platform.sweep_cluster

    model = throughput.load()
    if model is None and measured_seconds is None and not allow_provisional:
        # Keep the original refusal reachable: without any measurement there is no limit
        # to derive, and guessing one is what this whole path exists to avoid.
        resolve_timeout(None, allow_provisional=False)

    def limit_for(spec: JobSpec) -> int:
        if model is not None:
            return timeout_for(spec, model=model)
        return resolve_timeout(measured_seconds, allow_provisional=allow_provisional)

    client = ml_client(platform)
    results = written_results(platform)

    if dry_run:
        # The workspace is consulted here too, for the same reason the real path consults
        # it. A dry run that assumes every spec is new answers a different question from
        # the one asked: re-running a wave to recover four lost jobs reported that it would
        # submit three hundred, which is the kind of number a decision gets made on.
        # Reading the workspace queues nothing.
        outcomes = []
        for spec in specs:
            forced = spec.extra_tags.get("canary_role") in force_roles
            attempt, reason = _next_attempt(client, spec, image, force=forced,
                                            results=results)
            if attempt is None:
                outcomes.append(SubmissionOutcome(
                    spec, "would_skip", spec.name(1), detail=reason or "",
                ))
                continue
            outcomes.append(SubmissionOutcome(
                spec, "dry_run", spec.name(attempt),
                detail=f"would run attempt {attempt} on {target} with a "
                       f"{limit_for(spec)}s limit",
            ))
        return outcomes

    outcomes: list[SubmissionOutcome] = []
    for spec in specs:
        forced = spec.extra_tags.get("canary_role") in force_roles
        attempt, reason = _next_attempt(client, spec, image, force=forced,
                                        results=results)
        if attempt is None:
            outcomes.append(
                SubmissionOutcome(spec, "already_present", spec.name(1), detail=reason or "")
            )
            continue

        timeout = limit_for(spec)
        job = build_job(spec, platform, image, target, timeout, cache_vintage, attempt)
        try:
            created = client.jobs.create_or_update(job)
            outcomes.append(
                SubmissionOutcome(spec, "submitted", created.name,
                                  detail=f"attempt {attempt} on {target}, limit {timeout}s")
            )
        except Exception as exc:  # noqa: BLE001
            # One rejected job must not abandon the rest of the wave un-submitted with
            # nothing written down about which ones those were.
            outcomes.append(
                SubmissionOutcome(spec, "failed", spec.name(attempt),
                                  detail=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
            )
    return outcomes


def _existing_attempt(client, name: str) -> tuple[str | None, str | None]:
    """Status and the image an existing attempt ran, or (None, None) if it never existed."""
    try:
        job = client.jobs.get(name)
    except Exception:  # noqa: BLE001
        return None, None
    return job.status, (job.tags or {}).get("image")
