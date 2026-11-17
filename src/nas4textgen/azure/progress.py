"""Where a multi-day campaign has got to, computed without touching it.

A sweep of eleven to thirteen days cannot be watched. What it needs is a report that
answers, from outside, whether it is still going, how far it has got, what it has cost, and
when it will finish — and that answers them without doing anything a running job could
notice.

**Read-only, and that is a property worth stating rather than assuming.** Everything here
comes from listing job metadata and reading manifests that jobs have already written. It
submits nothing, cancels nothing, and does not open the corpus or the checkpoints. Running
it during a wave is as safe as not running it.

**A row is counted when its result exists, not when its job says Completed.** Those are
different numbers and the difference is not small: 210 jobs of wave four were accepted,
never scheduled, and marked Completed at the instant they were created — no container, no
log, no output. Counting job status reported 1,100 of 1,099 architectures measured, which
is more than the space holds; the tell was the figure exceeding 100 per cent, and it had
been inflating the report for days before that. What becomes a row in the benchmark is a
manifest in the container, so that is what is counted, and a job that finished without
leaving one is reported under its own heading rather than averaged into the total.

**The estimate is honest about what it does not know.** Time remaining is derived from the
rate at which rows have actually completed, not from a projection of how long each
architecture should take — the throughput model already declined to project from four
points, and a completion rate measures the real thing including queueing, preemption and
retries. Where too few rows exist to estimate a rate, it says so instead of extrapolating
from two points.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

#: Job states Azure ML reports. Grouped by what they mean for the campaign rather than by
#: the API's own taxonomy: what matters is whether a row exists, is coming, or will not.
FINISHED = {"Completed"}
IN_FLIGHT = {"Running", "Preparing", "Provisioning", "Starting", "Finalizing"}
WAITING = {"Queued", "NotStarted", "Scheduled"}
LOST = {"Failed", "Canceled", "NotResponding"}

#: Below this many completed rows, a completion rate is a straight line through noise and
#: the report says so rather than naming a date.
MIN_ROWS_FOR_ESTIMATE = 8


@dataclass
class RunRecord:
    """One job, reduced to what a progress report needs."""

    name: str
    fingerprint: str
    status: str
    created: datetime | None
    completed: datetime | None
    wave: str = ""
    role: str = ""
    #: The configuration this job measured. Carried because ``planned`` counts
    #: architectures at one configuration, and a fingerprint covers the corpus, seed and
    #: token budget too — so an auxiliary study at a different budget produces new
    #: fingerprints for architectures already counted, and the total climbs past the size
    #: of the space.
    corpus: str = ""
    seed: str = ""
    token_budget: str = ""


class CountsExceedTheSpace(RuntimeError):
    """Raised when more architectures are counted as measured than the space holds."""


@dataclass
class Progress:
    """The state of a campaign at one moment."""

    planned: int
    completed: int
    in_flight: int
    waiting: int
    lost: int
    distinct_measured: int
    node_hours: float
    failures: Counter = field(default_factory=Counter)
    #: Fingerprints whose job reported Completed and whose manifest is not in the
    #: container. Each is an architecture the benchmark does not have and the job list says
    #: it does.
    completed_without_result: list[str] = field(default_factory=list)
    resumes: int = 0
    started: datetime | None = None
    now: datetime | None = None
    #: Nodes the campaign can occupy. The projection is a rate per node-hour multiplied by
    #: this, so a quota change moves the finish date without re-measuring anything.
    concurrency: int = 5

    @property
    def remaining(self) -> int:
        return max(0, self.planned - self.distinct_measured)

    @property
    def fraction(self) -> float:
        return self.distinct_measured / self.planned if self.planned else 0.0

    @property
    def elapsed(self) -> timedelta | None:
        if self.started is None or self.now is None:
            return None
        return self.now - self.started

    def estimate(self) -> tuple[timedelta | None, str]:
        """Time remaining, from rows per node-hour rather than rows per wall-clock hour.

        The distinction is not pedantry; the first version of this got it badly wrong. It
        divided rows by elapsed wall clock, and elapsed wall clock spans the gaps between
        waves — the hours while a canary was being diagnosed, or while nothing was
        submitted at all. With the cluster busy ten per cent of the elapsed time, that
        arithmetic reported ninety-five days remaining for a sweep on track to finish in
        nine.

        Rows per node-hour measures what the machines achieved when they were working, and
        multiplying by the concurrency projects it forward. It still includes everything a
        per-architecture projection omits — queueing within a wave, preemption, retries —
        because those all consume node-hours.
        """
        if self.distinct_measured < MIN_ROWS_FOR_ESTIMATE:
            return None, (
                f"{self.distinct_measured} row(s) finished; at least "
                f"{MIN_ROWS_FOR_ESTIMATE} are needed before a rate means anything"
            )
        if self.node_hours <= 0:
            return None, "no node-hours recorded yet"

        per_node_hour = self.distinct_measured / self.node_hours
        per_hour = per_node_hour * self.concurrency
        if per_hour <= 0:
            return None, "no rows completed yet"

        return timedelta(hours=self.remaining / per_hour), (
            f"{per_node_hour:.2f} rows/node-hour over {self.node_hours:,.0f} node-hours, "
            f"at {self.concurrency} nodes"
        )

    def as_dict(self) -> dict[str, object]:
        remaining, why = self.estimate()
        return {
            "planned": self.planned,
            "measured": self.distinct_measured,
            "completed_without_result": len(self.completed_without_result),
            "completed_without_result_fingerprints": self.completed_without_result[:50],
            "fraction_complete": round(self.fraction, 4),
            "jobs": {
                "completed": self.completed,
                "in_flight": self.in_flight,
                "waiting": self.waiting,
                "lost": self.lost,
            },
            "node_hours": round(self.node_hours, 2),
            "resumes": self.resumes,
            "failures": dict(self.failures.most_common()),
            "elapsed_hours": (
                round(self.elapsed.total_seconds() / 3600, 2) if self.elapsed else None
            ),
            "concurrency": self.concurrency,
            "cluster_occupancy": (
                round(self.node_hours / (self.concurrency * self.elapsed.total_seconds() / 3600), 4)
                if self.elapsed and self.elapsed.total_seconds() > 0 else None
            ),
            "estimated_hours_remaining": (
                round(remaining.total_seconds() / 3600, 1) if remaining else None
            ),
            "estimated_finish": (
                (self.now + remaining).isoformat() if remaining and self.now else None
            ),
            "estimate_basis": why,
            "generated_at": self.now.isoformat() if self.now else None,
        }

    def render(self) -> str:
        remaining, why = self.estimate()
        width = 32
        filled = round(width * self.fraction)
        bar = "#" * filled + "." * (width - filled)

        lines = [
            "",
            f"  {self.distinct_measured:,} of {self.planned:,} rows measured "
            f"({self.fraction:.1%})",
            f"  [{bar}]",
            "",
            f"  {'running':<14}{self.in_flight:>6}",
            f"  {'queued':<14}{self.waiting:>6}",
            f"  {'completed':<14}{self.completed:>6}",
            f"  {'lost':<14}{self.lost:>6}",
        ]
        if self.completed_without_result:
            lines.append(
                f"  {'no result':<14}{len(self.completed_without_result):>6}   "
                "reported Completed and left no manifest — resubmit these"
            )
        if self.resumes:
            lines.append(
                f"  {'resumes':<14}{self.resumes:>6}   "
                "work recovered rather than repeated after a reclaim"
            )
        lines.append("")
        lines.append(f"  node-hours     {self.node_hours:>9,.1f}")

        if self.elapsed:
            hours = self.elapsed.total_seconds() / 3600
            lines.append(f"  elapsed        {hours:>9,.1f} h")
            # Reported because it is why wall clock is the wrong denominator: the gaps
            # between waves are hours in which nothing could have finished.
            occupancy = self.node_hours / (self.concurrency * hours) if hours else 0.0
            lines.append(f"  occupancy      {occupancy:>9.0%}   of the elapsed time, "
                         f"at {self.concurrency} nodes")
        if remaining is not None and self.now is not None:
            finish = (self.now + remaining).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"  remaining      {remaining.total_seconds()/3600:>9,.1f} h")
            lines.append(f"  finishes       {finish:>9}")
            lines.append(f"                 {why}")
        else:
            lines.append(f"  no estimate    {why}")

        if self.failures:
            lines.append("")
            lines.append("  lost jobs by cause")
            for cause, count in self.failures.most_common(6):
                lines.append(f"    {count:>4}  {cause}")

        return "\n".join(lines)


def _parse(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def summarise(
    records: list[RunRecord],
    planned: int,
    manifests: list[dict] | None = None,
    written_fingerprints: set[str] | None = None,  # required in practice; see below
    of_configuration: tuple[str, int, int] | None = None,
    now: datetime | None = None,
    concurrency: int = 5,
) -> Progress:
    """Reduce job records and written manifests to one report.

    Counted by fingerprint rather than by job, because a measurement retried three times is
    one row and three jobs. Reporting job counts would make a campaign that is failing and
    retrying look busier than one that is succeeding.
    """
    now = now or datetime.now(UTC)
    manifests = manifests or []

    completed = sum(r.status in FINISHED for r in records)
    in_flight = sum(r.status in IN_FLIGHT for r in records)
    waiting = sum(r.status in WAITING for r in records)
    lost = sum(r.status in LOST for r in records)

    # A job's status is a claim about the job; a manifest is the result the benchmark can
    # use, and only the second becomes a row.
    #
    # The fingerprints come from the caller because a manifest does not carry one — its
    # identity is the blob path, `runs/<fingerprint>/manifest.json`. Deriving it from the
    # content is what I tried first, guessing at three key names that are not there, and
    # the counter went from over-reporting every row to reporting none.
    # Restricted to the configuration `planned` refers to. Without this the 4x-budget
    # study's forty runs became forty more "measured architectures" and the report read
    # 1,118 of 1,099 — the third time this counter exceeded the size of the space, each
    # time by comparing two different units. What is being counted is architectures of the
    # sweep, so anything measured at another budget, seed or corpus is a different
    # question and belongs to a different total.
    def of_the_sweep(record: RunRecord) -> bool:
        if of_configuration is None:
            return True
        corpus, seed, budget = of_configuration
        return (
            (not record.corpus or record.corpus == corpus)
            and (not record.seed or record.seed == str(seed))
            and (not record.token_budget or record.token_budget == str(budget))
        )

    finished = {
        r.fingerprint for r in records
        if r.status in FINISHED and r.fingerprint and of_the_sweep(r)
    }
    if written_fingerprints is None:
        raise TypeError(
            "summarise needs the fingerprints whose manifest exists. Defaulting to the "
            "job list is what reported 1,100 of 1,099 architectures measured for days: "
            "210 jobs were marked Completed at the instant they were created and never "
            "ran. There is no safe default here, so there is not one."
        )
    written = set(written_fingerprints)
    measured = finished & written
    # Loud on purpose: a job that reported success and produced nothing is not a slow row,
    # it is an absent one, and the campaign will finish without noticing unless this is
    # visible next to the total.
    silent = sorted(finished - written)

    node_hours = 0.0
    resumes = 0
    for manifest in manifests:
        node_hours += float(manifest.get("wall_clock_seconds") or 0) / 3600
        resumes += int((manifest.get("cost") or {}).get("resumes") or 0)

    failures: Counter = Counter()
    for record in records:
        if record.status in LOST:
            failures[record.status] += 1

    starts = [r.created for r in records if r.created]
    return Progress(
        planned=planned,
        completed=completed,
        in_flight=in_flight,
        waiting=waiting,
        lost=lost,
        distinct_measured=len(measured),
        completed_without_result=silent,
        node_hours=node_hours,
        failures=failures,
        resumes=resumes,
        started=min(starts) if starts else None,
        now=now,
        concurrency=concurrency,
    )


def collect(planned: int, prefix: str = "nas-", concurrency: int = 5,
            corpus: str = "wikitext103", seed: int = 0,
            token_budget: int = 100_000_000) -> Progress:
    """Read the workspace and the artifact container. Submits nothing, cancels nothing."""
    from .data import _blob_service
    from .workspace import ml_client, resolve_platform

    platform = resolve_platform()
    client = ml_client(platform)

    records: list[RunRecord] = []
    for job in client.jobs.list():
        name = getattr(job, "name", "") or ""
        if not name.startswith(prefix):
            continue
        tags = getattr(job, "tags", None) or {}
        properties = getattr(job, "properties", None) or {}
        records.append(RunRecord(
            name=name,
            fingerprint=tags.get("fingerprint", ""),
            status=getattr(job, "status", "") or "",
            created=_parse(properties.get("StartTimeUtc")),
            completed=_parse(properties.get("EndTimeUtc")),
            wave=tags.get("wave", ""),
            role=tags.get("canary_role", ""),
            corpus=tags.get("corpus", ""),
            seed=tags.get("seed", ""),
            token_budget=tags.get("token_budget", ""),
        ))

    manifests: list[dict] = []
    written: set[str] = set()
    container = _blob_service(platform).get_container_client(platform.artifact_container)
    for blob in container.list_blobs(name_starts_with="runs/"):
        if not blob.name.endswith("manifest.json"):
            continue
        try:
            payload = container.get_blob_client(blob.name).download_blob().readall()
            manifests.append(json.loads(payload))
            # The fingerprint is the path's second segment. This is the only place it
            # exists — the manifest's own body does not carry it.
            written.add(blob.name.split("/")[1])
        except Exception:  # noqa: BLE001
            # A manifest being written as this reads is normal during a wave, and skipping
            # it is right: the report is a snapshot, not a transaction.
            continue

    return summarise(records, planned=planned, manifests=manifests,
                     written_fingerprints=written,
                     of_configuration=(corpus, seed, token_budget),
                     concurrency=concurrency)
