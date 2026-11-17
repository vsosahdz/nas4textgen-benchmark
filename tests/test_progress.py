"""Progress over a multi-day campaign, computed from records rather than from hope.

Every case is constructed so the right answer is known: a fixed set of job records and
manifests, and an assertion about what the report should say. The interesting properties
are what it counts, what it refuses to estimate, and that it never touches anything.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nas4textgen.azure.progress import (
    MIN_ROWS_FOR_ESTIMATE,
    Progress,
    RunRecord,
    summarise,
)

START = datetime(2026, 8, 18, 0, 0, tzinfo=UTC)


def wrote(*records: RunRecord) -> set[str]:
    """Fingerprints whose manifest exists, for a campaign where every finished job left one.

    Stated per test rather than defaulted inside ``summarise``, because the default was the
    defect: 210 jobs of wave four were marked Completed at the instant they were created
    and never ran, and a counter that reads job status called every one of them a measured
    architecture.
    """
    return {r.fingerprint for r in records if r.status == "Completed"}


def record(fingerprint: str, status: str, hours: float = 0.0, name: str = "") -> RunRecord:
    return RunRecord(
        name=name or f"nas-{fingerprint}-a1",
        fingerprint=fingerprint,
        status=status,
        created=START + timedelta(hours=hours),
        completed=None,
    )


class TestCounting:
    def test_a_row_is_a_measurement_not_a_job(self):
        """A measurement retried three times is one row and three jobs. Counting jobs would
        make a campaign that is failing and retrying look busier than one succeeding."""
        records = [
            record("aaa", "Failed"),
            record("aaa", "Failed"),
            record("aaa", "Completed"),
            record("bbb", "Completed"),
        ]
        progress = summarise(records, planned=10, written_fingerprints=wrote(*records),
                             now=START + timedelta(hours=1))

        assert progress.distinct_measured == 2, "two architectures were measured"
        assert progress.completed == 2
        assert progress.lost == 2
        assert progress.remaining == 8

    def test_the_same_fingerprint_completing_twice_counts_once(self):
        """It happened: a member was re-measured for comparability, and the row is still
        one row."""
        records = [record("aaa", "Completed", name="nas-aaa-a3"),
                   record("aaa", "Completed", name="nas-aaa-a4")]
        progress = summarise(records, planned=4, written_fingerprints=wrote(*records),
                             now=START + timedelta(hours=1))
        assert progress.completed == 2
        assert progress.distinct_measured == 1

    def test_states_are_grouped_by_what_they_mean_for_the_campaign(self):
        records = [
            record("a", "Running"), record("b", "Preparing"), record("c", "Queued"),
            record("d", "Completed"), record("e", "Canceled"),
        ]
        progress = summarise(records, planned=5, written_fingerprints=wrote(*records),
                             now=START + timedelta(hours=1))
        assert progress.in_flight == 2
        assert progress.waiting == 1
        assert progress.lost == 1

    def test_node_hours_and_resumes_come_from_the_manifests(self):
        """Work recovered after a reclaim is worth reporting: it is the difference between
        checkpointing paying for itself and not."""
        manifests = [
            {"wall_clock_seconds": 3600, "cost": {"resumes": 0}},
            {"wall_clock_seconds": 7200, "cost": {"resumes": 2}},
        ]
        progress = summarise([record("a", "Completed")], planned=2,
                             manifests=manifests, written_fingerprints={"a"},
                             now=START + timedelta(hours=1))
        assert progress.node_hours == pytest.approx(3.0)
        assert progress.resumes == 2


class TestEstimate:
    def _many(self, done: int, planned: int, hours: float, nodes: int = 5) -> Progress:
        records = [record(f"f{i}", "Completed") for i in range(done)]
        # One node-hour per row, so the rate is exactly one row per node-hour and the
        # arithmetic in each test below is checkable by hand.
        manifests = [{"wall_clock_seconds": 3600, "cost": {}} for _ in range(done)]
        return summarise(records, planned=planned, manifests=manifests,
                         written_fingerprints=wrote(*records),
                         now=START + timedelta(hours=hours), concurrency=nodes)

    def test_too_few_rows_gives_no_date_and_says_why(self):
        """Rather than extrapolating from two points, which is how a thirteen-day campaign
        acquires a confident wrong finish date on its first morning."""
        remaining, why = self._many(3, 1099, hours=2).estimate()
        assert remaining is None
        assert str(MIN_ROWS_FOR_ESTIMATE) in why

    def test_the_rate_is_per_node_hour_not_per_wall_clock_hour(self):
        """The first version divided by elapsed wall clock, which spans the gaps between
        waves — the hours spent diagnosing a canary, or with nothing submitted at all. With
        the cluster busy ten per cent of the elapsed time it reported ninety-five days
        remaining for a sweep on track to finish in nine.
        """
        # Ten rows, ten node-hours, a hundred left, five nodes: one row per node-hour is
        # five rows an hour, so twenty hours.
        progress = self._many(10, planned=110, hours=500, nodes=5)
        remaining, why = progress.estimate()

        assert remaining is not None
        assert remaining.total_seconds() / 3600 == pytest.approx(20.0, rel=0.01)
        assert "rows/node-hour" in why

        # And the answer does not move when the campaign sat idle for a month.
        idle = self._many(10, planned=110, hours=5000, nodes=5)
        assert idle.estimate()[0] == remaining

    def test_more_nodes_move_the_finish_without_remeasuring(self):
        """The projection is a rate per node-hour times the concurrency, so a quota change
        is a different date rather than a new experiment."""
        five = self._many(10, planned=110, hours=100, nodes=5).estimate()[0]
        twenty = self._many(10, planned=110, hours=100, nodes=20).estimate()[0]
        assert twenty.total_seconds() == pytest.approx(five.total_seconds() / 4, rel=0.01)

    def test_occupancy_is_reported_because_it_is_why_wall_clock_misleads(self):
        progress = self._many(10, planned=110, hours=500, nodes=5)
        assert progress.as_dict()["cluster_occupancy"] < 0.01
        assert "occupancy" in progress.render()

    def test_a_finished_campaign_estimates_no_time_remaining(self):
        remaining, _ = self._many(20, planned=20, hours=50).estimate()
        assert remaining is not None
        assert remaining.total_seconds() == 0

    def test_the_report_states_its_own_basis(self):
        """A projected finish without the rate it came from is a number to be believed
        rather than checked."""
        payload = self._many(12, planned=100, hours=60).as_dict()
        assert payload["estimated_finish"] is not None
        assert "rows/node-hour" in payload["estimate_basis"]
        assert "nodes" in payload["estimate_basis"]


class TestRendering:
    def test_the_report_is_readable_without_the_json(self):
        progress = summarise(
            [record(f"f{i}", "Completed") for i in range(10)]
            + [record("x", "Running"), record("y", "Failed")],
            planned=100,
            manifests=[{"wall_clock_seconds": 3600, "cost": {"resumes": 1}}] * 10,
            written_fingerprints={f"f{i}" for i in range(10)},
            now=START + timedelta(hours=10),
        )
        text = progress.render()
        assert "10 of 100 rows measured" in text
        assert "resumes" in text
        assert "finishes" in text
        assert "lost jobs by cause" in text

    def test_an_empty_campaign_does_not_crash(self):
        progress = summarise([], planned=1099, written_fingerprints=set(), now=START)
        assert "0 of 1,099" in progress.render()
        assert progress.as_dict()["estimated_finish"] is None


def test_collecting_reads_and_never_writes():
    """Stated as a property of the code rather than a promise in a docstring.

    A report that could cancel a job would be a report nobody dares run during a wave,
    which defeats the point of having one.
    """
    import inspect

    from nas4textgen.azure import progress as module

    source = inspect.getsource(module)
    for mutation in ("create_or_update", "jobs.cancel", "delete_blob", "upload_blob",
                     "begin_delete", ".cancel("):
        assert mutation not in source, f"{mutation} would make this more than a report"


class TestAJobThatSucceededAndProducedNothing:
    """The defect that inflated this report for days.

    Two hundred and ten jobs of wave four were accepted, never scheduled, and marked
    Completed at the instant they were created — no container, no log, no output. A counter
    reading job status called each one a measured architecture, and reported 1,100 of 1,099.
    """

    def test_a_completed_job_without_a_manifest_is_not_a_measured_row(self):
        records = [record("has", "Completed"), record("empty", "Completed")]
        progress = summarise(records, planned=2, written_fingerprints={"has"}, now=START)

        assert progress.distinct_measured == 1
        assert progress.completed == 2, "both jobs did report success"
        assert progress.completed_without_result == ["empty"]

    def test_the_absence_is_reported_beside_the_total_not_folded_into_it(self):
        records = [record("has", "Completed"), record("empty", "Completed")]
        text = summarise(records, planned=2, written_fingerprints={"has"},
                         now=START).render()
        assert "no result" in text, "an absent row must be visible, not averaged away"

    def test_omitting_what_was_written_is_an_error_rather_than_a_default(self):
        """There is no safe default: the old one over-reported every wave."""
        with pytest.raises(TypeError, match="1,100 of 1,099"):
            summarise([record("a", "Completed")], planned=1, now=START)
