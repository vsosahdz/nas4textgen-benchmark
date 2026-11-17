"""What the campaign cost, and the two ways of getting it wrong.

Both were made here. The figure reported for weeks came from process time, which a cloud
does not bill; then the command that fixed it divided the whole campaign's allocation by the
census alone and inflated the per-run cost by a factor of two.
"""

from __future__ import annotations

import pytest

from nas4textgen.cost import (
    DEDICATED_RATE,
    METRIC_RETENTION_DAYS,
    SPOT_RATE,
    ClusterCost,
    CostReport,
)


def _report(process: float = 2307.0, runs: int = 2553,
            active: float = 3095.0, idle: float = 451.0) -> CostReport:
    return CostReport(
        clusters=[
            ClusterCost("sweep-spot", active, idle, SPOT_RATE),
            ClusterCost("measure-dedicated", 0.0, 0.0, DEDICATED_RATE),
        ],
        process_node_hours=process,
        runs=runs,
        window=("2026-08-13", "2026-09-20"),
    )


class TestAllocationIsWhatIsBilled:
    def test_idle_node_time_counts(self):
        """A node held after its last job is a node paid for. The scale-down window is
        five minutes and there are thousands of jobs."""
        report = _report()
        assert report.allocated == pytest.approx(3546.0)
        assert report.usd == pytest.approx(3546.0 * SPOT_RATE)

    def test_the_overhead_over_process_time_is_reported_both_ways(self):
        """Active alone understates it; a reader deciding between many short jobs and few
        long ones needs the figure that includes the window."""
        report = _report()
        assert report.overhead_active == pytest.approx(3095 / 2307 - 1)
        assert report.overhead_allocated > report.overhead_active

    def test_a_cluster_that_never_ran_costs_nothing_and_is_still_listed(self):
        """Provisioned at five times the rate and recorded zero hours, because the
        measurement it exists for needs a device this subscription lacks. Silence would
        read as though it had been used."""
        report = _report()
        dedicated = next(c for c in report.clusters if c.name == "measure-dedicated")
        assert dedicated.usd == 0.0
        assert dedicated.name in [c["cluster"] for c in report.as_dict()["clusters"]]


class TestTheDenominatorIsRuns:
    def test_per_run_cost_counts_every_run_that_consumed_nodes(self):
        """Dividing by the census charged the 4x study and the transfer to it: 3.23
        node-hours per architecture against a true 1.39."""
        over_all_runs = _report(runs=2553).per_architecture_allocated
        over_census_only = _report(runs=1099).per_architecture_allocated
        assert over_all_runs == pytest.approx(3546 / 2553)
        assert over_census_only > 2 * over_all_runs, (
            "the wrong denominator more than doubles the figure, which is why it is named "
            "`runs` rather than `architectures`"
        )

    def test_the_forecast_uses_allocation_not_process_time(self):
        """Every quote this project made before the measurement was low by the overhead."""
        report = _report()
        on_allocation = report.forecast(886)
        on_process = 886 * (report.process_node_hours / report.runs) * SPOT_RATE
        assert on_allocation > on_process
        assert on_allocation / on_process == pytest.approx(
            report.allocated / report.process_node_hours, rel=1e-6
        )


class TestTheArtifactCarriesItsOwnExpiry:
    def test_the_retention_window_is_stated(self):
        """After ninety-three days the allocation figure cannot be recomputed from any
        source, which is the argument for an artifact rather than a calculation."""
        payload = _report().as_dict()
        assert str(METRIC_RETENTION_DAYS) in payload["measured_until"]

    def test_the_absent_bill_is_recorded_rather_than_implied(self):
        payload = _report().as_dict()
        assert "zero cost and zero usage" in payload["why_not_from_the_bill"]

    def test_the_rates_travel_with_the_figure(self):
        """A published price that moves later must not silently restate what this cost."""
        payload = _report().as_dict()
        assert payload["clusters"][0]["usd_per_node_hour"] == SPOT_RATE
