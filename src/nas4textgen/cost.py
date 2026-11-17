"""What the campaign cost, from what Azure allocated rather than from what it computed.

Every cost figure this project reported for weeks was derived from `wall_clock_seconds` in
the run manifests — the time each training process ran — multiplied by the published spot
rate. That is not what a cloud bills. Azure allocates a node before the process starts and
holds it after: pulling a six-gigabyte image, mounting a half-gigabyte corpus, and the
scale-down window that keeps a node for five minutes after its last job. Across roughly
3,500 short jobs the difference is **forty per cent**, and every quote this project made was
low by it.

**Azure's own billing could not settle it.** Cost Management returns zero cost *and* zero
usage quantity for this subscription — it is sponsored, and the consumption data lives at
the billing-account scope rather than the subscription's. So the independent measurement
comes from Azure Monitor instead, which emits Active Nodes and Idle Nodes per cluster: the
same source the idle-node alarms read, and a direct measure of allocation rather than of
computation.

**This stops being answerable in ninety-three days.** That is the platform-metric retention
window, and the campaign began in mid-August. Which is why the figure is written to an
artifact the moment it is measured rather than recomputed when the paper needs it.

**Two rates, and only one was ever used.** Low-priority nodes at 0.1260 USD/hour did the
whole campaign; the dedicated cluster at 0.6310 was provisioned and recorded zero hours,
because the hardware measurement it exists for needs a device with power instrumentation
that this subscription does not have.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

#: USD per node-hour, by cluster tier. Published prices for Standard_NC4as_T4_v3 in
#: southcentralus at the time of the campaign, recorded here because a price that moves
#: later must not silently restate what this campaign cost.
SPOT_RATE = 0.1260
DEDICATED_RATE = 0.6310

#: How long Azure keeps platform metrics. After this the allocation figure cannot be
#: recovered from anywhere, which is the argument for writing it down now.
METRIC_RETENTION_DAYS = 93


class MetricsUnavailable(RuntimeError):
    """Raised when allocation cannot be measured, rather than estimated instead."""


@dataclass(frozen=True)
class ClusterCost:
    """One cluster's allocation over the window."""

    name: str
    active_node_hours: float
    idle_node_hours: float
    rate: float

    @property
    def allocated_node_hours(self) -> float:
        """What is billed: a node held is a node paid for, busy or not."""
        return self.active_node_hours + self.idle_node_hours

    @property
    def usd(self) -> float:
        return self.allocated_node_hours * self.rate

    @property
    def idle_share(self) -> float:
        return self.idle_node_hours / self.active_node_hours if self.active_node_hours else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "cluster": self.name,
            "active_node_hours": round(self.active_node_hours, 2),
            "idle_node_hours": round(self.idle_node_hours, 2),
            "allocated_node_hours": round(self.allocated_node_hours, 2),
            "usd_per_node_hour": self.rate,
            "usd": round(self.usd, 2),
            "idle_share_of_active": round(self.idle_share, 4),
        }


@dataclass
class CostReport:
    """Allocation against computation, and the gap between them."""

    clusters: list[ClusterCost]
    #: Node-hours derived from the run manifests — process time, not allocation.
    process_node_hours: float
    #: Runs that consumed node time in this window, not architectures of the sweep.
    #:
    #: The distinction cost a wrong number. The progress artifact's `measured` counts
    #: architectures of the sweep's own configuration — deliberately, since it is compared
    #: against the size of the space — and the window also contains the 4x-budget study and
    #: the multi-corpus transfer. Dividing the whole campaign's allocation by the census
    #: alone charged every transfer run to it and reported 3.23 node-hours per architecture
    #: against a true 1.45, inflating the forecast by the same factor.
    runs: int
    window: tuple[str, str]

    @property
    def allocated(self) -> float:
        return sum(c.allocated_node_hours for c in self.clusters)

    @property
    def active(self) -> float:
        return sum(c.active_node_hours for c in self.clusters)

    @property
    def usd(self) -> float:
        return sum(c.usd for c in self.clusters)

    @property
    def overhead_active(self) -> float:
        """How much more node time was active than the processes accounted for."""
        return self.active / self.process_node_hours - 1 if self.process_node_hours else 0.0

    @property
    def overhead_allocated(self) -> float:
        return self.allocated / self.process_node_hours - 1 if self.process_node_hours else 0.0

    @property
    def per_architecture_allocated(self) -> float:
        """The figure to quote a future campaign with, rather than the process-time one."""
        return self.allocated / self.runs if self.runs else 0.0

    def forecast(self, runs: int) -> float:
        """USD for further runs, costed on allocation rather than on process time."""
        return runs * self.per_architecture_allocated * SPOT_RATE

    def as_dict(self) -> dict[str, object]:
        return {
            "window": {"from": self.window[0], "to": self.window[1]},
            "clusters": [c.as_dict() for c in self.clusters],
            "allocated_node_hours": round(self.allocated, 2),
            "process_node_hours": round(self.process_node_hours, 2),
            "usd": round(self.usd, 2),
            "runs_in_window": self.runs,
            "allocation_overhead_active": round(self.overhead_active, 4),
            "allocation_overhead_allocated": round(self.overhead_allocated, 4),
            "node_hours_per_run": {
                "process_time": round(
                    self.process_node_hours / self.runs, 4
                ) if self.runs else None,
                "allocated": round(self.per_architecture_allocated, 4),
                "counted_over": (
                    "every run that consumed node time in the window — the census, the "
                    "4x-budget study and the transfer — not the census alone. Dividing by "
                    "the census charges the whole campaign to it."
                ),
            },
            "why_allocation_not_process_time": (
                "Azure bills a node from allocation to release, which includes pulling the "
                "training image, mounting the corpus, and the scale-down window. None of "
                "that appears in wall_clock_seconds, and across thousands of short jobs it "
                "is a forty per cent difference. Every quote this project made before this "
                "measurement was low by it."
            ),
            "why_not_from_the_bill": (
                "Cost Management returns zero cost and zero usage for this subscription. It "
                "is sponsored and the consumption data sits at the billing-account scope, so "
                "no billed figure was available to reconcile against."
            ),
            "measured_until": (
                f"Platform metrics are retained {METRIC_RETENTION_DAYS} days, after which "
                "this cannot be recomputed from any source. That is why it is an artifact."
            ),
        }

    def render(self) -> str:
        lines = [
            f"  window {self.window[0]} to {self.window[1]}",
            "",
            f"  {'quantity':<38}{'node-hours':>12}{'USD':>10}",
            "  " + "-" * 60,
            f"  {'training process time (manifests)':<38}"
            f"{self.process_node_hours:>12,.0f}{self.process_node_hours * SPOT_RATE:>10,.0f}",
        ]
        for cluster in self.clusters:
            if cluster.allocated_node_hours == 0:
                continue
            lines += [
                f"  {cluster.name + ' active':<38}"
                f"{cluster.active_node_hours:>12,.0f}"
                f"{cluster.active_node_hours * cluster.rate:>10,.0f}",
                f"  {cluster.name + ' idle before release':<38}"
                f"{cluster.idle_node_hours:>12,.0f}"
                f"{cluster.idle_node_hours * cluster.rate:>10,.0f}",
            ]
        lines += [
            f"  {'ALLOCATED — what is billed':<38}{self.allocated:>12,.0f}{self.usd:>10,.0f}",
            "",
            f"  allocation overhead over process time  {self.overhead_active:>+7.1%} active, "
            f"{self.overhead_allocated:+.1%} with idle",
            f"  per run                                "
            f"{self.per_architecture_allocated:.2f} allocated node-hours",
        ]
        unused = [c.name for c in self.clusters if c.allocated_node_hours == 0]
        if unused:
            lines.append(
                f"  provisioned and never used             {', '.join(unused)}"
            )
        return "\n".join(lines)


def allocated_hours(
    resource: str, start: datetime, end: datetime, interval: str = "PT1H"
) -> dict[str, dict[str, float]]:
    """Node-hours per cluster, integrated from Azure Monitor.

    The window is walked in weekly slices because the metrics API caps points per call, and
    a truncated answer here would understate the total silently — which is the failure mode
    this whole measurement exists to correct.

    The maximum within each hour is used rather than the average: a node held for part of an
    hour is billed for that part, and averaging would discount it.
    """
    series: dict[str, dict[str, float]] = {}
    for metric in ("Active Nodes", "Idle Nodes"):
        cursor = start
        while cursor < end:
            chunk = min(cursor + timedelta(days=7), end)
            result = subprocess.run(
                ["az", "monitor", "metrics", "list", "--resource", resource,
                 "--metric", metric, "--dimension", "ClusterName",
                 "--aggregation", "Maximum", "--interval", interval,
                 "--start-time", cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "--end-time", chunk.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "-o", "json"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise MetricsUnavailable(
                    f"{metric} from {cursor.date()} to {chunk.date()}: "
                    f"{result.stderr.strip()[:160]}\n"
                    "Reported rather than estimated: an estimate here would be the "
                    "process-time figure this measurement exists to replace."
                )
            for entry in json.loads(result.stdout or "{}").get("value", []):
                for item in entry.get("timeseries", []):
                    cluster = next(
                        (m["value"] for m in item.get("metadatavalues", [])
                         if m["name"]["value"].lower() == "clustername"),
                        "unknown",
                    )
                    total = sum(
                        float(p["maximum"]) for p in item.get("data", [])
                        if p.get("maximum") is not None
                    )
                    bucket = series.setdefault(cluster, {"Active Nodes": 0.0,
                                                         "Idle Nodes": 0.0})
                    bucket[metric] += total
            cursor = chunk
    return series


def build_report(
    resource: str,
    start: datetime,
    end: datetime,
    process_node_hours: float,
    runs: int,
) -> CostReport:
    """Measure allocation and set it beside the process time already recorded."""
    measured = allocated_hours(resource, start.astimezone(UTC), end.astimezone(UTC))
    clusters = [
        ClusterCost(
            name=name,
            active_node_hours=values["Active Nodes"],
            idle_node_hours=values["Idle Nodes"],
            rate=DEDICATED_RATE if "dedicated" in name else SPOT_RATE,
        )
        for name, values in sorted(measured.items())
    ]
    return CostReport(
        clusters=clusters,
        process_node_hours=process_node_hours,
        runs=runs,
        window=(start.date().isoformat(), end.date().isoformat()),
    )
