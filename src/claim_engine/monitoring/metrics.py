"""In-process metrics aggregation for real-time operational visibility.

Tracks the KPIs that matter for an automated adjudication system: throughput,
auto-resolution rate, decision mix, latency percentiles and edge-case volume.
In production these counters are mirrored to Prometheus / CloudWatch; here they
are also exposed as a plain snapshot for the demo and tests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from claim_engine.models.decision import ClaimDecision
from claim_engine.models.enums import DecisionOutcome


@dataclass
class MetricsSnapshot:
    """An immutable point-in-time view of the engine's KPIs."""

    total_claims: int
    outcome_counts: dict[str, int]
    auto_resolution_rate: float
    auto_approval_rate: float
    manual_review_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    total_edge_cases: int

    def as_dict(self) -> dict:
        """Return the snapshot as a plain dict (for JSON logging / dashboards)."""
        return {
            "total_claims": self.total_claims,
            "outcome_counts": self.outcome_counts,
            "auto_resolution_rate": round(self.auto_resolution_rate, 4),
            "auto_approval_rate": round(self.auto_approval_rate, 4),
            "manual_review_rate": round(self.manual_review_rate, 4),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "total_edge_cases": self.total_edge_cases,
        }


class MetricsCollector:
    """Thread-safe accumulator of decision metrics.

    A single instance is shared across the pipeline. All mutations are guarded
    by a lock so the collector is safe under concurrent Lambda invocations that
    share a warm container.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._outcome_counts: dict[str, int] = {outcome.value: 0 for outcome in DecisionOutcome}
        self._latencies_ms: list[float] = []
        self._edge_case_total = 0

    def record_decision(self, decision: ClaimDecision, edge_case_flags: list[str]) -> None:
        """Record a single completed decision and its edge-case flags.

        Args:
            decision: The finalised decision.
            edge_case_flags: Flags raised by the edge-case detector for the claim.
        """
        with self._lock:
            self._outcome_counts[decision.outcome.value] += 1
            self._latencies_ms.append(decision.trace.latency_ms)
            self._edge_case_total += len(edge_case_flags)

    def snapshot(self) -> MetricsSnapshot:
        """Compute and return a consistent snapshot of all current metrics."""
        with self._lock:
            total = sum(self._outcome_counts.values())
            approved = self._outcome_counts[DecisionOutcome.AUTO_APPROVED.value]
            denied = self._outcome_counts[DecisionOutcome.AUTO_DENIED.value]
            review = self._outcome_counts[DecisionOutcome.MANUAL_REVIEW.value]
            latencies = sorted(self._latencies_ms)

            return MetricsSnapshot(
                total_claims=total,
                outcome_counts=dict(self._outcome_counts),
                auto_resolution_rate=self._safe_ratio(approved + denied, total),
                auto_approval_rate=self._safe_ratio(approved, total),
                manual_review_rate=self._safe_ratio(review, total),
                average_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
                p95_latency_ms=self._percentile(latencies, 95.0),
                total_edge_cases=self._edge_case_total,
            )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """Divide guarding against division by zero."""
        return (numerator / denominator) if denominator else 0.0

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        """Return the ``percentile`` (0-100) of an already-sorted list."""
        if not sorted_values:
            return 0.0
        rank = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        fraction = rank - lower_index
        return sorted_values[lower_index] + fraction * (
            sorted_values[upper_index] - sorted_values[lower_index]
        )
