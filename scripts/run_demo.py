#!/usr/bin/env python3
"""End-to-end demonstration of the claim approval engine (runs in mock mode).

Usage:
    python scripts/run_demo.py

The script builds the full pipeline, adjudicates the three seeded sample claims
(a clean approval, a hard denial and an ambiguous edge case), prints each
decision with its audit trace, and finishes with a live metrics snapshot.

No external credentials are required: with mock mode enabled the engine uses
deterministic in-memory implementations of Claude, Pinecone and Snowflake.
"""

from __future__ import annotations

import os
import sys

# Make 'src/' importable when running the script directly from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Force mock mode for the demo regardless of any local .env file.
os.environ.setdefault("CLAIM_ENGINE_MOCK_MODE", "true")

from claim_engine.config.settings import get_settings  # noqa: E402
from claim_engine.logging_utils import configure_logging  # noqa: E402
from claim_engine.models.decision import ClaimDecision  # noqa: E402
from claim_engine.pipeline.claim_pipeline import build_pipeline  # noqa: E402

# The claim ids seeded by the in-memory data store (see data_store.py).
_DEMO_CLAIM_IDS = ["CLM-APPROVE-001", "CLM-DENY-001", "CLM-REVIEW-001"]


def _print_decision(decision: ClaimDecision) -> None:
    """Pretty-print a single decision and its audit trace to stdout."""
    print("\n" + "=" * 78)
    print(f"CLAIM        : {decision.claim_id}")
    print(f"OUTCOME      : {decision.outcome.value}")
    print(f"CONFIDENCE   : {decision.confidence:.2f}")
    print(f"APPROVED INR : {decision.approved_amount:,.2f}")
    print(f"HUMAN REVIEW : {decision.requires_human_review}")
    print(f"SUMMARY      : {decision.summary}")
    print(f"LATENCY (ms) : {decision.trace.latency_ms:.2f}")
    print(f"SOURCES USED : {', '.join(decision.trace.retrieved_sources) or 'none'}")

    failed_rules = [f for f in decision.trace.rule_findings if not f.passed]
    if failed_rules:
        print("FAILED RULES :")
        for finding in failed_rules:
            print(f"   - [{finding.severity.value}] {finding.rule_id}: {finding.message}")

    if decision.trace.llm_assessment:
        assessment = decision.trace.llm_assessment
        print(f"LLM RATIONALE: {assessment.rationale}")
        if assessment.edge_case_flags:
            print(f"EDGE FLAGS   : {', '.join(assessment.edge_case_flags)}")


def main() -> None:
    """Build the pipeline, process the demo claims and report metrics."""
    settings = get_settings()
    configure_logging(settings.log_level)

    print("Automated Insurance Claim Approval Engine - demo (mock mode)")
    print(f"Mock mode: {settings.mock_mode} | Environment: {settings.environment}")

    pipeline = build_pipeline(settings)

    for claim_id in _DEMO_CLAIM_IDS:
        decision = pipeline.process_claim_by_id(claim_id)
        _print_decision(decision)

    print("\n" + "=" * 78)
    print("LIVE METRICS SNAPSHOT")
    print("=" * 78)
    snapshot = pipeline.metrics.snapshot().as_dict()
    for key, value in snapshot.items():
        print(f"  {key:>22}: {value}")


if __name__ == "__main__":
    main()
