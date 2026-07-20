"""
Agent configuration — model routing and cost control.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for model routing and cost control."""

    # Planning/classification: use cheap model
    planning_model: str = "claude-haiku-4-5-20251001"

    # Extraction and drafting: use capable model
    extraction_model: str = "claude-sonnet-4-5-20250929"

    # Verification: use capable model (accuracy matters here)
    verification_model: str = "claude-sonnet-4-5-20250929"

    # Cost ceiling in USD
    max_cost_usd: float = 2.00

    # Max tool-call iterations per email
    max_iterations_per_email: int = 10

    # Concurrent email workers (emails are independent; LLM I/O is the bottleneck)
    max_workers: int = 6

    # Deterministic decisions: temperature 0 reduces run-to-run variance on
    # classification/verification (still not fully deterministic, but tighter).
    temperature: float = 0.0
