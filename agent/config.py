"""
Agent configuration — model routing and cost control.
"""

import os
from dataclasses import dataclass


def make_client():
    """
    Build an Anthropic client with generous retries + timeout, and force IPv4.

    Diagnostics on some hosts (e.g. Railway) showed DNS + a raw IPv4 TLS socket to
    api.anthropic.com succeed, yet the SDK's httpx client failed with APIConnectionError.
    Root cause: the host advertises a AAAA (IPv6) record, httpx tries IPv6 first, and the
    container's IPv6 egress is broken — so every request hangs/fails on the IPv6 attempt.
    We pin the httpx transport to IPv4 (local_address="0.0.0.0"), which resolves it.
    Set PBC_FORCE_IPV4=0 to disable.
    """
    from anthropic import Anthropic

    # Sanitize the key: dashboards/paste often inject stray whitespace or line breaks,
    # which makes it an illegal HTTP header value and surfaces as a confusing
    # APIConnectionError. Strip all internal whitespace defensively.
    raw_key = os.environ.get("ANTHROPIC_API_KEY", "")
    clean_key = "".join(raw_key.split())

    kwargs = dict(
        api_key=clean_key or None,
        max_retries=int(os.environ.get("ANTHROPIC_MAX_RETRIES", "5")),
        timeout=float(os.environ.get("ANTHROPIC_TIMEOUT", "60")),
    )

    if os.environ.get("PBC_FORCE_IPV4", "1") == "1":
        try:
            import httpx
            # Binding the source address to an IPv4 any-address forces IPv4 egress.
            transport = httpx.HTTPTransport(local_address="0.0.0.0", retries=2)
            kwargs["http_client"] = httpx.Client(
                transport=transport,
                timeout=kwargs["timeout"],
            )
        except Exception:
            pass  # fall back to default client if httpx internals change

    return Anthropic(**kwargs)


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

    # Self-consistency voting on the verifier (OPTIONAL, default off).
    # Measured on the sample set, majority voting at higher temperature did NOT
    # reduce variance on the genuinely-ambiguous boundary items (e.g. PBC-26) — it
    # just added ~30% cost and re-introduced noise via the diversity temperature.
    # Kept as a tunable knob, but default 1 (single deterministic shot at temp 0),
    # which is the most stable configuration we measured. Set >1 to re-enable.
    verifier_votes: int = 1
    verifier_vote_temperature: float = 0.4
