"""Fail fast when a fallback model cannot hold the current thread.

Primary or MoA billing failure switches onto a smaller-window model. A long
thread then enters compression; a summary that would grow is refused, and the
host used to keep waiting on the compress or hygiene ceiling (minutes) before
the session died. This module decides that outcome before that wait. A compress
that actually fits the fallback window still commits on the existing path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.conversation_compression import (
    context_compression_timed_out,
    request_exceeds_model_window,
)
from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)

# Seconds, not the 600s compress/hygiene ceiling. One allowed summary attempt on
# an over-window fallback; a refusal or timeout must return inside this budget.
FAILFAST_WAIT_SECONDS = 15.0


def fallback_route_active(agent: Any) -> bool:
    """True after ``try_activate_fallback`` has swapped off the primary route."""
    return (
        getattr(agent, "_provider_fallback_active", False) is True
        or getattr(agent, "_fallback_activated", False) is True
    )


def over_model_window(tokens: Any, window: Any) -> bool:
    """True when both sides are known and the request is past the window."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return False
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        return False
    return tokens > window


def over_fallback_window(agent: Any, request_tokens: Any) -> bool:
    """True when the active route is a fallback and the request exceeds its window."""
    if not fallback_route_active(agent):
        return False
    return request_exceeds_model_window(agent, request_tokens) is True


def _refresh_ineffective_count(compressor: Any) -> None:
    """Reload the durable strike count so a hygiene refusal is visible to the turn agent."""
    loader = getattr(type(compressor), "_load_ineffective_compression_count", None)
    if not callable(loader):
        return
    try:
        loader(compressor)
    except Exception:
        logger.debug("fallback fail-fast could not reload ineffective compression count", exc_info=True)


def _ineffective_strikes(compressor: Any) -> int:
    raw = getattr(compressor, "_ineffective_compression_count", 0)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    return raw


def protected_floor_exceeds_window(compressor: Any, messages: Any) -> bool:
    """True when the unsquashable head and tail already fill the window.

    Summarizing the middle cannot make that request fit, so the summary call
    would only burn the compress ceiling and then be refused.
    """
    window = getattr(compressor, "context_length", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return False
    if not isinstance(messages, list) or not messages:
        return False
    bounds = getattr(compressor, "_compress_window", None)
    if not callable(bounds):
        return False
    try:
        start, end = bounds(messages)
    except Exception:
        return False
    if isinstance(start, bool) or not isinstance(start, int):
        return False
    if isinstance(end, bool) or not isinstance(end, int):
        return False
    start = max(0, min(start, len(messages)))
    end = max(start, min(end, len(messages)))
    floor = estimate_messages_tokens_rough(messages[:start] + messages[end:])
    return floor >= window


def compression_cannot_shrink(agent: Any, messages: Any) -> bool:
    """True when another summary cannot make this transcript smaller.

    A would-grow refusal, a prior ineffective strike, or a protected head/tail
    that already fills the window. Reads the strike counter; does not write it.
    """
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return False
    _refresh_ineffective_count(compressor)
    if getattr(compressor, "_last_compress_refused_would_grow", False) is True:
        return True
    if _ineffective_strikes(compressor) >= 1:
        return True
    return protected_floor_exceeds_window(compressor, messages)


def failfast_message(agent: Any, request_tokens: Any, *, window: Any = None) -> str:
    """User-visible /new guidance. The gateway auto-reset notice is appended separately."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        compressor = getattr(agent, "context_compressor", None)
        window = getattr(compressor, "context_length", 0) or 0
        try:
            window = int(window)
        except (TypeError, ValueError):
            window = 0
    model = getattr(agent, "model", None) or "the fallback model"
    tokens = request_tokens if isinstance(request_tokens, int) and not isinstance(request_tokens, bool) else 0
    return (
        f"This conversation is too long for {model} "
        f"(~{tokens:,} tokens vs a {window:,}-token window) and compression "
        "cannot shrink it. Start a fresh session with /new."
    )


def prepare_fallback_compress(
    agent: Any, messages: Any, request_tokens: Any,
) -> tuple[Optional[str], Optional[float]]:
    """Decide the one allowed compress on an over-window fallback.

    Returns ``(fail_message, None)`` when compression must not start, or
    ``(None, budget_seconds)`` for the single attempt whose host wait is capped
    so a refusal or timeout returns in seconds. ``(None, None)`` leaves the
    existing compress path unchanged (not a fallback, or the request fits).
    """
    if not over_fallback_window(agent, request_tokens):
        return None, None
    if getattr(agent, "_fallback_over_window_compress_attempted", False) or compression_cannot_shrink(
        agent, messages
    ):
        message = failfast_message(agent, request_tokens)
        logger.warning("Fallback context fail-fast (no compress): %s", message)
        return message, None
    agent._fallback_over_window_compress_attempted = True
    return None, FAILFAST_WAIT_SECONDS


def fallback_compress_failed(agent: Any, request_tokens: Any) -> bool:
    """After that one attempt: refused, timed out, or still over the fallback window."""
    if not fallback_route_active(agent):
        return False
    compressor = getattr(agent, "context_compressor", None)
    if getattr(compressor, "_last_compress_refused_would_grow", False) is True:
        return True
    if context_compression_timed_out(agent) and over_fallback_window(agent, request_tokens):
        return True
    return over_fallback_window(agent, request_tokens)


def fallback_window_exhausted_result(
    agent: Any, messages: Any, conversation_history: Any, api_call_count: int, request_tokens: Any,
) -> dict:
    """Turn result the gateway already auto-resets on (``compression_exhausted``)."""
    from agent.conversation_loop import _partial_turn_result

    message = failfast_message(agent, request_tokens)
    logger.warning("Fallback context fail-fast: %s", message)
    persist = getattr(agent, "_persist_session", None)
    if callable(persist):
        try:
            persist(messages, conversation_history)
        except Exception:
            logger.debug("fallback fail-fast session persist failed", exc_info=True)
    flush = getattr(agent, "_flush_status_buffer", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            logger.debug("fallback fail-fast status flush failed", exc_info=True)
    return _partial_turn_result(
        message, messages, api_call_count,
        failed=True, compression_exhausted=True,
        turn_exit_reason="fallback_context_failfast",
        failure_reason="context_overflow", failure_retryable=False,
    )


def apply_failfast_budget(
    idle_timeout: float, total_ceiling: float, budget_seconds: Optional[float],
) -> tuple[float, float, bool]:
    """Cap a host compress wait. Returns ``(idle, ceiling, stall_fallback)``.

    ``budget_seconds is None`` keeps the caller's budgets and the stall retry.
    A positive budget collapses both sides and skips the extra LLM retry so a
    timeout cannot start a second multi-minute wait.
    """
    if budget_seconds is None or budget_seconds <= 0:
        return idle_timeout, total_ceiling, True
    budget = float(budget_seconds)
    idle = min(idle_timeout, budget) if idle_timeout > 0 else budget
    ceiling = min(total_ceiling, budget)
    if ceiling < idle:
        idle = ceiling
    return idle, ceiling, False


def tighten_hygiene_wait(
    timeout_seconds: float, total_ceiling_seconds: float, max_turn_hold_seconds: float, *,
    approx_tokens: Any, context_length: Any,
) -> tuple[float, float, float]:
    """Shorten a hygiene wait that is already past the model window.

    Under-window hygiene (the 85% safety net) keeps its configured ceiling.
    An over-window wait cannot be "continued uncompressed" usefully, so a
    trickle summary must not hold the turn out to ``hygiene_total_ceiling_seconds``.
    """
    if not over_model_window(approx_tokens, context_length):
        return timeout_seconds, total_ceiling_seconds, max_turn_hold_seconds
    ceiling = min(float(total_ceiling_seconds), FAILFAST_WAIT_SECONDS)
    timeout = min(float(timeout_seconds), ceiling)
    hold = min(float(max_turn_hold_seconds), FAILFAST_WAIT_SECONDS)
    return timeout, ceiling, hold
