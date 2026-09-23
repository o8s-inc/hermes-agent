"""A fallback onto a smaller window must not wait out the compress or hygiene ceiling.

When the summary would grow, or the one allowed attempt still does not fit, the turn
ends immediately with /new guidance and ``compression_exhausted`` (the gateway
auto-reset contract). A shrink that fits the fallback window still commits.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.compression_facade import _CommitFenceRegistration, _run_under_progress_timeout
from agent.conversation_compression import CompressionCommitFence
from agent.fallback_window_failfast import (
    FAILFAST_WAIT_SECONDS, apply_failfast_budget, compression_cannot_shrink,
    prepare_fallback_compress, protected_floor_exceeds_window, tighten_hygiene_wait,
)
from agent.turn_context import PreflightCompressionTimedOut
from agent.turn_context_compaction import CompactionOutcome, _run_preflight_passes
from agent.turn_overflow import _recover_context_length
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression
from agent.turn_retry_state import TurnRetryState


def _compressor(*, context_length, would_grow=False):
    return SimpleNamespace(
        context_length=context_length,
        threshold_tokens=1_000,
        should_compress=lambda _tokens: True,
        get_active_compression_failure_cooldown=lambda: None,
        _last_compress_refused_would_grow=would_grow,
        _ineffective_compression_count=0,
    )


def _agent(compressor, *, fallback=True):
    agent = SimpleNamespace(
        compression_enabled=True,
        context_compressor=compressor,
        model="qwen-code",
        provider="custom",
        base_url="http://litellm.internal",
        max_tokens=0,
        tools=None,
        session_id="s1",
        log_prefix="",
        max_compression_attempts=3,
        _provider_fallback_active=fallback,
        _fallback_activated=fallback,
        _fallback_over_window_compress_attempted=False,
        _compression_feasibility_checked=True,
        _compression_skipped_due_to_lock=None,
        _compression_blocked_transient=None,
        _emit_status=MagicMock(),
        _persist_session=MagicMock(),
        _flush_status_buffer=MagicMock(),
        _buffer_diagnostic_status=MagicMock(),
        _buffer_vprint=MagicMock(),
        _vprint=MagicMock(),
        _api_call_count=1,
        iteration_budget=SimpleNamespace(refund=MagicMock()),
    )
    return agent


def _verdict(messages):
    return PreflightGateVerdict(
        action="", pending_moa_prepared_request=None, messages=messages,
        active_system_prompt="sys", conversation_history=list(messages), api_call_count=1,
        compression_attempts=0, final_response=None, failed=False, _turn_exit_reason=None,
        _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False, _last_preflight_pressure=None,
    )


def _run_preflight(agent, messages, pressure, compress):
    agent._compress_context = compress
    return run_preflight_compression(
        agent, _verdict(messages), compressor=agent.context_compressor,
        request_pressure_tokens=pressure, provider_overflow_preflight=False,
        defer_preflight=lambda _tokens: False, moa_prepared_request=None, system_message="sys",
        user_message="hi", max_compression_attempts=3, effective_task_id="t",
    )


_LONG = [
    {"role": "user", "content": "keep"},
    {"role": "assistant", "content": "this thread is already past the fallback window"},
]


def test_would_grow_preflight_failfast_does_not_call_compress():
    agent = _agent(_compressor(context_length=8_192, would_grow=True))
    calls = []

    def compress(*_args, **_kwargs):
        calls.append(time.monotonic())
        time.sleep(30)
        return _LONG, "sys"

    started = time.monotonic()
    verdict = _run_preflight(agent, _LONG, 50_000, compress)
    elapsed = time.monotonic() - started

    assert calls == []
    assert elapsed < 2
    assert verdict.action == "return"
    assert "/new" in verdict.result["final_response"]
    assert verdict.result["compression_exhausted"] is True
    assert verdict.result["failed"] is True
    assert verdict.result["turn_exit_reason"] == "fallback_context_failfast"


def test_second_over_window_attempt_in_the_same_turn_does_not_compress_again():
    agent = _agent(_compressor(context_length=8_192))
    calls = []

    def compress(messages, _system, **kwargs):
        calls.append(kwargs.get("failfast_budget_seconds"))
        agent.context_compressor._last_compress_refused_would_grow = True
        return messages, "sys"

    first = _run_preflight(agent, _LONG, 50_000, compress)
    second = _run_preflight(agent, _LONG, 50_000, compress)

    assert calls == [FAILFAST_WAIT_SECONDS]
    assert "/new" in first.result["final_response"]
    assert "/new" in second.result["final_response"]
    assert second.result["compression_exhausted"] is True


def test_shrink_that_fits_the_fallback_window_continues():
    agent = _agent(_compressor(context_length=8_192))
    seen = {}

    def compress(_messages, _system, **kwargs):
        seen.update(kwargs)
        return [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}], "sys"

    verdict = _run_preflight(agent, _LONG, 50_000, compress)

    assert seen["failfast_budget_seconds"] == FAILFAST_WAIT_SECONDS
    assert verdict.action == "continue"
    assert verdict.result is None


def test_lock_skip_on_an_over_window_fallback_is_not_exhaustion():
    agent = _agent(_compressor(context_length=8_192))
    agent._compression_skipped_due_to_lock = "other-path"

    def compress(messages, _system, **_kwargs):
        return messages, "sys"

    verdict = _run_preflight(agent, _LONG, 50_000, compress)

    assert verdict.action == "fallthrough"
    assert verdict.result is None
    assert agent._fallback_over_window_compress_attempted is False


def test_turn_start_would_grow_raises_without_compressing():
    agent = _agent(_compressor(context_length=8_192, would_grow=True))
    called = []
    agent._compress_context = lambda *_a, **_k: called.append(True)
    out = CompactionOutcome(
        messages=list(_LONG), active_system_prompt="sys", conversation_history=None,
        current_turn_user_idx=0,
    )
    started = time.monotonic()
    with pytest.raises(PreflightCompressionTimedOut, match="/new"):
        _run_preflight_passes(agent, out, agent.context_compressor, 50_000, "sys", "t")
    assert called == []
    assert time.monotonic() - started < 2


def test_protected_floor_that_fills_the_window_skips_the_summary():
    def bounds(_messages):
        return 0, 0

    compressor = _compressor(context_length=100)
    compressor._compress_window = bounds
    messages = [
        {"role": "user", "content": "x" * 800},
        {"role": "assistant", "content": "y" * 800},
    ]
    assert protected_floor_exceeds_window(compressor, messages) is True
    agent = _agent(compressor)
    message, budget = prepare_fallback_compress(agent, messages, 5_000)
    assert budget is None
    assert message is not None and "/new" in message
    assert compression_cannot_shrink(agent, messages) is True


def test_apply_failfast_budget_collapses_the_host_wait_and_skips_the_stall_retry():
    idle, ceiling, stall = apply_failfast_budget(300.0, 600.0, FAILFAST_WAIT_SECONDS)
    assert idle == FAILFAST_WAIT_SECONDS
    assert ceiling == FAILFAST_WAIT_SECONDS
    assert stall is False

    idle, ceiling, stall = apply_failfast_budget(300.0, 600.0, None)
    assert (idle, ceiling, stall) == (300.0, 600.0, True)


def test_tighten_hygiene_wait_caps_only_when_already_past_the_window():
    assert tighten_hygiene_wait(30.0, 600.0, 10.0, approx_tokens=1_000, context_length=8_000) == (
        30.0, 600.0, 10.0,
    )
    timeout, ceiling, hold = tighten_hygiene_wait(
        30.0, 600.0, 60.0, approx_tokens=20_000, context_length=8_000,
    )
    assert ceiling == FAILFAST_WAIT_SECONDS
    assert timeout <= FAILFAST_WAIT_SECONDS
    assert hold <= FAILFAST_WAIT_SECONDS


def test_failfast_host_wait_returns_in_seconds_and_does_not_start_a_stall_retry():
    messages = [{"role": "user", "content": "x"}]
    calls = []

    def run(fence=None, target_messages=None, same_turn_fallback_recovery=False):
        calls.append(same_turn_fallback_recovery)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and fence is not None and not (
            fence.is_cancelled or fence.deadline_exceeded
        ):
            fence.touch_progress()
            time.sleep(0.05)
        return target_messages, ""

    agent = _agent(_compressor(context_length=100))
    fence = CompressionCommitFence()
    started = time.monotonic()
    result = _run_under_progress_timeout(
        agent, run, messages, "sys",
        active_fence=fence, registration=_CommitFenceRegistration(fence),
        fence_registration_lock=threading.RLock(),
        idle_timeout=30.0, total_ceiling=60.0, approx_tokens=10_000,
        failfast_budget_seconds=0.4,
    )
    elapsed = time.monotonic() - started

    assert result[0] is messages
    assert elapsed < 2
    assert calls == [False]


def test_overflow_would_grow_failfast_does_not_sleep_before_retry(monkeypatch):
    compressor = _compressor(context_length=8_192, would_grow=True)
    agent = _agent(compressor)
    messages = [
        {"role": "user", "content": "u" * 40_000},
        {"role": "assistant", "content": "a" * 40_000},
    ]
    slept = []
    called = []
    monkeypatch.setattr("agent.turn_overflow.time.sleep", lambda seconds: slept.append(seconds))

    def compress(*_args, **_kwargs):
        called.append(True)
        time.sleep(30)
        return messages, "sys"

    agent._compress_context = compress
    from agent.turn_overflow import _Recovery

    st = _Recovery(
        agent=agent, api_messages=messages, system_message="sys", effective_task_id="t",
        api_call_count=1, max_compression_attempts=3, messages=list(messages),
        active_system_prompt="sys", conversation_history=list(messages), approx_tokens=50_000,
        compression_attempts=0,
    )
    verdict = _recover_context_length(st, TurnRetryState(), "context length exceeded")

    assert called == []
    assert slept == []
    assert verdict.action == "return"
    assert "/new" in verdict.result["final_response"]
    assert verdict.result["compression_exhausted"] is True
