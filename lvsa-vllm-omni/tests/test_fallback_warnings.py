"""Tests for the _fallback.warn_fallback helper."""

import os

import pytest


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset dedup cache and default env between tests."""
    # Force warnings on by default for testing
    monkeypatch.setenv("LVSA_WARN_FALLBACK", "1")
    # Re-import to pick up env change (_FALLBACK_WARN is module-level)
    import importlib

    import lvsa_vllm_omni._fallback as fallback
    importlib.reload(fallback)
    fallback.reset_warnings()
    yield fallback
    fallback.reset_warnings()


def test_warn_fallback_prints(reset_state, capsys):
    """Basic warning is printed."""
    from lvsa_vllm_omni._fallback import warn_fallback
    warn_fallback("origin_a", "reason_x", 100, {"k": "v"})
    captured = capsys.readouterr()
    assert "[LVSA-FALLBACK]" in captured.out
    assert "origin_a" in captured.out
    assert "reason_x" in captured.out
    assert "seq_len=100" in captured.out
    assert "k=v" in captured.out


def test_warn_fallback_deduplicates(reset_state, capsys):
    """Same (origin, reason, seq_len) warns once."""
    from lvsa_vllm_omni._fallback import warn_fallback
    warn_fallback("origin_a", "reason_x", 100)
    warn_fallback("origin_a", "reason_x", 100)
    warn_fallback("origin_a", "reason_x", 100)
    captured = capsys.readouterr()
    # Only one occurrence of the origin
    assert captured.out.count("origin_a") == 1


def test_warn_fallback_different_seq_len(reset_state, capsys):
    """Different seq_len produces a separate warning."""
    from lvsa_vllm_omni._fallback import warn_fallback
    warn_fallback("origin_a", "reason_x", 100)
    warn_fallback("origin_a", "reason_x", 200)
    captured = capsys.readouterr()
    assert captured.out.count("origin_a") == 2


def test_warn_fallback_different_reason(reset_state, capsys):
    """Different reason produces a separate warning."""
    from lvsa_vllm_omni._fallback import warn_fallback
    warn_fallback("origin_a", "reason_x", 100)
    warn_fallback("origin_a", "reason_y", 100)
    captured = capsys.readouterr()
    assert "reason_x" in captured.out
    assert "reason_y" in captured.out


def test_warn_fallback_disabled(monkeypatch, capsys):
    """LVSA_WARN_FALLBACK=0 silences all warnings."""
    monkeypatch.setenv("LVSA_WARN_FALLBACK", "0")
    import importlib
    import lvsa_vllm_omni._fallback as fallback
    importlib.reload(fallback)
    fallback.warn_fallback("origin_a", "reason_x", 100)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_reset_warnings(reset_state, capsys):
    """After reset, the same warning fires again."""
    from lvsa_vllm_omni._fallback import warn_fallback, reset_warnings
    warn_fallback("origin_a", "reason_x", 100)
    reset_warnings()
    warn_fallback("origin_a", "reason_x", 100)
    captured = capsys.readouterr()
    assert captured.out.count("origin_a") == 2
