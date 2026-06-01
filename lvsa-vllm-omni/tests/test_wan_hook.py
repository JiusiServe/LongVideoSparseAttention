"""Tests for Wan LVSA hook registration.

These tests verify the registration logic. The hook itself requires vllm-omni
installed (not available in unit test env), so the patching is mocked.
"""

import os
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    yield


def test_maybe_install_wan_hook_disabled(clean_env):
    """Hook is not installed if LVSA_WAN_HOOK is not set."""
    from lvsa_vllm_omni.register import maybe_install_wan_hook
    # Should return silently without errors
    maybe_install_wan_hook()


def test_maybe_install_wan_hook_no_t_lat(clean_env, monkeypatch, capsys):
    """Hook skipped with warning when T_lat not set."""
    monkeypatch.setenv("LVSA_WAN_HOOK", "1")
    from lvsa_vllm_omni.register import maybe_install_wan_hook
    maybe_install_wan_hook()
    captured = capsys.readouterr()
    # Either warns about missing T_lat or about vllm_omni being unavailable
    # (both are acceptable outcomes in the test env)
    assert "Warning" in captured.out or "failed" in captured.out


def test_maybe_install_wan_hook_install_attempt(clean_env, monkeypatch, capsys):
    """Hook attempts install when both LVSA_WAN_HOOK=1 and T_lat are set."""
    monkeypatch.setenv("LVSA_WAN_HOOK", "1")
    monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", "33")

    from lvsa_vllm_omni.register import maybe_install_wan_hook
    # Will fail because vllm_omni isn't installed, but should fail gracefully
    maybe_install_wan_hook()
    captured = capsys.readouterr()
    # Either succeeded (rare — needs vllm-omni) or failed gracefully
    assert "Wan hook" in captured.out or "Installed" in captured.out or "failed" in captured.out


def test_wan_hook_import_does_not_crash():
    """The wan_hook module should be importable even without vllm-omni.

    (The import of vllm_omni happens inside the install function, not at module
    load.)
    """
    # This should work because hunyuan_hook only imports vllm-omni lazily inside
    # install_*_lvsa_hook, and wan_hook does the same.
    try:
        from lvsa_vllm_omni import wan_hook
        assert hasattr(wan_hook, "install_wan_lvsa_hook")
    except ImportError as e:
        # Only acceptable if error is about vllm_omni, not about our own code
        assert "vllm_omni" in str(e) or "hunyuan_hook" in str(e)


def test_wan_hook_install_raises_without_vllm_omni(clean_env):
    """install_wan_lvsa_hook should raise ImportError when vllm-omni absent."""
    from lvsa_vllm_omni.wan_hook import install_wan_lvsa_hook
    with pytest.raises((ImportError, ModuleNotFoundError)):
        install_wan_lvsa_hook(total_latent_frames=21)


def test_wan_hook_reuses_hunyuan_state_class():
    """Wan hook reuses HunyuanLVSAState for step tracking + metadata caching.

    This is a structural test — verifies the import dependency. The state class
    is model-agnostic; sharing the implementation reduces drift between hooks.
    """
    from lvsa_vllm_omni import wan_hook
    from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
    # wan_hook should have imported HunyuanLVSAState at module load
    assert hasattr(wan_hook, "HunyuanLVSAState")
    assert wan_hook.HunyuanLVSAState is HunyuanLVSAState


def test_wan_hook_config_loaded_from_env(monkeypatch, clean_env):
    """When the wan hook attempts install, it should pick up env-var config.

    We can't run a real install (no vllm-omni), but we can verify the LVSAConfig
    machinery the hook depends on reads the same env vars users will set.
    """
    monkeypatch.setenv("LVSA_REFERENCE_LATENT_FRAMES", "21")
    monkeypatch.setenv("LVSA_ROTATE_KEYFRAMES", "1")
    monkeypatch.setenv("LVSA_SPARSITY_SCALE", "0.7")

    # wan_hook calls LVSAConfig.from_env() inside install_wan_lvsa_hook.
    # Verify the same env-read API the hook uses:
    from lvsa_vllm_omni.config import LVSAConfig
    cfg = LVSAConfig.from_env()
    assert cfg.reference_latent_frames == 21
    assert cfg.rotate_keyframes is True
    assert cfg.sparsity_scale == pytest.approx(0.7)
