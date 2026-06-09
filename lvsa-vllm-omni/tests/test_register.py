"""Tests for the plugin entry-point and the env-gated hook installers.

The actual ``register_lvsa_backend()`` requires vllm-omni installed, so we
verify:
  - Module imports cleanly without vllm-omni.
  - The gating functions (``maybe_install_*_hook``) handle missing env vars
    and vllm-omni absence gracefully.
  - The hooks are not installed at module-load time (lazy).
"""
from __future__ import annotations

import os
import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    yield


class TestImportSurface:
    def test_module_importable_without_vllm_omni(self):
        """register.py should import cleanly even if vllm-omni isn't installed."""
        from lvsa_vllm_omni import register
        assert hasattr(register, "register_lvsa_backend")
        assert hasattr(register, "maybe_install_hunyuan_hook")
        assert hasattr(register, "maybe_install_wan_hook")
        assert hasattr(register, "maybe_install_cosmos3_hook")

    def test_register_lvsa_backend_lazy_imports(self):
        """register_lvsa_backend must defer the vllm-omni import until called."""
        from lvsa_vllm_omni.register import register_lvsa_backend
        # Calling it without vllm-omni installed should fail with ImportError
        with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
            register_lvsa_backend()


class TestMaybeInstallHunyuanHook:
    def test_disabled_by_default(self, clean_env):
        """Without LVSA_HUNYUAN_HOOK=1, install is a silent no-op."""
        from lvsa_vllm_omni.register import maybe_install_hunyuan_hook
        maybe_install_hunyuan_hook()  # should not raise

    def test_disabled_when_explicitly_false(self, monkeypatch, clean_env):
        monkeypatch.setenv("LVSA_HUNYUAN_HOOK", "0")
        from lvsa_vllm_omni.register import maybe_install_hunyuan_hook
        maybe_install_hunyuan_hook()

    def test_enabled_without_t_lat_warns(self, monkeypatch, clean_env, capsys):
        """When LVSA_HUNYUAN_HOOK=1 but T_lat not set, warns and skips."""
        monkeypatch.setenv("LVSA_HUNYUAN_HOOK", "1")
        from lvsa_vllm_omni.register import maybe_install_hunyuan_hook
        maybe_install_hunyuan_hook()
        captured = capsys.readouterr()
        # Either warns about missing T_lat or fails gracefully
        assert (
            "Warning" in captured.out
            or "failed" in captured.out
            or "TOTAL_LATENT_FRAMES" in captured.out
        )

    def test_enabled_with_t_lat_attempts_install(self, monkeypatch, clean_env, capsys):
        """With both env vars set, attempts install — fails gracefully without vllm-omni."""
        monkeypatch.setenv("LVSA_HUNYUAN_HOOK", "1")
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", "33")
        from lvsa_vllm_omni.register import maybe_install_hunyuan_hook
        maybe_install_hunyuan_hook()  # should not raise
        captured = capsys.readouterr()
        # In CI without vllm-omni, the install will catch ImportError and print
        # "Hook installation failed" or similar. We just verify no exception
        # bubbled up and there's some output.
        assert "Hook" in captured.out or "failed" in captured.out or "Installed" in captured.out


class TestMaybeInstallCosmos3Hook:
    """LVSA_COSMOS3_HOOK gating — mirrors the wan/hunyuan installers."""

    def test_disabled_by_default(self, clean_env):
        """Without LVSA_COSMOS3_HOOK=1, install is a silent no-op."""
        from lvsa_vllm_omni.register import maybe_install_cosmos3_hook
        maybe_install_cosmos3_hook()  # should not raise

    def test_disabled_when_explicitly_false(self, monkeypatch, clean_env, capsys):
        monkeypatch.setenv("LVSA_COSMOS3_HOOK", "0")
        from lvsa_vllm_omni.register import maybe_install_cosmos3_hook
        maybe_install_cosmos3_hook()
        # Falsy → silent, no output
        assert capsys.readouterr().out == ""

    def test_enabled_without_t_lat_warns(self, monkeypatch, clean_env, capsys):
        """LVSA_COSMOS3_HOOK=1 but T_lat unset → warn-and-skip, no raise."""
        monkeypatch.setenv("LVSA_COSMOS3_HOOK", "1")
        from lvsa_vllm_omni.register import maybe_install_cosmos3_hook
        maybe_install_cosmos3_hook()
        captured = capsys.readouterr()
        # Either the explicit "T_lat not set" warning, or — if vllm-omni's
        # cosmos3 import is attempted first and fails — the graceful failure.
        assert (
            "Warning" in captured.out
            or "TOTAL_LATENT_FRAMES" in captured.out
            or "failed" in captured.out
        )

    def test_enabled_with_t_lat_attempts_install(self, monkeypatch, clean_env, capsys):
        """With both env vars set, attempts install — fails gracefully w/o vllm-omni."""
        monkeypatch.setenv("LVSA_COSMOS3_HOOK", "1")
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", "48")
        from lvsa_vllm_omni.register import maybe_install_cosmos3_hook
        maybe_install_cosmos3_hook()  # must not raise — except Exception is caught
        captured = capsys.readouterr()
        # Without vllm-omni the cosmos3_hook import raises, caught and printed as
        # "Cosmos3 hook installation failed: ...". (With vllm-omni it would install.)
        assert (
            "failed" in captured.out
            or "Cosmos3" in captured.out
            or "Installed" in captured.out
            or "hook" in captured.out.lower()
        )


class TestHookBoolParsing:
    """LVSA_HUNYUAN_HOOK / LVSA_WAN_HOOK truthy-string handling."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes"])
    def test_truthy_values_trigger(self, value, monkeypatch, clean_env, capsys):
        monkeypatch.setenv("LVSA_HUNYUAN_HOOK", value)
        # Triggering with no T_lat will print a warning; using that as proof
        # the truthy check passed.
        from lvsa_vllm_omni.register import maybe_install_hunyuan_hook
        maybe_install_hunyuan_hook()
        captured = capsys.readouterr()
        assert captured.out != ""  # some output (Warning or failed)

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "anything-else"])
    def test_falsy_values_skip(self, value, monkeypatch, clean_env, capsys):
        monkeypatch.setenv("LVSA_HUNYUAN_HOOK", value)
        from lvsa_vllm_omni.register import maybe_install_hunyuan_hook
        maybe_install_hunyuan_hook()
        captured = capsys.readouterr()
        # Falsy values are silent — no print
        assert captured.out == ""
