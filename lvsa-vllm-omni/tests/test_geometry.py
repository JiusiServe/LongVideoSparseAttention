"""Tests for the patches-per-frame geometry resolution chain.

Resolution order (first match wins):
  1. ``LVSA_PATCHES_PER_FRAME`` explicit override (comma-separated list or int).
  2. Derivation from ``LVSA_VIDEO_HEIGHT`` + ``LVSA_VIDEO_WIDTH`` +
     ``LVSA_VAE_SPATIAL_FACTOR`` (default 8) + ``LVSA_PATCH_SIZE`` (default 2).
  3. Built-in default: ``[1560]`` (Wan/HunyuanVideo at 480×832).
"""
from __future__ import annotations

import os
import pytest

from lvsa_vllm_omni.config import candidate_patches_per_frame


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    yield


class TestDefault:
    def test_returns_default_when_nothing_set(self):
        assert candidate_patches_per_frame() == [1560]


class TestExplicitOverride:
    def test_single_int_override(self, monkeypatch):
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", "2340")
        assert candidate_patches_per_frame() == [2340]

    def test_comma_separated_list(self, monkeypatch):
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", "1560,2340,3600")
        assert candidate_patches_per_frame() == [1560, 2340, 3600]

    def test_whitespace_around_commas(self, monkeypatch):
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", " 1560 , 2340 ")
        assert candidate_patches_per_frame() == [1560, 2340]

    def test_override_wins_over_resolution(self, monkeypatch):
        # Set both override and resolution; override should win.
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", "999")
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "720")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "1280")
        assert candidate_patches_per_frame() == [999]

    def test_invalid_override_falls_through_to_default(self, monkeypatch):
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", "not-an-int")
        # Falls through to resolution derivation, then default (no resolution set)
        assert candidate_patches_per_frame() == [1560]

    def test_invalid_override_falls_through_to_resolution(self, monkeypatch):
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", "not-an-int")
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "720")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "1280")
        # (720/8/2) * (1280/8/2) = 45 * 80 = 3600
        assert candidate_patches_per_frame() == [3600]


class TestResolutionDerivation:
    def test_480_832_matches_default(self, monkeypatch):
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "480")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "832")
        # (480/8/2) * (832/8/2) = 30 * 52 = 1560
        assert candidate_patches_per_frame() == [1560]

    def test_720p(self, monkeypatch):
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "720")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "1280")
        # (720/8/2) * (1280/8/2) = 45 * 80 = 3600
        assert candidate_patches_per_frame() == [3600]

    def test_custom_vae_spatial_factor(self, monkeypatch):
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "480")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "832")
        monkeypatch.setenv("LVSA_VAE_SPATIAL_FACTOR", "16")
        # (480/16/2) * (832/16/2) = 15 * 26 = 390
        assert candidate_patches_per_frame() == [390]

    def test_custom_patch_size(self, monkeypatch):
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "480")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "832")
        monkeypatch.setenv("LVSA_PATCH_SIZE", "4")
        # (480/8/4) * (832/8/4) = 15 * 26 = 390
        assert candidate_patches_per_frame() == [390]

    def test_partial_resolution_falls_through_to_default(self, monkeypatch):
        # Only height set, not width — falls through
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "720")
        assert candidate_patches_per_frame() == [1560]

    def test_zero_resolution_falls_through_to_default(self, monkeypatch):
        # Explicit zero is ignored; derivation requires positive values.
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "0")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "0")
        assert candidate_patches_per_frame() == [1560]
