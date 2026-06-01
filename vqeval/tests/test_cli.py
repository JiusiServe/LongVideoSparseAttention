"""Tests for CLI argument parsing and helpers."""

import pytest
from click.testing import CliRunner

from vqeval.cli import main, _parse_weights, _parse_dimensions


class TestParseWeights:
    def test_basic(self):
        result = _parse_weights("spatial=0.3,temporal=0.3")
        assert result["spatial_quality"] == 0.3
        assert result["temporal_coherence"] == 0.3

    def test_aliases(self):
        result = _parse_weights("artifacts=0.5,dynamic=0.2")
        assert result["artifact_detection"] == 0.5
        assert result["dynamic_quality"] == 0.2

    def test_full_names(self):
        result = _parse_weights("spatial_quality=0.4")
        assert result["spatial_quality"] == 0.4

    def test_empty_string(self):
        result = _parse_weights("")
        # Should return defaults
        assert "spatial_quality" in result


class TestParseDimensions:
    def test_short_names(self):
        result = _parse_dimensions("spatial,temporal,loop")
        assert result == ["spatial_quality", "temporal_coherence", "loop_quality"]

    def test_full_names(self):
        result = _parse_dimensions("spatial_quality,artifact_detection")
        assert result == ["spatial_quality", "artifact_detection"]

    def test_single(self):
        result = _parse_dimensions("spatial")
        assert result == ["spatial_quality"]


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "vqeval" in result.output.lower()

    def test_evaluate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--prompt" in result.output

    def test_evaluate_help_has_figures_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--figures" in result.output
        assert "--figure-format" in result.output

    def test_batch_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0

    def test_batch_help_has_figures_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--figures" in result.output
        assert "--figure-format" in result.output

    def test_batch_help_has_csv_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--from-csv" in result.output
        assert "--video-col" in result.output
        assert "--prompt-col" in result.output
        assert "--status-col" in result.output
        assert "--status-ok" in result.output

    def test_calibrate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["calibrate", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output

    def test_evaluate_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "nonexistent.mp4"])
        assert result.exit_code != 0
