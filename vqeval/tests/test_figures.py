"""Tests for figure generation using synthetic/mock Report objects (no GPU needed)."""

import pytest
from pathlib import Path

from vqeval.core.report import Report
from vqeval.evaluators.base import EvalResult
from vqeval.figures import FigureGenerator


@pytest.fixture
def single_report():
    """A Report with enough dimensions for radar chart and some traces."""
    spatial_traces = {
        "frame_indices": list(range(10)),
        "brisque_per_frame": [30.0 + i for i in range(10)],
        "sharpness_per_frame": [100.0 - i * 2 for i in range(10)],
    }
    temporal_traces = {
        "clip_consecutive_sims": [0.95 - i * 0.01 for i in range(9)],
        "flow_magnitudes": [2.0 + i * 0.5 for i in range(9)],
    }

    spatial_result = EvalResult(
        dimension="spatial_quality",
        score=72.5,
        verdict="good",
        metrics={"brisque_mean": 28.4, "niqe_mean": 4.2},
        traces=spatial_traces,
    )
    temporal_result = EvalResult(
        dimension="temporal_coherence",
        score=65.0,
        verdict="fair",
        metrics={"flickering_score": 78.0, "motion_smoothness": 61.0},
        traces=temporal_traces,
    )

    return Report(
        video_path="test_video.mp4",
        video_meta={
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "duration": 5.0,
            "codec": "h264",
            "total_frames": 150,
            "sampled_frames": 150,
        },
        dimension_results={
            "spatial_quality": {
                "score": 72.5,
                "verdict": "good",
                "brisque_mean": 28.4,
                "niqe_mean": 4.2,
            },
            "temporal_coherence": {
                "score": 65.0,
                "verdict": "fair",
                "flickering_score": 78.0,
                "motion_smoothness": 61.0,
            },
            "loop_quality": {
                "score": 80.0,
                "verdict": "good",
                "cycle_detected": False,
            },
            "artifact_detection": {
                "score": 55.0,
                "verdict": "fair",
            },
        },
        composite_score=68.5,
        composite_verdict="fair",
        weights={
            "spatial_quality": 0.25,
            "temporal_coherence": 0.25,
            "loop_quality": 0.25,
            "artifact_detection": 0.25,
        },
        config={"video_path": "test_video.mp4"},
        elapsed_seconds=12.5,
        raw_results={
            "spatial_quality": spatial_result,
            "temporal_coherence": temporal_result,
        },
    )


@pytest.fixture
def minimal_report():
    """A Report with minimal data and no traces."""
    return Report(
        video_path="minimal.mp4",
        video_meta={"width": 640, "height": 480, "fps": 24.0, "duration": 2.0},
        dimension_results={
            "spatial_quality": {"score": 50.0, "verdict": "fair"},
        },
        composite_score=50.0,
        composite_verdict="fair",
        weights={"spatial_quality": 1.0},
        config={},
        elapsed_seconds=3.0,
    )


@pytest.fixture
def batch_reports(single_report, minimal_report):
    """A list of reports for batch figure generation."""
    return [single_report, minimal_report]


class TestFigureGeneratorInstantiation:
    def test_can_instantiate(self):
        gen = FigureGenerator()
        assert gen is not None

    def test_has_generate_single_method(self):
        gen = FigureGenerator()
        assert callable(getattr(gen, "generate_single", None))

    def test_has_generate_batch_method(self):
        gen = FigureGenerator()
        assert callable(getattr(gen, "generate_batch", None))


class TestSingleVideoFigures:
    def test_generate_single_returns_list(self, single_report, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="png")
        assert isinstance(paths, list)

    def test_generate_single_creates_files(self, single_report, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="png")
        assert len(paths) > 0
        for p in paths:
            assert Path(p).exists(), f"Expected figure file not found: {p}"

    def test_generate_single_png_format(self, single_report, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="png")
        for p in paths:
            assert str(p).endswith(".png")

    def test_generate_single_pdf_format(self, single_report, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="pdf")
        for p in paths:
            assert str(p).endswith(".pdf")

    def test_generate_single_creates_output_dir(self, single_report, tmp_path):
        output_dir = tmp_path / "subfolder" / "figures"
        gen = FigureGenerator()
        gen.generate_single(single_report, output_dir, fmt="png")
        assert output_dir.exists()

    def test_radar_chart_created(self, single_report, tmp_path):
        """With 4 dimensions, a radar chart should be generated."""
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="png")
        filenames = [Path(p).name for p in paths]
        assert "radar_scores.png" in filenames

    def test_dimension_bars_created(self, single_report, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="png")
        filenames = [Path(p).name for p in paths]
        assert "dimension_scores.png" in filenames

    def test_temporal_profile_created_with_traces(self, single_report, tmp_path):
        """Temporal profile should be created when spatial traces are available."""
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="png")
        filenames = [Path(p).name for p in paths]
        assert "temporal_profile.png" in filenames

    def test_coherence_profile_created_with_traces(self, single_report, tmp_path):
        """Coherence profile should be created when temporal traces are available."""
        gen = FigureGenerator()
        paths = gen.generate_single(single_report, tmp_path, fmt="png")
        filenames = [Path(p).name for p in paths]
        assert "coherence_profile.png" in filenames

    def test_minimal_report_still_produces_figures(self, minimal_report, tmp_path):
        """Even a report with minimal data should produce at least score overview charts."""
        gen = FigureGenerator()
        paths = gen.generate_single(minimal_report, tmp_path, fmt="png")
        # Should at least have the dimension bars (radar needs 3+ dims)
        assert len(paths) >= 1
        for p in paths:
            assert Path(p).exists()


class TestBatchFigures:
    def test_generate_batch_returns_list(self, batch_reports, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_batch(batch_reports, tmp_path, fmt="png")
        assert isinstance(paths, list)

    def test_generate_batch_creates_files(self, batch_reports, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_batch(batch_reports, tmp_path, fmt="png")
        assert len(paths) > 0
        for p in paths:
            assert Path(p).exists(), f"Expected figure file not found: {p}"

    def test_batch_ranking_created(self, batch_reports, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_batch(batch_reports, tmp_path, fmt="png")
        filenames = [Path(p).name for p in paths]
        assert "batch_ranking.png" in filenames

    def test_batch_dimension_comparison_created(self, batch_reports, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_batch(batch_reports, tmp_path, fmt="png")
        filenames = [Path(p).name for p in paths]
        assert "batch_dimension_comparison.png" in filenames

    def test_batch_distributions_created(self, batch_reports, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_batch(batch_reports, tmp_path, fmt="png")
        filenames = [Path(p).name for p in paths]
        assert "batch_distributions.png" in filenames

    def test_batch_also_generates_per_video_figures(self, batch_reports, tmp_path):
        """Batch generation should also create per-video subdirectories."""
        gen = FigureGenerator()
        paths = gen.generate_batch(batch_reports, tmp_path, fmt="png")
        # Check that subdirectories were created for individual videos
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1  # At least one per-video subdirectory

    def test_batch_files_are_nonzero_size(self, batch_reports, tmp_path):
        gen = FigureGenerator()
        paths = gen.generate_batch(batch_reports, tmp_path, fmt="png")
        for p in paths:
            assert Path(p).stat().st_size > 0, f"Figure file is empty: {p}"
