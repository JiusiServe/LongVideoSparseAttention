"""Tests for report generation."""

import json
import csv
import pytest

from vqeval.core.report import Report, ReportGenerator


@pytest.fixture
def sample_report():
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
        },
        composite_score=68.5,
        composite_verdict="fair",
        weights={"spatial_quality": 0.5, "temporal_coherence": 0.5},
        config={"video_path": "test_video.mp4"},
        elapsed_seconds=12.5,
    )


class TestReport:
    def test_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert d["video_path"] == "test_video.mp4"
        assert d["composite_score"] == 68.5
        assert d["composite_verdict"] == "fair"
        assert "spatial_quality" in d["dimensions"]
        assert "temporal_coherence" in d["dimensions"]

    def test_to_json(self, sample_report):
        j = sample_report.to_json()
        parsed = json.loads(j)
        assert parsed["composite_score"] == 68.5
        assert "dimensions" in parsed

    def test_to_json_is_valid_json(self, sample_report):
        j = sample_report.to_json()
        # Should not raise
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_raw_results_default_empty(self):
        report = Report(
            video_path="v.mp4",
            video_meta={"width": 640, "height": 480},
            dimension_results={},
            composite_score=50.0,
            composite_verdict="fair",
            weights={},
            config={},
            elapsed_seconds=1.0,
        )
        assert report.raw_results == {}

    def test_raw_results_excluded_from_to_dict(self, sample_report):
        sample_report.raw_results = {"spatial_quality": "some_eval_result_obj"}
        d = sample_report.to_dict()
        assert "raw_results" not in d

    def test_raw_results_excluded_from_to_json(self, sample_report):
        sample_report.raw_results = {"spatial_quality": "some_eval_result_obj"}
        j = sample_report.to_json()
        parsed = json.loads(j)
        assert "raw_results" not in parsed


class TestReportGenerator:
    def test_to_json_file(self, sample_report, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "report.json")
        gen.to_json_file(sample_report, path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["composite_score"] == 68.5

    def test_to_csv(self, sample_report, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "report.csv")
        gen.to_csv([sample_report], path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["video_path"] == "test_video.mp4"
        assert float(rows[0]["composite_score"]) == 68.5

    def test_to_csv_multiple_reports(self, sample_report, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "report.csv")
        gen.to_csv([sample_report, sample_report], path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_to_csv_empty(self, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "report.csv")
        gen.to_csv([], path)
        # File should not be created for empty list

    def test_to_html(self, sample_report, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "report.html")
        gen.to_html(sample_report, path)

        with open(path) as f:
            html = f.read()
        assert "VQeval" in html
        assert "68" in html  # composite score
        assert "test_video.mp4" in html

    def test_to_html_batch(self, sample_report, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "batch.html")
        gen.to_html_batch([sample_report], path)

        with open(path) as f:
            html = f.read()
        assert "VQeval" in html
        assert "1" in html  # total count
