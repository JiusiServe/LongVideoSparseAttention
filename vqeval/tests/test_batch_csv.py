"""Tests for CSV-driven batch mode."""

import csv
import json
import pytest

from vqeval.core.config import BatchConfig, EvalConfig
from vqeval.core.report import Report, ReportGenerator


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal benchmark CSV for testing."""
    csv_path = tmp_path / "bench.csv"
    rows = [
        {"model": "wan", "backend": "baseline", "frame_label": "1x",
         "status": "ok", "latency_s": "100", "video_path": "videos/a.mp4"},
        {"model": "wan", "backend": "lvsa", "frame_label": "1x",
         "status": "ok", "latency_s": "80", "video_path": "videos/b.mp4"},
        {"model": "wan", "backend": "baseline", "frame_label": "2x",
         "status": "error", "latency_s": "0", "video_path": "videos/c.mp4"},
        {"model": "cog", "backend": "baseline", "frame_label": "1x",
         "status": "oom", "latency_s": "0", "video_path": "videos/d.mp4"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@pytest.fixture
def sample_report_with_meta():
    """A Report with extra_meta populated."""
    return Report(
        video_path="videos/a.mp4",
        video_meta={"width": 480, "height": 832, "fps": 8.0, "duration": 10.0,
                     "codec": "h264", "total_frames": 81, "sampled_frames": 20},
        dimension_results={
            "spatial_quality": {"score": 70.0, "verdict": "good"},
        },
        composite_score=70.0,
        composite_verdict="good",
        weights={"spatial_quality": 1.0},
        config={},
        elapsed_seconds=5.0,
        extra_meta={"model": "wan", "backend": "baseline", "frame_label": "1x",
                     "latency_s": "100", "status": "ok"},
    )


class TestBatchConfigCsvFields:
    def test_default_csv_fields(self):
        cfg = BatchConfig()
        assert cfg.from_csv is None
        assert cfg.video_col == "video_path"
        assert cfg.prompt_col is None
        assert cfg.status_col == "status"
        assert cfg.status_ok == "ok"

    def test_custom_csv_fields(self):
        cfg = BatchConfig(
            from_csv="data.csv",
            video_col="file",
            prompt_col="caption",
            status_col="result",
            status_ok="success",
        )
        assert cfg.from_csv == "data.csv"
        assert cfg.video_col == "file"
        assert cfg.prompt_col == "caption"
        assert cfg.status_col == "result"
        assert cfg.status_ok == "success"


class TestReportExtraMeta:
    def test_extra_meta_default_empty(self):
        report = Report(
            video_path="v.mp4",
            video_meta={},
            dimension_results={},
            composite_score=50.0,
            composite_verdict="fair",
            weights={},
            config={},
            elapsed_seconds=1.0,
        )
        assert report.extra_meta == {}

    def test_extra_meta_in_to_dict(self, sample_report_with_meta):
        d = sample_report_with_meta.to_dict()
        assert "extra_meta" in d
        assert d["extra_meta"]["model"] == "wan"

    def test_extra_meta_excluded_when_empty(self):
        report = Report(
            video_path="v.mp4",
            video_meta={},
            dimension_results={},
            composite_score=50.0,
            composite_verdict="fair",
            weights={},
            config={},
            elapsed_seconds=1.0,
        )
        d = report.to_dict()
        assert "extra_meta" not in d

    def test_extra_meta_in_json(self, sample_report_with_meta):
        j = sample_report_with_meta.to_json()
        parsed = json.loads(j)
        assert parsed["extra_meta"]["backend"] == "baseline"


class TestCsvMergedOutput:
    def test_csv_with_extra_meta(self, sample_report_with_meta, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "merged.csv")
        gen.to_csv([sample_report_with_meta], path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        # Extra meta columns should be present
        assert row["model"] == "wan"
        assert row["backend"] == "baseline"
        assert row["latency_s"] == "100"
        # Quality columns should also be present
        assert float(row["composite_score"]) == 70.0
        assert row["composite_verdict"] == "good"

    def test_csv_extra_meta_columns_come_first(self, sample_report_with_meta, tmp_path):
        gen = ReportGenerator()
        path = str(tmp_path / "merged.csv")
        gen.to_csv([sample_report_with_meta], path)

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)

        # Extra meta keys should appear before video_path
        model_idx = header.index("model")
        video_idx = header.index("video_path")
        assert model_idx < video_idx

    def test_csv_without_extra_meta(self, tmp_path):
        """Reports without extra_meta should produce the standard CSV format."""
        report = Report(
            video_path="v.mp4",
            video_meta={"width": 640, "height": 480, "duration": 5.0},
            dimension_results={"spatial_quality": {"score": 60.0, "verdict": "fair"}},
            composite_score=60.0,
            composite_verdict="fair",
            weights={"spatial_quality": 1.0},
            config={},
            elapsed_seconds=2.0,
        )
        gen = ReportGenerator()
        path = str(tmp_path / "standard.csv")
        gen.to_csv([report], path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["video_path"] == "v.mp4"
        assert "model" not in rows[0]


class TestCsvParsing:
    def test_csv_read(self, sample_csv):
        """Verify the test CSV fixture is well-formed."""
        with open(sample_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4
        ok_rows = [r for r in rows if r["status"] == "ok"]
        assert len(ok_rows) == 2

    def test_status_filtering(self, sample_csv):
        """Verify only 'ok' rows would be processed."""
        with open(sample_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        assert len(ok_rows) == 2
        assert all(r["status"] == "ok" for r in ok_rows)


class TestFigureShortNames:
    def test_short_name_from_report_with_meta(self, sample_report_with_meta):
        from vqeval.figures.batch import _short_name_from_report
        name = _short_name_from_report(sample_report_with_meta)
        assert name == "wan/baseline/1x"

    def test_short_name_from_report_without_meta(self):
        from vqeval.figures.batch import _short_name_from_report
        report = Report(
            video_path="some_long_video_name.mp4",
            video_meta={},
            dimension_results={},
            composite_score=50.0,
            composite_verdict="fair",
            weights={},
            config={},
            elapsed_seconds=1.0,
        )
        name = _short_name_from_report(report)
        assert name == "some_long_video_name"

    def test_short_name_truncation(self):
        from vqeval.figures.batch import _short_name_from_report
        report = Report(
            video_path="a" * 100 + ".mp4",
            video_meta={},
            dimension_results={},
            composite_score=50.0,
            composite_verdict="fair",
            weights={},
            config={},
            elapsed_seconds=1.0,
        )
        name = _short_name_from_report(report)
        assert len(name) <= 40
