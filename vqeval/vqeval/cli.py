"""CLI entry point for VQeval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch
from rich.console import Console
from rich.table import Table

from vqeval.core.config import EvalConfig, BatchConfig, DEFAULT_WEIGHTS
from vqeval.core.pipeline import EvalPipeline, BatchPipeline
from vqeval.core.report import ReportGenerator

# Import evaluators to trigger registration
import vqeval.evaluators.spatial_quality
import vqeval.evaluators.temporal_coherence
import vqeval.evaluators.loop_quality
import vqeval.evaluators.artifact_detection
import vqeval.evaluators.dynamic_quality
import vqeval.evaluators.text_alignment

console = Console()


def _parse_weights(weights_str: str) -> dict[str, float]:
    """Parse 'spatial=0.3,temporal=0.3,artifacts=0.4' into a dict."""
    result = dict(DEFAULT_WEIGHTS)
    # Map short names to full names
    aliases = {
        "spatial": "spatial_quality",
        "temporal": "temporal_coherence",
        "loop": "loop_quality",
        "artifacts": "artifact_detection",
        "dynamic": "dynamic_quality",
        "text": "text_alignment",
    }
    for pair in weights_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.strip()
        key = aliases.get(key, key)
        result[key] = float(val.strip())
    return result


def _parse_dimensions(dims_str: str) -> list[str]:
    """Parse 'spatial,temporal,loop' into full dimension names."""
    aliases = {
        "spatial": "spatial_quality",
        "temporal": "temporal_coherence",
        "loop": "loop_quality",
        "artifacts": "artifact_detection",
        "dynamic": "dynamic_quality",
        "text": "text_alignment",
    }
    result = []
    for d in dims_str.split(","):
        d = d.strip()
        result.append(aliases.get(d, d))
    return result


@click.group()
@click.version_option(package_name="vqeval")
def main():
    """VQeval - AI-generated video quality assessment tool."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--prompt", type=str, default=None, help="Generation prompt for text-video alignment")
@click.option("--reference-image", type=click.Path(exists=True), default=None,
              help="Reference/conditioning image")
@click.option("--dimensions", type=str, default=None,
              help="Comma-separated dimensions to evaluate (e.g., spatial,temporal,loop)")
@click.option("--weights", type=str, default=None,
              help="Custom weights (e.g., spatial=0.3,temporal=0.3,artifacts=0.4)")
@click.option("--presets", type=click.Path(exists=True), default=None,
              help="Path to custom normalization presets JSON")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSON report path")
@click.option("--html-report", type=click.Path(), default=None,
              help="Generate HTML visual report")
@click.option("--export-frames", type=click.Path(), default=None,
              help="Export annotated frames to directory")
@click.option("--device", type=str, default=None,
              help="Compute device (cuda/cpu, auto-detected by default)")
@click.option("--figures", type=click.Path(), default=None,
              help="Output directory for publication-quality figures")
@click.option("--figure-format", type=click.Choice(["png", "pdf", "both"]),
              default="png", help="Figure output format (default: png)")
@click.option("--sample-fps", type=float, default=None,
              help="Frame sampling rate in fps (default: 2 for videos >5s). Use 0 for all frames.")
def evaluate(video_path, prompt, reference_image, dimensions, weights,
             presets, output, html_report, export_frames, device, figures, figure_format,
             sample_fps):
    """Evaluate quality of an AI-generated video."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = EvalConfig(
        video_path=video_path,
        prompt=prompt,
        reference_image=reference_image,
        dimensions=_parse_dimensions(dimensions) if dimensions else None,
        weights=_parse_weights(weights) if weights else dict(DEFAULT_WEIGHTS),
        presets_path=presets,
        export_frames_dir=export_frames,
        html_report=html_report,
        device=device,
        sample_fps=sample_fps,
    )

    pipeline = EvalPipeline(config)
    report = pipeline.run()

    # Print summary table
    _print_summary_table(report)

    # Output JSON
    json_str = report.to_json()
    if output:
        Path(output).write_text(json_str)
        console.print(f"\n[green]JSON report saved: {output}[/green]")
    else:
        console.print(f"\n[dim]JSON report:[/dim]")
        console.print(json_str)

    # HTML report
    if html_report:
        generator = ReportGenerator()
        generator.to_html(report, html_report)
        console.print(f"[green]HTML report saved: {html_report}[/green]")

    # Figures
    if figures:
        from vqeval.figures import FigureGenerator
        fig_gen = FigureGenerator()
        paths = fig_gen.generate_single(report, Path(figures), fmt=figure_format)
        console.print(f"[green]Figures saved to {figures}/ ({len(paths)} files)[/green]")


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False), required=False,
                default=None)
@click.option("--from-csv", type=click.Path(exists=True, dir_okay=False), default=None,
              help="CSV file with video paths and metadata (alternative to INPUT_DIR)")
@click.option("--video-col", type=str, default="video_path",
              help="CSV column containing video file paths (default: video_path)")
@click.option("--prompt-col", type=str, default=None,
              help="CSV column containing per-video prompts")
@click.option("--status-col", type=str, default="status",
              help="CSV column to filter rows by (default: status)")
@click.option("--status-ok", type=str, default="ok",
              help="Value in status column that means 'process this row' (default: ok)")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output CSV report path")
@click.option("--html-report", type=click.Path(), default=None,
              help="Generate batch HTML report")
@click.option("--prompt", type=str, default=None, help="Shared generation prompt")
@click.option("--dimensions", type=str, default=None,
              help="Comma-separated dimensions to evaluate")
@click.option("--device", type=str, default=None,
              help="Compute device (cuda/cpu)")
@click.option("--figures", type=click.Path(), default=None,
              help="Output directory for publication-quality figures")
@click.option("--figure-format", type=click.Choice(["png", "pdf", "both"]),
              default="png", help="Figure output format (default: png)")
@click.option("--sample-fps", type=float, default=None,
              help="Frame sampling rate in fps (default: 2 for videos >5s). Use 0 for all frames.")
def batch(input_dir, from_csv, video_col, prompt_col, status_col, status_ok,
          output, html_report, prompt, dimensions, device, figures, figure_format,
          sample_fps):
    """Process a directory of videos and produce a summary report.

    Provide either INPUT_DIR (a directory to scan) or --from-csv (a CSV file
    with video paths and metadata). When using --from-csv, video paths are
    resolved relative to the CSV file's directory.
    """
    if not input_dir and not from_csv:
        raise click.UsageError("Provide either INPUT_DIR or --from-csv.")
    if input_dir and from_csv:
        raise click.UsageError("Provide either INPUT_DIR or --from-csv, not both.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_config = EvalConfig(
        prompt=prompt,
        dimensions=_parse_dimensions(dimensions) if dimensions else None,
        device=device,
        sample_fps=sample_fps,
    )
    batch_config = BatchConfig(
        input_dir=input_dir or "",
        from_csv=from_csv,
        video_col=video_col,
        prompt_col=prompt_col,
        status_col=status_col,
        status_ok=status_ok,
        output_csv=output,
        html_report=html_report,
        eval_config=eval_config,
    )

    pipeline = BatchPipeline(batch_config)
    reports = pipeline.run()

    if reports:
        avg = sum(r.composite_score for r in reports) / len(reports)
        console.print(f"\n[bold]Batch summary: {len(reports)} videos, average score: {avg:.1f}[/bold]")

        if figures:
            from vqeval.figures import FigureGenerator
            fig_gen = FigureGenerator()
            paths = fig_gen.generate_batch(reports, Path(figures), fmt=figure_format)
            console.print(f"[green]Figures saved to {figures}/ ({len(paths)} files)[/green]")


@main.command()
@click.option("--dataset", type=click.Path(exists=True, file_okay=False), required=True,
              help="Directory of calibration videos")
@click.option("--output", "-o", type=click.Path(), default="custom_presets.json",
              help="Output presets JSON path")
@click.option("--device", type=str, default=None)
def calibrate(dataset, output, device):
    """Calibrate normalization parameters from a dataset of videos."""
    from vqeval.normalization.calibration import Calibrator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[bold]Calibrating from: {dataset}[/bold]")
    console.print("[yellow]Calibration requires running full evaluation on all videos.[/yellow]")
    console.print("[yellow]This may take a while...[/yellow]")

    # Run batch evaluation to collect raw metric values
    eval_config = EvalConfig(device=device)
    batch_config = BatchConfig(input_dir=dataset, eval_config=eval_config)
    pipeline = BatchPipeline(batch_config)
    reports = pipeline.run()

    if not reports:
        console.print("[red]No reports generated. Check your dataset.[/red]")
        return

    # Extract raw metric values and build presets
    calibrator = Calibrator()
    presets = {}

    # Collect raw values per metric across all reports
    metric_values: dict[str, list[float]] = {}
    for r in reports:
        for dim_name, dim_data in r.dimension_results.items():
            for key, val in dim_data.items():
                if isinstance(val, (int, float)) and key not in ("score",):
                    full_key = f"{dim_name}.{key}"
                    metric_values.setdefault(full_key, []).append(float(val))

    for metric_name, values in metric_values.items():
        if len(values) >= 5:
            presets[metric_name] = calibrator.calibrate(metric_name, values)

    calibrator.save_presets(presets, output)
    console.print(f"[green]Calibration presets saved: {output}[/green]")
    console.print(f"  {len(presets)} metrics calibrated from {len(reports)} videos")


def _print_summary_table(report):
    """Print a formatted summary table of results."""
    table = Table(title="Quality Assessment Results", show_header=True)
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Verdict", justify="center")

    report_dict = report.to_dict()
    for dim_name, dim_data in report_dict["dimensions"].items():
        score = dim_data.get("score", 0)
        verdict = dim_data.get("verdict", "")
        color = _verdict_color(verdict)
        table.add_row(
            dim_name.replace("_", " ").title(),
            f"{score:.1f}",
            f"[{color}]{verdict}[/{color}]",
        )

    table.add_section()
    composite_color = _verdict_color(report_dict["composite_verdict"])
    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{report_dict['composite_score']:.1f}[/bold]",
        f"[bold {composite_color}]{report_dict['composite_verdict']}[/bold {composite_color}]",
    )

    console.print(table)


def _verdict_color(verdict: str) -> str:
    return {
        "excellent": "green",
        "good": "green",
        "fair": "yellow",
        "poor": "red",
        "bad": "red",
    }.get(verdict, "white")


if __name__ == "__main__":
    main()
