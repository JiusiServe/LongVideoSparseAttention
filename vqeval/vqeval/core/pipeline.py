"""Evaluation pipeline orchestrating video loading and evaluator execution."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import torch

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from vqeval.core.config import EvalConfig, BatchConfig, score_to_verdict
from vqeval.core.video_loader import VideoLoader, VideoData
from vqeval.core.report import Report, ReportGenerator
from vqeval.evaluators.base import get_evaluator_class, BaseEvaluator
from vqeval.models.model_registry import ModelRegistry

console = Console()


class EvalPipeline:
    """Orchestrates the full evaluation pipeline for a single video."""

    def __init__(self, config: EvalConfig):
        self.config = config
        loader_kwargs = {}
        if config.sample_fps is not None:
            if config.sample_fps < 0:
                raise ValueError(f"--sample-fps must be >= 0, got {config.sample_fps}")
            if config.sample_fps == 0:
                # 0 means "all frames": set threshold to infinity so every video
                # is treated as short enough to sample all frames.
                loader_kwargs["sample_all_threshold"] = float("inf")
            else:
                loader_kwargs["min_sample_fps"] = config.sample_fps
        self.loader = VideoLoader(**loader_kwargs)
        self.registry = ModelRegistry(device=config.device)

    def run(self) -> Report:
        """Execute the full evaluation pipeline."""
        start_time = time.time()

        # Load video
        console.print(f"[bold]Loading video:[/bold] {self.config.video_path}")
        video = self.loader.load(self.config.video_path)
        console.print(
            f"  {video.meta.width}x{video.meta.height}, "
            f"{video.meta.fps:.1f} fps, {video.meta.duration:.2f}s, "
            f"{video.num_frames} sampled frames"
        )

        # Determine active dimensions
        active_dims = self.config.get_active_dimensions()
        weights = self.config.get_effective_weights()

        # Run evaluators
        results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for dim_name in active_dims:
                task = progress.add_task(f"Evaluating {dim_name}...", total=1)
                evaluator = self._create_evaluator(dim_name)
                if evaluator.is_applicable(video):
                    try:
                        result = evaluator.evaluate(video)
                        results[dim_name] = result
                    except Exception as e:
                        console.print(f"  [red]Error in {dim_name}: {e}[/red]")
                else:
                    console.print(f"  [dim]Skipping {dim_name} (not applicable)[/dim]")
                progress.update(task, completed=1)
                # Free intermediate GPU memory between evaluators
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Compute composite score
        composite = self._compute_composite(results, weights)

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Evaluation complete[/bold green] in {elapsed:.1f}s"
        )
        console.print(f"[bold]Overall score: {composite:.1f} ({score_to_verdict(composite)})[/bold]")

        return Report(
            video_path=self.config.video_path,
            video_meta={
                "width": video.meta.width,
                "height": video.meta.height,
                "fps": video.meta.fps,
                "duration": video.meta.duration,
                "codec": video.meta.codec,
                "total_frames": video.meta.total_frames,
                "sampled_frames": video.num_frames,
            },
            dimension_results={
                name: result.to_dict()[name] for name, result in results.items()
            },
            composite_score=composite,
            composite_verdict=score_to_verdict(composite),
            weights=weights,
            config=self.config.to_dict(),
            elapsed_seconds=elapsed,
            raw_results=results,
        )

    def _create_evaluator(self, name: str) -> BaseEvaluator:
        cls = get_evaluator_class(name)
        return cls(config=self.config, model_registry=self.registry)

    def _compute_composite(
        self, results: dict[str, "EvalResult"], weights: dict[str, float]
    ) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for name, result in results.items():
            w = weights.get(name, 0.0)
            weighted_sum += result.score * w
            total_weight += w
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight


class BatchPipeline:
    """Process a directory of videos in batch mode."""

    def __init__(self, batch_config: BatchConfig):
        self.batch_config = batch_config

    def run(self) -> list[Report]:
        """Process all videos and return reports."""
        if self.batch_config.from_csv:
            return self._run_from_csv()
        return self._run_from_dir()

    def _run_from_dir(self) -> list[Report]:
        """Process all videos in a directory."""
        video_files = VideoLoader.get_video_files(self.batch_config.input_dir)
        if not video_files:
            console.print("[yellow]No supported video files found.[/yellow]")
            return []

        console.print(f"[bold]Found {len(video_files)} videos to process[/bold]")
        jobs = [(vp, self.batch_config.eval_config.prompt, {}) for vp in video_files]
        return self._process_jobs(jobs)

    def _run_from_csv(self) -> list[Report]:
        """Process videos listed in a CSV file."""
        csv_path = Path(self.batch_config.from_csv)
        csv_dir = csv_path.parent

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        video_col = self.batch_config.video_col
        status_col = self.batch_config.status_col
        status_ok = self.batch_config.status_ok
        prompt_col = self.batch_config.prompt_col

        # Filter and resolve paths
        jobs: list[tuple[str, str | None, dict]] = []
        skipped = 0
        for row in rows:
            # Filter by status if the column exists
            if status_col and status_col in row:
                if row[status_col] != status_ok:
                    skipped += 1
                    continue

            raw_path = row.get(video_col, "")
            if not raw_path:
                continue

            # Resolve relative path: try cwd first, then CSV's directory
            video_path = Path(raw_path)
            if not video_path.is_absolute():
                if (Path.cwd() / video_path).exists():
                    video_path = Path.cwd() / video_path
                elif (csv_dir / video_path).exists():
                    video_path = csv_dir / video_path
                else:
                    console.print(f"[yellow]Skipping (not found): {raw_path}[/yellow]")
                    skipped += 1
                    continue
            elif not video_path.exists():
                console.print(f"[yellow]Skipping (not found): {raw_path}[/yellow]")
                skipped += 1
                continue

            # Per-row prompt (falls back to --prompt)
            prompt = None
            if prompt_col and prompt_col in row:
                prompt = row[prompt_col] or None
            if prompt is None:
                prompt = self.batch_config.eval_config.prompt

            # Extra metadata: all CSV columns except video_path
            extra = {k: v for k, v in row.items() if k != video_col}

            jobs.append((str(video_path), prompt, extra))

        if skipped:
            console.print(f"[dim]Skipped {skipped} rows (filtered or missing)[/dim]")
        if not jobs:
            console.print("[yellow]No processable videos found in CSV.[/yellow]")
            return []

        console.print(f"[bold]Found {len(jobs)} videos to process from CSV[/bold]")
        return self._process_jobs(jobs)

    def _process_jobs(
        self, jobs: list[tuple[str, str | None, dict]]
    ) -> list[Report]:
        """Run evaluation for a list of (video_path, prompt, extra_meta) tuples."""
        reports = []
        total = len(jobs)

        for i, (video_path, prompt, extra_meta) in enumerate(jobs, 1):
            console.print(f"\n[bold cyan]--- Video {i}/{total} ---[/bold cyan]")
            config = EvalConfig(
                video_path=video_path,
                loop=self.batch_config.eval_config.loop,
                prompt=prompt,
                dimensions=self.batch_config.eval_config.dimensions,
                weights=self.batch_config.eval_config.weights,
                presets_path=self.batch_config.eval_config.presets_path,
                device=self.batch_config.eval_config.device,
                sample_fps=self.batch_config.eval_config.sample_fps,
            )
            try:
                pipeline = EvalPipeline(config)
                report = pipeline.run()
                report.extra_meta = extra_meta
                reports.append(report)
            except Exception as e:
                console.print(f"[red]Error processing {video_path}: {e}[/red]")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Generate batch outputs
        if reports:
            generator = ReportGenerator()
            if self.batch_config.output_csv:
                generator.to_csv(reports, self.batch_config.output_csv)
                console.print(f"\n[green]CSV report saved: {self.batch_config.output_csv}[/green]")
            if self.batch_config.html_report:
                generator.to_html_batch(reports, self.batch_config.html_report)
                console.print(f"[green]HTML report saved: {self.batch_config.html_report}[/green]")

        return reports
