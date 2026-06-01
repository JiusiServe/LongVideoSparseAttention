# VQeval

AI-generated video quality assessment tool. Evaluates videos across multiple quality dimensions and produces structured reports.

## Features

- **6 quality dimensions**: spatial quality, temporal coherence, repetition detection, AI artifact detection, dynamic quality, text-video alignment
- **Plugin architecture**: each dimension is an independent evaluator that can be enabled/disabled
- **Shared model management**: CLIP, DINOv2, RAFT, and other models are loaded once and shared across evaluators
- **Multiple output formats**: JSON, CSV, and HTML visual reports
- **Batch processing**: evaluate entire directories or CSV-driven benchmark suites
- **CSV integration**: `--from-csv` reads video paths and metadata from a CSV, merges quality scores back
- **Calibration**: build custom normalization presets from your own dataset
- **GPU-accelerated**: runs on a single NVIDIA GPU (8 GB+ VRAM), with CPU fallback

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended, 8 GB+ VRAM)
- FFmpeg (optional, for audio stream detection)

### Key dependencies

| Library | Purpose |
|---------|---------|
| torch + torchvision | GPU inference, DINOv2 via torch.hub |
| opencv-python / opencv-contrib-python | Video decoding, BRISQUE, optical flow (Farneback) |
| pyiqa | NIQE, CLIP-IQA, and other IQA metrics |
| open_clip_torch | CLIP model loading (ViT-B/16) |
| mediapipe | Pose/hand detection for anatomical error detection |
| scikit-image | SSIM, gradient computation |
| lpips | Perceptual distance metrics |
| jinja2 | HTML report templating |
| matplotlib | Figure generation (radar charts, bar plots, heatmaps) |
| rich | CLI formatting and progress bars |
| click | CLI argument parsing |

## Usage

### Evaluate a single video

```bash
vqeval evaluate video.mp4
```

### With text prompt for alignment scoring

```bash
vqeval evaluate video.mp4 --prompt "a cat walking through autumn leaves"
```

### Select specific dimensions

```bash
vqeval evaluate video.mp4 --dimensions spatial,temporal,loop
```

### Custom weights

```bash
vqeval evaluate video.mp4 --weights spatial=0.3,temporal=0.3,artifacts=0.4
```

### Save reports

```bash
vqeval evaluate video.mp4 -o report.json --html-report report.html
```

### Generate figures

```bash
vqeval evaluate video.mp4 --figures ./figs/
```

### Generate figures in PDF format

```bash
vqeval evaluate video.mp4 --figures ./figs/ --figure-format pdf
```

### Batch mode (directory)

```bash
vqeval batch ./generated_videos/ --output report.csv --html-report report.html
```

### Batch mode with figures

```bash
vqeval batch ./videos/ --figures ./figs/ --output report.csv
```

### Batch mode from CSV

Process videos listed in a benchmark CSV file. Rows are filtered by a status column, video paths are resolved relative to the CSV file's directory, and all CSV metadata columns are carried through to the output.

```bash
vqeval batch --from-csv benchmark_results.csv \
    --prompt "A dog running in the forest." \
    --output results.csv \
    --figures ./figs/
```

The output CSV merges benchmark metadata with quality scores:

```
model,backend,frame_label,latency_s,...,composite_score,spatial_quality_score,...
wan_1.3b,baseline,1x,165.4,...,72.3,70.0,...
```

#### CSV options

| Option | Default | Description |
|--------|---------|-------------|
| `--from-csv PATH` | — | CSV file with video paths and metadata |
| `--video-col NAME` | `video_path` | Column containing video file paths |
| `--prompt-col NAME` | — | Column containing per-video prompts (overrides `--prompt`) |
| `--status-col NAME` | `status` | Column to filter rows by |
| `--status-ok VALUE` | `ok` | Value in status column that means "process this row" |

Video paths in the CSV are resolved relative to the CSV file's parent directory. Rows where the status column does not match `--status-ok` are skipped, as are rows pointing to missing files.

### Calibrate normalization on your own dataset

```bash
vqeval calibrate --dataset ./my_videos/ --output custom_presets.json
vqeval evaluate video.mp4 --presets custom_presets.json
```

### CPU-only mode

```bash
vqeval evaluate video.mp4 --device cpu
```

## Quality Dimensions

Each dimension produces a score from 0 (worst) to 100 (best) plus detailed metrics.

### 1. Spatial Quality

Per-frame image quality using BRISQUE, NIQE, CLIP-IQA, and Laplacian blur detection. Frames are sampled adaptively: all frames for videos under 5s, uniform 2 fps+ sampling for longer videos.

### 2. Temporal Coherence

Measures consistency across frames: CLIP-based flickering detection, RAFT optical flow motion smoothness, DINOv2 subject consistency, CLIP background consistency, and ORB keypoint warping detection.

### 3. Repetition Detection (loop_quality)

Detects unwanted repetitive/cyclic motion — a common AI video failure mode where the model falls into an "attention sink" and the video degenerates into a GIF-like repeating pattern. A video that loops or repeats itself scores **low** (bad). Analyzed via:

- **Self-similarity matrix**: CLIP embedding similarity across all frame pairs, detecting periodic diagonal stripe patterns
- **Near-duplicate detection**: identifies frames far apart in time that are nearly identical
- **Optical flow periodicity**: FFT analysis of flow magnitudes to detect repeating motion cycles
- **Temporal autocorrelation**: checks if the sequence of frame-to-frame changes repeats itself

### 4. Artifact Detection

AI-specific artifacts: banding/posterization (CAMBI-like analysis), anatomical errors via MediaPipe (impossible joint configurations, extra/missing fingers), texture tiling via autocorrelation, edge halo detection, and temporal noise anomaly analysis.

### 5. Dynamic Quality

Evaluates meaningful motion: dynamic degree (percentage of pixels with significant flow), motion diversity (flow direction entropy), and aesthetic quality via CLIP-IQA.

### 6. Text-Video Alignment

Only evaluated when `--prompt` is provided. Computes per-frame CLIP text-video cosine similarity and detects temporal drift (declining alignment over time).

## Output Format

### JSON report structure

```json
{
  "video_path": "video.mp4",
  "video_meta": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "duration": 5.0,
    "codec": "h264",
    "total_frames": 150,
    "sampled_frames": 150
  },
  "composite_score": 72.3,
  "composite_verdict": "good",
  "weights": {
    "spatial_quality": 0.22,
    "temporal_coherence": 0.28,
    "loop_quality": 0.17,
    "artifact_detection": 0.22,
    "dynamic_quality": 0.11
  },
  "dimensions": {
    "spatial_quality": {
      "score": 72.0,
      "verdict": "good",
      "brisque_mean": 28.4,
      "niqe_mean": 4.2,
      "clip_iqa_mean": 0.68,
      "blur_frame_pct": 3.2
    },
    "temporal_coherence": {
      "score": 65.0,
      "verdict": "fair",
      "flickering_score": 78.0,
      "motion_smoothness": 61.0,
      "subject_consistency": 0.92,
      "background_consistency": 0.96,
      "warping_events": 3
    },
    "loop_quality": {
      "score": 85.0,
      "verdict": "good",
      "repetition_severity": 0.15,
      "cycle_detected": false,
      "cycle_period_frames": 0,
      "duplicate_frame_ratio": 0.02,
      "flow_periodicity": 0.1
    }
  },
  "elapsed_seconds": 12.5
}
```

### Composite scoring

The overall score is a weighted average of active dimension scores. Default weights:

| Dimension | Weight | Notes |
|-----------|--------|-------|
| Spatial quality | 0.20 | Always active |
| Temporal coherence | 0.25 | Always active |
| Repetition detection | 0.15 | Always active |
| Artifact detection | 0.20 | Always active |
| Dynamic quality | 0.10 | Always active |
| Text alignment | 0.10 | Only with `--prompt` |

Weights are automatically redistributed when text alignment is inactive (no prompt provided).

### Verdicts

| Score | Verdict |
|-------|---------|
| 90-100 | excellent |
| 70-89 | good |
| 50-69 | fair |
| 30-49 | poor |
| 0-29 | bad |

## Figures

When `--figures <dir>` is provided, VQeval generates publication-quality plots (300 DPI) summarizing the evaluation results.

### Single-video figures

| Figure | Description |
|--------|-------------|
| Radar chart | Overall dimension scores on a single polar plot |
| Dimension bars | Horizontal bar chart of each dimension score with verdict coloring |
| Temporal profile | Frame-level quality metrics over time |
| Self-similarity heatmap | CLIP embedding similarity across all frame pairs (used by repetition detection) |
| Coherence profile | Per-frame temporal coherence metrics (flickering, motion smoothness) |
| Text alignment drift | Per-frame CLIP text-video similarity over time (only with `--prompt`) |
| Motion profile | Optical flow magnitude and direction entropy per frame |

### Batch-mode figures

In batch mode, the following summary figures are generated in addition to per-video figures (saved in subdirectories named after each video):

| Figure | Description |
|--------|-------------|
| Ranking | Horizontal bar chart ranking all videos by composite score |
| Dimension comparison | Grouped bar chart comparing dimension scores across videos |
| Distributions | Box plots showing score distributions for each dimension |
| Grouped comparison | Model x backend composite score chart (only with `--from-csv` when metadata includes `model` and `backend` columns) |

When using `--from-csv`, figure labels automatically use metadata (e.g., `wan_1.3b/baseline/1x`) instead of truncated filenames.

### Supported formats

Use `--figure-format` to control the output format:

| Value | Description |
|-------|-------------|
| `png` | PNG images (default) |
| `pdf` | PDF vector graphics |
| `both` | Both PNG and PDF |

All figures are rendered at 300 DPI, suitable for publication and print.

## Architecture

```
vqeval/
├── cli.py                    # Entry point, argument parsing
├── core/
│   ├── pipeline.py           # Orchestrates evaluation pipeline
│   ├── video_loader.py       # Decoding, frame sampling, caching
│   ├── report.py             # Report generation (JSON, HTML, CSV)
│   └── config.py             # Weights, thresholds, normalization params
├── evaluators/
│   ├── base.py               # Abstract base class + registry
│   ├── spatial_quality.py    # Dimension 1: per-frame quality
│   ├── temporal_coherence.py # Dimension 2: cross-frame consistency
│   ├── loop_quality.py       # Dimension 3: repetition detection
│   ├── artifact_detection.py # Dimension 4: AI-specific artifacts
│   ├── dynamic_quality.py    # Dimension 5: motion and aesthetics
│   └── text_alignment.py     # Dimension 6: prompt alignment
├── figures/
│   ├── single.py             # Per-video figure generation
│   └── batch.py              # Batch summary figure generation
├── models/
│   └── model_registry.py     # Lazy-load shared models
├── normalization/
│   ├── calibration.py        # Normalization + calibration tools
│   └── presets.json          # Default normalization ranges
└── reports/
    └── templates/            # HTML report templates (Jinja2)
```

### Shared model management

Several evaluators share the same underlying models. The `ModelRegistry` singleton lazy-loads each model on first request and caches it:

| Model | Used by | Approx VRAM |
|-------|---------|-------------|
| CLIP ViT-B/16 (open_clip) | Flickering, BG consistency, repetition detection, text alignment | ~400 MB |
| DINOv2 ViT-B/14 (torch.hub) | Subject consistency | ~350 MB |
| OpenCV Farneback | Motion smoothness, flow periodicity, dynamic degree | CPU only |
| LPIPS (AlexNet) | Perceptual distance metrics | ~100 MB |
| MediaPipe Pose/Hands | Anatomical error detection | ~100 MB |

Peak VRAM stays under ~2 GB for inference models.

### Frame sampling and caching

- **Decode once**: `VideoLoader` decodes the video into a shared numpy array. All evaluators read from this buffer.
- **Embedding cache**: CLIP and DINO frame embeddings are computed once and cached on the `VideoData` object for reuse across evaluators.
- **Flow cache**: Optical flow fields computed by one evaluator are available to others.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests cover configuration, normalization, report generation, evaluator registration, and CLI parsing. Tests that require GPU models (video loader, full pipeline) need `torch` and a CUDA device.

## Supported formats

- **Video**: MP4, MOV, WebM, AVI, MKV (H.264, H.265, VP9, AV1 codecs)
- **Resolution**: up to 4K (optimized for 720p-1080p)

## License

MIT
