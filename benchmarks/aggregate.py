#!/usr/bin/env python3
"""Aggregate per-video VQeval + VBench-Long JSONs into one summary CSV.

Walks <PAPER_OUTDIR> looking for `<tag>.vqeval.json` and `<tag>.vbench.json`
files. Parses the canonical naming `<model>__<backend>__<horizon>__<prompt>`
to recover the sweep metadata, joins with metrics, and writes a tidy CSV.

Output: <PAPER_OUTDIR>/_summary.csv with columns:
    model, backend, horizon, prompt_tag,
    vq_composite, vq_spatial, vq_temporal, vq_loop,
    vq_artifacts, vq_dynamic, vq_text_alignment,
    vbench_subject, vbench_temporal, vbench_motion,
    vbench_background, vbench_imaging, vbench_avg,
    wall_time_s, peak_gpu_gb

Plus a roll-up `_summary_means.csv` with mean ± std across the prompts for
every (model, backend, horizon) cell.
"""
from __future__ import annotations
import csv
import json
import re
import statistics
import sys
from pathlib import Path


def parse_tag(stem: str) -> dict | None:
    """`wan13b__lvsa__3x__dog_forest` → dict"""
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    return {"model": parts[0], "backend": parts[1], "horizon": parts[2], "prompt_tag": parts[3]}


def parse_log_summary(log_path: Path) -> dict:
    """Pull wall time + peak GB from the generation log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    wall = None
    peak = None
    m = re.search(r"[Dd]one in ([\d.]+)s", text)
    if m:
        wall = float(m.group(1))
    else:
        # Fallback: parse tqdm progress bar from the diffusion loop ending like
        # "50/50 [12:21<00:00, 14.8s/it]" — extract HH:MM:SS. Anchored on the
        # inference-step count (50) explicitly so we don't pick up earlier
        # progress bars (e.g. checkpoint loading).
        # NOTE: assumes --steps 50 (the SotA spec).
        flat = text.replace("\r", "\n")
        m = re.search(r"50/50 \[(?:(\d+):)?(\d+):(\d+)<", flat)
        if m:
            h_str, m_str, s_str = m.groups()
            h = int(h_str) if h_str else 0
            wall = float(h * 3600 + int(m_str) * 60 + int(s_str))
    m = re.search(r"Peak GPU memory.*?(\d+\.?\d*)\s*GB|peak allocated:\s*(\d+)\s*MB", text)
    if m:
        if m.group(1):
            peak = float(m.group(1))
        elif m.group(2):
            peak = float(m.group(2)) / 1024.0  # MB → GB
    return {"wall_time_s": wall, "peak_gpu_gb": peak}


def load_vqeval(path: Path) -> dict:
    """Pull the 6 dimensions + composite out of a vqeval JSON."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out = {}
    if "composite_score" in d:
        out["vq_composite"] = d["composite_score"]
    elif "composite" in d:
        out["vq_composite"] = d["composite"]
    dims = d.get("dimensions", {})
    for src, dst in [
        ("spatial_quality", "vq_spatial"),
        ("temporal_coherence", "vq_temporal"),
        ("loop_quality", "vq_loop"),
        ("artifact_detection", "vq_artifacts"),
        ("dynamic_quality", "vq_dynamic"),
        ("text_alignment", "vq_text_alignment"),
    ]:
        v = dims.get(src)
        if isinstance(v, dict):
            v = v.get("score")
        out[dst] = v
    return out


def load_vbench(path: Path) -> dict:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    s = d.get("scores", {})
    return {
        "vbench_subject": s.get("subject_consistency"),
        "vbench_temporal": s.get("temporal_flickering"),
        "vbench_motion": s.get("motion_smoothness"),
        "vbench_background": s.get("background_consistency"),
        "vbench_imaging": s.get("imaging_quality"),
        "vbench_avg": d.get("average"),
    }


def aggregate(outdir: Path):
    rows: list[dict] = []
    for mp4 in sorted(outdir.glob("*.mp4")):
        if mp4.stat().st_size == 0:
            continue  # placeholder
        meta = parse_tag(mp4.stem)
        if meta is None:
            continue
        row = dict(meta)
        row.update(parse_log_summary(mp4.with_suffix(".log")))
        row.update(load_vqeval(outdir / f"{mp4.stem}.vqeval.json"))
        row.update(load_vbench(outdir / f"{mp4.stem}.vbench.json"))
        rows.append(row)

    if not rows:
        print("No data found.", file=sys.stderr)
        return 1

    cols = [
        "model", "backend", "horizon", "prompt_tag",
        "vq_composite", "vq_spatial", "vq_temporal", "vq_loop",
        "vq_artifacts", "vq_dynamic", "vq_text_alignment",
        "vbench_subject", "vbench_temporal", "vbench_motion",
        "vbench_background", "vbench_imaging", "vbench_avg",
        "wall_time_s", "peak_gpu_gb",
    ]
    summary_csv = outdir / "_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Tidy CSV:   {summary_csv}  ({len(rows)} rows)")

    means_csv = outdir / "_summary_means.csv"
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["model"], r["backend"], r["horizon"])
        groups.setdefault(key, []).append(r)

    metric_cols = [c for c in cols if c.startswith(("vq_", "vbench_", "wall_", "peak_"))]
    with means_csv.open("w", newline="") as f:
        out_cols = ["model", "backend", "horizon", "n"] + sum(
            ([f"{c}_mean", f"{c}_std"] for c in metric_cols), []
        )
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for (m, b, h), items in sorted(groups.items()):
            row = {"model": m, "backend": b, "horizon": h, "n": len(items)}
            for c in metric_cols:
                vals = [it.get(c) for it in items if isinstance(it.get(c), (int, float))]
                if vals:
                    row[f"{c}_mean"] = statistics.mean(vals)
                    row[f"{c}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            w.writerow(row)
    print(f"Means CSV:  {means_csv}  ({len(groups)} groups)")
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Directory containing <tag>.{mp4,log,vqeval.json,vbench.json} files")
    args = ap.parse_args()
    sys.exit(aggregate(args.outdir))


if __name__ == "__main__":
    main()
