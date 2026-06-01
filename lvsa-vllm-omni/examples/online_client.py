#!/usr/bin/env python3
"""Send a video generation request to a running vllm-omni server with LVSA.

The server side is launched separately via:

    examples/serve_wan.sh /path/to/Wan2.1-T2V-1.3B-Diffusers      # Wan
    examples/serve_hunyuan.sh /path/to/HunyuanVideo-1.5-...       # HunyuanVideo

This client uses vllm-omni's async job API:
  1. POST /v1/videos    →  {id, status: "queued"}
  2. GET  /v1/videos/{id}        (poll until status == "completed" or "failed")
  3. GET  /v1/videos/{id}/content   (download the mp4)

Example:

    python examples/online_client.py \\
        --host localhost:8098 \\
        --prompt "A dog running in the forest." \\
        --frames 81 \\
        --size 832x480 \\
        --fps 16 --steps 40 --seed 42 \\
        --output out.mp4
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("This script needs `requests`: pip install requests", file=sys.stderr)
    sys.exit(2)


def submit(host: str, args: argparse.Namespace) -> str:
    """POST /v1/videos and return the job id."""
    url = f"http://{host}/v1/videos"
    form = {
        "prompt": args.prompt,
        "size": args.size,
        "fps": str(args.fps),
        "num_inference_steps": str(args.steps),
        "guidance_scale": str(args.guidance),
        "seed": str(args.seed),
    }
    # Wan 2.2 uses guidance_scale_2 for the low-noise CFG; ignored by other models.
    if args.guidance2 is not None:
        form["guidance_scale_2"] = str(args.guidance2)
    if args.flow_shift is not None:
        form["flow_shift"] = str(args.flow_shift)
    if args.boundary_ratio is not None:
        form["boundary_ratio"] = str(args.boundary_ratio)
    if args.negative_prompt:
        form["negative_prompt"] = args.negative_prompt
    # vllm-omni accepts either `seconds` or `num_frames` depending on version;
    # send the one the user provided.
    if args.num_frames is not None:
        form["num_frames"] = str(args.num_frames)
    elif args.seconds is not None:
        form["seconds"] = str(args.seconds)

    print(f"[client] POST {url}")
    for k, v in form.items():
        print(f"  {k} = {v}")
    r = requests.post(url, data=form, timeout=60)
    r.raise_for_status()
    body = r.json()
    job_id = body.get("id") or body.get("video_id") or body.get("request_id")
    if not job_id:
        raise RuntimeError(f"No id in response: {body}")
    print(f"[client] job_id = {job_id}  status = {body.get('status')}")
    return job_id


def poll(host: str, job_id: str, poll_s: float, max_s: float) -> dict:
    """Poll GET /v1/videos/{id} until completed or failed."""
    url = f"http://{host}/v1/videos/{job_id}"
    t0 = time.time()
    last_status = None
    while True:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        body = r.json()
        status = body.get("status")
        if status != last_status:
            elapsed = time.time() - t0
            print(f"[client] t={elapsed:5.1f}s  status = {status}")
            last_status = status
        if status == "completed":
            return body
        if status == "failed":
            raise RuntimeError(f"Server reported failure: {body}")
        if time.time() - t0 > max_s:
            raise TimeoutError(f"Job did not complete in {max_s}s")
        time.sleep(poll_s)


def download(host: str, job_id: str, output: Path) -> None:
    url = f"http://{host}/v1/videos/{job_id}/content"
    print(f"[client] GET {url}")
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
    print(f"[client] wrote {output} ({output.stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:8098",
                    help="vllm-omni server host:port (default localhost:8098)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--num-frames", type=int, default=None,
                    help="Number of frames to generate. Mutually exclusive with --seconds.")
    ap.add_argument("--seconds", type=float, default=None,
                    help="Video duration in seconds. Mutually exclusive with --num-frames.")
    ap.add_argument("--size", default="832x480", help="WIDTHxHEIGHT (default 832x480)")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.0,
                    help="CFG scale (Wan 2.2 high-noise stage, HunyuanVideo single stage)")
    ap.add_argument("--guidance2", type=float, default=None,
                    help="Wan 2.2 low-noise CFG (omit for HunyuanVideo)")
    ap.add_argument("--flow-shift", type=float, default=None,
                    help="Flow guidance scaling (e.g. 5.0 for HV 480p)")
    ap.add_argument("--boundary-ratio", type=float, default=None,
                    help="Attention window ratio (Wan 2.2 only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=3600,
                    help="Max wait for job completion (seconds)")
    ap.add_argument("--output", type=Path, default=Path("out.mp4"))
    args = ap.parse_args()

    if args.num_frames is None and args.seconds is None:
        args.num_frames = 81  # Wan training horizon default

    job_id = submit(args.host, args)
    body = poll(args.host, job_id, args.poll_interval, args.timeout)
    download(args.host, job_id, args.output)
    print("[client] done")


if __name__ == "__main__":
    main()
