"""Standalone LVSA generation for NVIDIA Cosmos 3.0 (diffusers).

Example:
  python examples/cosmos_generate.py --model /data/Cosmos3-Nano \
      --prompt "A dog running in the forest." --num-frames 189 \
      --height 720 --width 1280 --steps 35 --lvsa --output-name cosmos_1x

Requires diffusers main (>=0.39.0.dev0) for Cosmos3OmniPipeline.

Note on output attribute: Cosmos3OmniPipelineOutput uses `.video` (not `.frames`).
The pipeline's video_processor.postprocess_video already indexes batch dim 0, so
out.video is a list[PIL.Image] (for default output_type="pil") — the frames directly.
"""
import argparse
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video

from lvsa.cosmos3 import install_cosmos3_lvsa


def main():
    ap = argparse.ArgumentParser(
        description="Cosmos 3.0 video generation with optional LVSA sparse attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True, metavar="PATH",
                    help="Path or HuggingFace Hub ID of the Cosmos3OmniPipeline model.")
    ap.add_argument("--prompt", default="A dog running in the forest.",
                    help="Text prompt describing the video to generate.")
    ap.add_argument("--num-frames", type=int, default=189,
                    help="Number of frames to generate.")
    ap.add_argument("--height", type=int, default=720, help="Frame height (px).")
    ap.add_argument("--width", type=int, default=1280, help="Frame width (px).")
    ap.add_argument("--steps", type=int, default=35, help="Denoising steps.")
    ap.add_argument("--guidance", type=float, default=6.0, help="CFG scale.")
    ap.add_argument("--seed", type=int, default=16, help="Random seed.")
    ap.add_argument("--fps", type=int, default=16, help="Output FPS.")
    ap.add_argument("--lvsa", action="store_true",
                    help="Install LVSA block-sparse attention (else dense baseline).")
    ap.add_argument("--window-size", type=int, default=12,
                    help="Half-width of the LVSA sliding window in video frames.")
    ap.add_argument("--n-first-frames", type=int, default=4,
                    help="Number of leading video frames always included as global context.")
    ap.add_argument("--sparsity-scale", type=float, default=1.0,
                    help="Scale factor for the attention sparsity budget. "
                         "<1.0 = more sparse, >1.0 = less sparse.")
    ap.add_argument("--output-dir", type=Path, default=Path("out/adhoc"),
                    help="Output directory for the generated video. Created if missing.")
    ap.add_argument("--output-name", required=True,
                    help="Output filename inside --output-dir. Extension .mp4 appended if missing.")
    args = ap.parse_args()

    print(f"[init] loading {args.model} ...")
    # enable_safety_checker=False prevents the pipeline __init__ from constructing
    # CosmosSafetyChecker (which requires the external `cosmos_guardrail` package).
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, enable_safety_checker=False
    ).to("cuda")

    if args.lvsa:
        install_cosmos3_lvsa(
            pipe.transformer,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            window_size=args.window_size,
            n_first_frames=args.n_first_frames,
            sparsity_scale=args.sparsity_scale,
        )
    else:
        print("[attn] dense (no LVSA)")

    print(
        f"[generate] {args.num_frames} frames  "
        f"{args.height}x{args.width}  "
        f"{args.steps} steps  guidance={args.guidance}  seed={args.seed}"
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        enable_safety_check=False,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    )
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**2

    # Cosmos3OmniPipelineOutput exposes .video (not .frames).
    # video_processor.postprocess_video already strips the batch dim, so
    # out.video is list[PIL.Image] directly — no [0] needed.
    frames = out.video

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name if args.output_name.endswith(".mp4") else args.output_name + ".mp4"
    out_path = args.output_dir / name
    export_to_video(frames, str(out_path), fps=args.fps)
    print(f"[cosmos] wrote {out_path} ({len(frames)} frames)")
    print(f"[BENCH] gen_s={wall:.2f} peak_mb={peak:.0f} steps={args.steps} frames={len(frames)}")


if __name__ == "__main__":
    main()
