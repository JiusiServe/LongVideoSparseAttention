"""
cogvideox_generate.py — Video generation with CogVideoX, supporting single and multi-GPU inference.
========================================================================================================

Usage
-----
# Single GPU — baseline (no LVSA)
python cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720

# Single GPU with LVSA
python cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720 --lvsa

# Single GPU with LVSA + FlashInfer
python cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720 --lvsa --flashinfer

# Multi-GPU context-parallel (requires torchrun)
torchrun --nproc_per_node=2 cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720 --lvsa
"""

import os
import time
import argparse

import torch
import torch.distributed as dist

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from lvsa.adapters.cogvideox import CogVideoXAdapter
from lvsa.device import (
    get_device,
    get_distributed_backend,
    max_memory_allocated,
    mem_get_info,
)
from lvsa.parallel import (
    patch_rotary_emb_for_context_parallel,
    install_lvsa_processors,
    compute_and_validate_seq_len,
    setup_context_parallel,
)


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CogVideoX generation with optional LVSA and multi-GPU context parallelism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="PATH",
        help="Path or HuggingFace Hub ID of the CogVideoX pipeline "
        "(e.g. THUDM/CogVideoX-2b, THUDM/CogVideoX-5b).",
    )

    # ── Prompt ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the video to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for quality filtering.",
    )

    # ── Video dimensions ──────────────────────────────────────────────────────
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Frames to generate (default 49 for CogVideoX native).",
    )
    parser.add_argument("--height", type=int, default=480, help="Frame height (px).")
    parser.add_argument("--width", type=int, default=720, help="Frame width (px).")

    # ── Sampling ──────────────────────────────────────────────────────────────
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=6.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=int, default=8, help="Output FPS.")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for the generated video. Created if missing.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename inside --output-dir. If omitted, a descriptive "
        "name encoding model, geometry, backend, and run parameters is "
        "auto-generated. Extension (.mp4) is appended automatically if missing.",
    )

    # ── LVSA ───────────────────────────────────────────────────────────────────
    lvsa = parser.add_argument_group(
        "Sliding Window Attention (LVSA)",
        "Block-sparse attention to reduce memory for long videos. "
        "Add --lvsa to enable; all other LVSA flags are ignored otherwise.",
    )
    lvsa.add_argument("--lvsa", action="store_true", help="Enable LVSA (block-sparse attention).")
    lvsa.add_argument("--window-size", type=int, default=12, help="LVSA window half-width (video frames).")
    lvsa.add_argument("--n-first-frames", type=int, default=4, help="Leading global context frames.")
    lvsa.add_argument("--key-frame-interval", type=int, default=16, help="Periodic keyframe interval (video frames).")
    lvsa.add_argument("--sparsity-scale", type=float, default=1.0,
                     help="Scale factor for the attention sparsity budget. "
                     "<1.0 = more sparse, >1.0 = less sparse. Default 1.0.")
    lvsa.add_argument("--auto-keyframes", action="store_true", help="Auto-compute key-frame-interval.")
    lvsa.add_argument("--flashinfer", action="store_true", help="Use FlashInfer block-sparse attention.")
    lvsa.add_argument("--show-mask", action="store_true", help="Print attention mask.")
    lvsa.add_argument("--show-mask-compact", nargs="?", const="once", default=None, choices=["once", "step"])
    lvsa.add_argument("--rotate-keyframes", action="store_true", help="Rotate keyframes each step.")

    # ── Profiling ─────────────────────────────────────────────────────────────
    parser.add_argument("--profile", action="store_true", help="Log per-step timing.")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Distributed vs single-GPU detection ────────────────────────────────────
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(get_distributed_backend())
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        rank = 0
        world = 1

    device = get_device(rank)

    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
    torch.backends.cuda.matmul.allow_tf32 = True

    if rank == 0:
        mode = f"distributed (world_size={world})" if distributed else "single-GPU"
        print(f"[init] {mode}  device={device}")

    # ── Create model adapter ─────────────────────────────────────────────────
    adapter = CogVideoXAdapter()

    # ── Patch standard rotary BEFORE loading (multi-GPU only) ─────────────────
    if world > 1:
        patch_rotary_emb_for_context_parallel(adapter, rank, world)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    t0 = time.time()
    if rank == 0:
        print(f"[model] loading from {args.model} ...")

    pipe = CogVideoXPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # ── Enable VAE tiling to avoid OOM during decode ─────────────────────────
    pipe.vae.enable_tiling()
    if rank == 0:
        print("[vae] tiling enabled")

    # ── Sequence-length validation ────────────────────────────────────────────
    compute_and_validate_seq_len(
        args.num_frames,
        args.height,
        args.width,
        pipe.transformer.config,
        pipe.vae_scale_factor_temporal,
        getattr(pipe, "vae_scale_factor_spatial", 8),
        world,
        rank,
    )

    # ── LVSA processor installation ────────────────────────────────────────────
    lvsa_processor = None
    if args.lvsa:
        lvsa_processor = install_lvsa_processors(
            pipe, args, rank, world, adapter,
            sparsity_scale=args.sparsity_scale,
        )
    elif rank == 0:
        print("[attn] using standard full attention (no LVSA)")

    # ── Context-parallel plan (multi-GPU only) ────────────────────────────────
    if world > 1:
        setup_context_parallel(adapter, pipe.transformer, world)

    if rank == 0:
        print(f"[model] loaded in {time.time() - t0:.1f}s")

    # ── Generate ──────────────────────────────────────────────────────────────
    if rank == 0:
        print(
            f"[generate] {args.num_frames} frames  "
            f"{args.height}x{args.width}  "
            f"{args.steps} steps  guidance={args.guidance}  seed={args.seed}"
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_gen = time.time()

    # ── Rotating keyframes / profiling ─────────────────────────────────────
    if lvsa_processor is not None or args.profile:
        if rank == 0 and lvsa_processor:
            print("[LVSA] windowed attention for all steps")

        _step_counter = [0]
        _step_times: list = []
        _orig_scheduler_step = pipe.scheduler.step

        def _hooked_scheduler_step(*s_args, **s_kwargs):
            step_index = _step_counter[0]

            if lvsa_processor is not None:
                if args.rotate_keyframes:
                    lvsa_processor.set_step(step_index)

                if rank == 0 and args.show_mask_compact == "step":
                    print(f"\n[LVSA-WINDOW] step {step_index}:")
                    lvsa_processor.print_attention_mask_compact()

            if args.profile and rank == 0:
                now = time.time()
                _step_times.append(now)
                if len(_step_times) > 1:
                    dt = now - _step_times[-2]
                    print(f"[profile] step {step_index}: {dt:.3f}s")

            result = _orig_scheduler_step(*s_args, **s_kwargs)
            _step_counter[0] += 1
            return result

        pipe.scheduler.step = _hooked_scheduler_step

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).frames[0]

    gen_duration = time.time() - t_gen
    if rank == 0:
        print(f"[generate] done in {gen_duration:.1f}s")

    mem_mb = max_memory_allocated() / 1024**2
    print(f"[rank{rank}][mem] peak allocated: {mem_mb:.0f} MB")

    # ── Save (rank 0 only) ────────────────────────────────────────────────────
    if rank == 0:
        mem_mb = max_memory_allocated() / 1024**2
        if args.lvsa:
            backend = "flashinfer" if args.flashinfer else "sdpa"
            kfi_tag = "auto" if args.auto_keyframes else str(args.key_frame_interval)
            rot_tag = "_rot" if args.rotate_keyframes else ""
            ring_tag = ""
            lvsa_tag = (
                f"_lvsa_w{args.window_size}_f{args.n_first_frames}"
                f"_kfi{kfi_tag}{rot_tag}{ring_tag}_{backend}"
            )
        else:
            lvsa_tag = "_fullatt"
        if args.output_name:
            filename = args.output_name
            if not filename.endswith(".mp4"):
                filename += ".mp4"
        else:
            stem = os.path.basename(__file__).split(".")[0]
            model_tag = os.path.basename(args.model)
            prompt_tag = args.prompt.replace(" ", "_")[:30]
            filename = (
                f"{stem}"
                f"_{model_tag}"
                f"_gpu{world}"
                f"_{args.height}x{args.width}@{args.fps}"
                f"_frames{args.num_frames}"
                f"{lvsa_tag}"
                f"_steps{args.steps}_cfg{args.guidance}"
                f"_seed{args.seed}"
                f"_dur{gen_duration:.0f}s_mem{mem_mb:.0f}MB"
                f"_{prompt_tag}"
                f".mp4"
            )
        out_path = os.path.join(args.output_dir, filename)
        os.makedirs(args.output_dir, exist_ok=True)

        t_exp = time.time()
        export_to_video(output, out_path, fps=args.fps)
        print(f"[export] {out_path}  ({time.time() - t_exp:.1f}s)")

        free, total = mem_get_info(device)
        print(
            f"[mem] peak allocated: {mem_mb:.0f} MB  |  current used: {(total - free) / 1024**2:.0f} MB"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
