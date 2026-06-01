"""
hunyuan_generate.py — Video generation with HunyuanVideo-1.5, supporting single and multi-GPU inference.
================================================================================================================

**Status: experimental.**  The HunyuanVideoAdapter has not been tested end-to-end with actual
HunyuanVideo weights.  LVSA is applied only to single-stream blocks; dual-stream blocks use
standard full attention.

Usage
-----
# Single GPU (no torchrun needed)
python hunyuan_parallel_lvsa.py \\
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
    --prompt "A dog running in the forest." \\
    --num-frames 61 --height 480 --width 832

# Single GPU with LVSA
python hunyuan_parallel_lvsa.py \\
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
    --prompt "A dog running in the forest." \\
    --num-frames 61 --height 480 --width 832 --lvsa

# Multi-GPU context-parallel (requires torchrun)
torchrun --nproc_per_node=2 hunyuan_parallel_lvsa.py \\
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
    --prompt "A dog running in the forest." \\
    --num-frames 61 --height 480 --width 832 --lvsa
"""

import os
import time
import argparse

import torch
import torch.distributed as dist

from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

from lvsa.adapters.hunyuan_video import HunyuanVideoAdapter
from lvsa.parallel import (
    patch_rotary_emb_for_context_parallel,
    install_lvsa_processors,
    compute_and_validate_seq_len,
    setup_context_parallel,
)


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HunyuanVideo generation with optional LVSA and multi-GPU context parallelism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="PATH",
        help="Path or HuggingFace Hub ID of the HunyuanVideo pipeline "
        "(e.g. hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v).",
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
        default=61,
        help="Frames to generate (default 61 for ~4s at 15fps).",
    )
    parser.add_argument("--height", type=int, default=480, help="Frame height (px).")
    parser.add_argument("--width", type=int, default=832, help="Frame width (px).")

    # ── Sampling ──────────────────────────────────────────────────────────────
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=6.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=int, default=24, help="Output FPS.")

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
        "auto-generated. Extension (.mp4 or .pt for --output-latent) is "
        "appended automatically if missing.",
    )
    parser.add_argument(
        "--output-latent",
        action="store_true",
        help="Save the denoised latent tensor (.pt) instead of decoding to mp4. "
        "Use for long ratios where VAE decode OOMs; the latent can be decoded "
        "offline on higher-memory hardware.",
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
    from lvsa.device import (
        enable_fast_matmul,
        get_device,
        get_distributed_backend,
        max_memory_allocated,
        mem_get_info,
    )

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
    enable_fast_matmul()

    if rank == 0:
        mode = f"distributed (world_size={world})" if distributed else "single-GPU"
        print(f"[init] {mode}  device={device}")

    # ── Create model adapter ─────────────────────────────────────────────────
    adapter = HunyuanVideoAdapter()

    # ── Patch standard rotary BEFORE loading (multi-GPU only) ─────────────────
    if world > 1:
        patch_rotary_emb_for_context_parallel(adapter, rank, world)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    t0 = time.time()
    if rank == 0:
        print(f"[model] loading from {args.model} ...")

    pipe = HunyuanVideo15Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # ── Enable VAE tiling to avoid OOM during decode ─────────────────────────
    # HunyuanVideo's 3D VAE is very memory-hungry; without tiling, decoding
    # 121+ frames at 480×832 requires ~12 GB for a single conv3d layer.
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
        pipe.vae_scale_factor_spatial,
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
    # HunyuanVideo15Pipeline does not support callback_on_step_end, so we
    # hook into the scheduler's step() method to inject per-step logic.
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

    # HunyuanVideo-1.5 uses a guider (ClassifierFreeGuidance) instead of
    # guidance_scale at runtime.  Set the guidance scale on the guider.
    if hasattr(pipe, "guider") and pipe.guider is not None:
        pipe.guider = pipe.guider.new(guidance_scale=args.guidance)

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=generator,
    )
    if args.output_latent:
        pipe_kwargs["output_type"] = "latent"
        pipe_out = pipe(**pipe_kwargs)
        output = pipe_out.frames if hasattr(pipe_out, "frames") else pipe_out
        if hasattr(output, "__getitem__") and not torch.is_tensor(output):
            output = output[0]
    else:
        output = pipe(**pipe_kwargs).frames[0]

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
        ext = "pt" if args.output_latent else "mp4"
        if args.output_name:
            filename = args.output_name
            if not filename.endswith(f".{ext}"):
                filename += f".{ext}"
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
                f".{ext}"
            )
        out_path = os.path.join(args.output_dir, filename)
        os.makedirs(args.output_dir, exist_ok=True)

        t_exp = time.time()
        if args.output_latent:
            torch.save({"latent": output.detach().cpu(),
                        "num_frames": args.num_frames,
                        "height": args.height,
                        "width": args.width,
                        "prompt": args.prompt,
                        "seed": args.seed,
                        "steps": args.steps}, out_path)
            print(f"[save-latent] {out_path}  shape={tuple(output.shape)}  ({time.time() - t_exp:.1f}s)")
        else:
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
