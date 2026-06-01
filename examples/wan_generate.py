"""
wan_generate.py — Video generation with Wan 2.x, supporting single and multi-GPU inference.
=================================================================================================

Usage
-----
# Single GPU (no torchrun needed)
python wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A dog running in the forest." \\
    --num-frames 81 --height 480 --width 832

# Single GPU with LVSA
python wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A dog running in the forest." \\
    --num-frames 81 --height 480 --width 832 --lvsa

# Multi-GPU context-parallel (requires torchrun)
torchrun --nproc_per_node=2 wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A dog running in the forest." \\
    --num-frames 81 --height 480 --width 832 --lvsa

# Multi-GPU, 4 GPUs, 481 frames at 720p with LVSA
torchrun --nproc_per_node=4 wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A timelapse of a blooming flower." \\
    --num-frames 481 --height 720 --width 1280 \\
    --lvsa --window-size 32 --n-first-frames 16 --key-frame-interval 16
"""

import os
import time
import argparse

import torch
import torch.distributed as dist

from diffusers import WanPipeline
from diffusers.utils import export_to_video

from lvsa.adapters.wan import WanAdapter
from lvsa.parallel import (
    patch_rotary_emb_for_context_parallel,
    install_lvsa_processors,
    compute_and_validate_seq_len,
    setup_context_parallel,
)


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan video generation with optional LVSA and multi-GPU context parallelism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="PATH",
        help="Path or HuggingFace Hub ID of the Wan pipeline "
        "(e.g. /models/Wan2.2-T2V-A14B-Diffusers).",
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
        default=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
        help="Negative prompt (default: standard Wan quality filter).",
    )

    # ── Video dimensions ──────────────────────────────────────────────────────
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Frames to generate. Must be on the 4k+1 grid (81, 121, 161, …).",
    )
    parser.add_argument("--height", type=int, default=480, help="Frame height (px).")
    parser.add_argument("--width", type=int, default=832, help="Frame width (px).")

    # ── Sampling ──────────────────────────────────────────────────────────────
    parser.add_argument("--steps", type=int, default=40, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=16, help="Random seed.")
    parser.add_argument("--fps", type=int, default=16, help="Output FPS.")

    # ── Loading ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Load pipeline with device_map='balanced' to spread model layers "
        "across all visible GPUs. Useful to fit large models without CP.",
    )

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
    lvsa.add_argument(
        "--lvsa",
        action="store_true",
        help="Enable WanDistributedLVSAProcessor (block-sparse attention).",
    )

    lvsa.add_argument(
        "--window-size",
        type=int,
        default=12,
        help="Half-width of the LVSA sliding window in *video* frames. "
        "Converted to latent frames internally.",
    )
    lvsa.add_argument(
        "--n-first-frames",
        type=int,
        default=4,
        help="Number of leading video frames always included as global context. "
        "Converted to latent frames internally.",
    )
    lvsa.add_argument(
        "--key-frame-interval",
        type=int,
        default=16,
        help="Interval (in video frames) between periodic keyframes used as "
        "global context. 0 disables periodic keyframes beyond n-first-frames. "
        "Converted to latent frames internally; auto-adjusted if too small. "
        "Ignored when --auto-keyframes is set.",
    )
    lvsa.add_argument(
        "--sparsity-scale",
        type=float,
        default=1.0,
        help="Scale factor for the attention sparsity budget. "
        "<1.0 makes attention more sparse (fewer attended frames), "
        ">1.0 makes it less sparse (more attended frames). "
        "Default 1.0 preserves the original behaviour.",
    )
    lvsa.add_argument(
        "--auto-keyframes",
        action="store_true",
        help="Automatically compute key-frame-interval so that total attended "
        "frames per query approximates 21 (the reference budget). "
        "Overrides --key-frame-interval.",
    )
    lvsa.add_argument(
        "--flashinfer",
        action="store_true",
        help="Use FlashInfer BlockSparseAttentionWrapper for LVSA instead of "
        "per-frame SDPA. Requires flashinfer to be installed.",
    )
    lvsa.add_argument(
        "--show-mask",
        action="store_true",
        help="Print the T×T attention mask matrix showing which latent frames "
        "each query frame attends to (G=global, W=window, X=both). "
        "Useful for debugging LVSA patterns. Requires --lvsa.",
    )
    lvsa.add_argument(
        "--show-mask-compact",
        nargs="?",
        const="once",
        default=None,
        choices=["once", "step"],
        help="Compact 1-char-per-column attention mask. "
        "'once' (default if flag given alone) prints at init. "
        "'step' prints at every denoising step (useful with --rotate-keyframes).",
    )
    lvsa.add_argument(
        "--rotate-keyframes",
        action="store_true",
        help="Shift periodic keyframes by 1 position each denoising step, "
        "cycling through all positions over key_frame_interval steps. "
        "This ensures every frame acts as a global anchor at some point.",
    )
    # ── RIFLEx (training-free length extrapolation via RoPE) ──────────────────
    riflex = parser.add_argument_group(
        "RIFLEx (arXiv 2502.15894)",
        "Training-free temporal RoPE rescaling. Orthogonal to LVSA; can be "
        "combined with --lvsa to stack with LVSA. Use --riflex-s > 1.0 when "
        "generating beyond the training horizon.",
    )
    riflex.add_argument(
        "--riflex",
        action="store_true",
        help="Apply RIFLEx RoPE rescaling before generation.",
    )
    riflex.add_argument(
        "--riflex-s",
        type=float,
        default=1.0,
        help="Extrapolation ratio s. s=1 is a no-op; s=2 targets 2x training "
        "length, etc.",
    )
    riflex.add_argument(
        "--riflex-k",
        type=int,
        default=None,
        help="Override auto-detected temporal frequency index k. Auto-detected "
        "if omitted (argmin_j |period_j - L|).",
    )
    riflex.add_argument(
        "--riflex-train-len",
        type=int,
        default=None,
        help="Training latent-frame count L. Defaults to the adapter's "
        "reference (21 for Wan 1.3B).",
    )

    # ── Profiling ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Log per-step wall-clock timing for profiling attention phases.",
    )

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
        device_count,
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
    adapter = WanAdapter()

    # ── Patch standard rotary BEFORE loading (multi-GPU only) ─────────────────
    if world > 1:
        patch_rotary_emb_for_context_parallel(adapter, rank, world)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    t0 = time.time()
    if rank == 0:
        print(f"[model] loading from {args.model} ...")

    if args.balanced:
        pipe = WanPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        if rank == 0:
            print(f"[model] device_map: {pipe.hf_device_map}")
    else:
        pipe = WanPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to(device)

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

    # ── Warn about LVSA-dependent flags used without --lvsa ────────────────────
    if not args.lvsa:
        lvsa_deps = []
        if args.flashinfer:
            lvsa_deps.append("--flashinfer")
        if args.window_size != 12:
            lvsa_deps.append("--window-size")
        if args.n_first_frames != 4:
            lvsa_deps.append("--n-first-frames")
        if args.key_frame_interval != 16:
            lvsa_deps.append("--key-frame-interval")
        if args.auto_keyframes:
            lvsa_deps.append("--auto-keyframes")
        if args.show_mask:
            lvsa_deps.append("--show-mask")
        if args.show_mask_compact:
            lvsa_deps.append("--show-mask-compact")
        if args.rotate_keyframes:
            lvsa_deps.append("--rotate-keyframes")
        if lvsa_deps and rank == 0:
            print(f"[WARNING] {', '.join(lvsa_deps)} ignored without --lvsa")

    # ── RIFLEx RoPE rescaling (independent of LVSA; apply before LVSA install) ──
    if args.riflex:
        from lvsa.riflex import apply_riflex_to_wan_pipe
        info = apply_riflex_to_wan_pipe(
            pipe,
            s=args.riflex_s,
            k=args.riflex_k,
            training_length=args.riflex_train_len,
        )
        if rank == 0:
            if info["applied"]:
                print(
                    f"[riflex] applied: s={info['s']}, k={info['k']}, "
                    f"L={info['training_length']}, t_dim={info['t_dim']}"
                )
            else:
                print(f"[riflex] no-op (s={info['s']}, k={info['k']})")
    elif rank == 0 and (args.riflex_s != 1.0 or args.riflex_k is not None
                        or args.riflex_train_len is not None):
        print("[WARNING] --riflex-s/--riflex-k/--riflex-train-len ignored "
              "without --riflex")

    # ── LVSA processor installation (works on both single and multi-GPU) ───────
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
            f"{args.height}×{args.width}  "
            f"{args.steps} steps  guidance={args.guidance}  seed={args.seed}"
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_gen = time.time()

    # ── Step callback for rotating keyframes / mask display / profiling
    step_callback = None

    need_callback = (
        (lvsa_processor and args.rotate_keyframes)
        or (lvsa_processor and args.show_mask_compact == "step")
        or args.profile
        or os.environ.get("LVSA_STEP_TIME_LOG", "0") == "1"
        or os.environ.get("LVSA_MEM_LOG", "0") == "1"
    )

    if need_callback:
        if rank == 0 and lvsa_processor:
            print("[LVSA] windowed attention for all steps")

        _step_times: list = []

        def step_callback(pipe_obj, step_index, timestep, callback_kwargs):
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

            # Opt-in per-step wall-clock log (LVSA_STEP_TIME_LOG=1). Emits the
            # same [LVSA-TIME] format as the vllm-omni plugin paths so the
            # plot script can consume either output uniformly.
            if rank == 0 and os.environ.get("LVSA_STEP_TIME_LOG", "0") == "1":
                import time as _time
                _now = _time.perf_counter()
                _last = getattr(step_callback, "_last_t", None)
                if _last is not None:
                    print(f"[LVSA-TIME] step={step_index - 1} dt={_now - _last:.3f}s", flush=True)
                step_callback._last_t = _now  # type: ignore[attr-defined]

            # Opt-in per-step memory log (LVSA_MEM_LOG=1, device-agnostic).
            if rank == 0 and os.environ.get("LVSA_MEM_LOG", "0") == "1":
                from lvsa.device import memory_stats
                _stats = memory_stats()
                if _stats is not None:
                    _kind, _dev, _alloc, _reserved, _peak = _stats
                    print(
                        f"[LVSA-MEM] step={step_index} {_kind}={_dev} "
                        f"alloc={_alloc:.2f}GB reserved={_reserved:.2f}GB peak={_peak:.2f}GB",
                        flush=True,
                    )

            return callback_kwargs

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        callback_on_step_end=step_callback,
    ).frames[0]

    gen_duration = time.time() - t_gen
    if rank == 0:
        print(f"[generate] done in {gen_duration:.1f}s")

    mem_mb = max_memory_allocated() / 1024**2
    print(f"[rank{rank}][mem] peak allocated: {mem_mb:.0f} MB")

    # ── Save (rank 0 only) ────────────────────────────────────────────────────
    if rank == 0:
        if args.balanced:
            mem_mb = sum(
                max_memory_allocated(di) / 1024**2
                for di in range(device_count())
            )
        else:
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
                f"_{'balanced' if args.balanced else f'gpu{world}'}"
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

        if args.balanced:
            total_peak = 0
            total_used = 0
            num_devices = device_count()
            for di in range(num_devices):
                peak_i = max_memory_allocated(di) / 1024**2
                free_i, total_i = mem_get_info(di)
                used_i = (total_i - free_i) / 1024**2
                total_peak += peak_i
                total_used += used_i
                print(f"[mem] gpu{di}: peak={peak_i:.0f} MB  used={used_i:.0f} MB")
            print(f"[mem] total: peak={total_peak:.0f} MB  used={total_used:.0f} MB")
        else:
            free, total = mem_get_info(device)
            print(
                f"[mem] peak allocated: {mem_mb:.0f} MB  |  current used: {(total - free) / 1024**2:.0f} MB"
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
