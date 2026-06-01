#!/usr/bin/env python3
"""Offline (no HTTP server) Wan 2.x generation via vllm-omni's Python API.

Loads the model in-process, generates a video, and writes it to disk.
LVSA is engaged through env vars set at startup (see below).

Example:

    python examples/offline_wan.py \\
        --model /path/to/Wan2.1-T2V-1.3B-Diffusers \\
        --prompt "A dog running in the forest." \\
        --num-frames 81 --steps 40 \\
        --output-dir . --output-name dog_offline.mp4
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Diffusers checkpoint path")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--guidance2", type=float, default=None,
                    help="Wan 2.2 low-noise CFG (omit for Wan 2.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    # ── Output ──────────────────────────────────────────────────────────────
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--output-name", default="wan_offline.mp4")
    # ── LVSA toggles ────────────────────────────────────────────────────────
    ap.add_argument("--no-lvsa", action="store_true",
                    help="Disable LVSA (dense baseline).")
    ap.add_argument("--sparsity-scale", type=float, default=1.0)
    ap.add_argument("--no-rotate", action="store_true",
                    help="Disable rotating keyframes.")
    return ap.parse_args()


def _to_pil_frames(frames):
    """Normalize a heterogeneous frame container to list[PIL.Image].

    vllm-omni's OmniRequestOutput.images may be:
      - a torch.Tensor of shape [T, C, H, W] or [T, H, W, C], float [0,1]
        or uint8 [0,255]
      - a list/tuple of per-frame torch tensors / numpy arrays / PIL images
      - a list-of-lists when num_outputs_per_prompt > 1
    """
    import numpy as np
    import torch
    from PIL import Image

    def _frame_to_pil(f):
        if isinstance(f, Image.Image):
            return f.convert("RGB")
        if isinstance(f, torch.Tensor):
            arr = f.detach().cpu().float().numpy()
        elif isinstance(f, np.ndarray):
            arr = f
        else:
            raise TypeError(f"unsupported per-frame type: {type(f)}")
        # Squeeze leading singleton dims (vllm-omni often returns 5D
        # (1, 1, H, W, C) shapes per element).
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        # CHW → HWC if needed
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)
        # Dtype normalization
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 + 1e-3:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return Image.fromarray(arr).convert("RGB")

    # Unwrap top-level container variants
    if isinstance(frames, (list, tuple)):
        if len(frames) == 1 and isinstance(frames[0], (torch.Tensor, np.ndarray)) \
                and frames[0].ndim >= 4:
            # Whole video packed into one batched array — unwrap and iterate T.
            frames = frames[0]
        elif frames and isinstance(frames[0], (list, tuple)):
            # list-of-lists (multi-output): take first batch
            frames = frames[0]
    if isinstance(frames, (torch.Tensor, np.ndarray)):
        # Strip leading singleton dims until we have (T, H, W, C) or (T, C, H, W)
        while frames.ndim > 4 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim == 4:
            return [_frame_to_pil(frames[i]) for i in range(frames.shape[0])]
        if frames.ndim == 3:
            # Single frame
            return [_frame_to_pil(frames)]
        raise TypeError(f"unsupported tensor ndim={frames.ndim} shape={tuple(frames.shape)}")
    return [_frame_to_pil(f) for f in frames]


def main():
    args = parse_args()

    # ── LVSA env-var setup (must happen BEFORE vllm_omni import) ────────────
    # The plugin hooks read these at module import / class instantiation time.
    if not args.no_lvsa:
        os.environ["DIFFUSION_ATTENTION_BACKEND"] = "LVSA"
        os.environ["LVSA_WAN_HOOK"] = "1"
        # Latent-frame count for Wan: (num_frames - 1) // 4 + 1
        t_lat = (args.num_frames - 1) // 4 + 1
        os.environ["LVSA_TOTAL_LATENT_FRAMES"] = str(t_lat)
        os.environ["LVSA_REFERENCE_LATENT_FRAMES"] = "21"
        os.environ["LVSA_AUTO_KEYFRAMES"] = "1"
        os.environ.setdefault("LVSA_ROTATE_KEYFRAMES", "0" if args.no_rotate else "1")
        os.environ.setdefault("LVSA_SPARSITY_SCALE", str(args.sparsity_scale))

    # Register LVSA backend with vllm-omni
    from lvsa_vllm_omni.register import register_lvsa_backend
    register_lvsa_backend()

    # ── Imports (must come after register_lvsa_backend) ─────────────────────
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from diffusers.utils import export_to_video

    print(f"[offline_wan] loading {args.model} (tp={args.tensor_parallel_size}, dtype={args.dtype})")
    omni = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    try:
        print(f"[offline_wan] generating {args.num_frames} frames at {args.width}x{args.height}")
        params_kwargs = dict(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            return_frames=True,
        )
        if args.guidance2 is not None:
            params_kwargs["guidance_scale_2"] = args.guidance2

        params = OmniDiffusionSamplingParams(**params_kwargs)
        if args.negative_prompt:
            from vllm_omni.inputs.data import OmniTextPrompt
            prompt_in = OmniTextPrompt(prompt=args.prompt, negative_prompt=args.negative_prompt)
        else:
            prompt_in = args.prompt

        results = omni.generate(prompt_in, params)
        result = results[0] if isinstance(results, (list, tuple)) else results

        raw_frames = result.images
        if raw_frames is None or (hasattr(raw_frames, "__len__") and len(raw_frames) == 0):
            raise RuntimeError("Omni returned no frames; latents available at result.latents")

        print(f"[offline_wan] raw frames type={type(raw_frames).__name__}", end="")
        if hasattr(raw_frames, "shape"):
            print(f" shape={tuple(raw_frames.shape)}")
        elif hasattr(raw_frames, "__len__"):
            first = raw_frames[0]
            print(f" len={len(raw_frames)} first={type(first).__name__}"
                  f"{' shape=' + str(tuple(first.shape)) if hasattr(first, 'shape') else ''}")
        else:
            print()
        frames = _to_pil_frames(raw_frames)
        print(f"[offline_wan] normalized to {len(frames)} PIL frames "
              f"({frames[0].size if frames else '?'})")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / args.output_name
        if not str(out_path).endswith(".mp4"):
            out_path = out_path.with_suffix(".mp4")
        export_to_video(frames, str(out_path), fps=args.fps)
        print(f"[offline_wan] wrote {out_path}")
    finally:
        # Always release worker subprocesses & GPU memory — otherwise vllm-omni
        # workers can outlive an abnormal exit and orphan ~25 GB per cell.
        try:
            omni.close()
        except Exception as _e:
            print(f"[offline_wan] warning: omni.close() raised {type(_e).__name__}: {_e}")


if __name__ == "__main__":
    main()
