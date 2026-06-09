"""Offline LVSA generation via the in-process vLLM-Omni `Omni` API.

No HTTP server (so no /v1/videos/sync 600s timeout). Exercises the SAME LVSA
code as the served path:
  * wan / hunyuan -> LVSA attention BACKEND (attention_impl.py + flashinfer_runner.py,
    the LSE-merge), selected via diffusion_attention_config {self: LVSA} + LVSA_BACKEND.
  * cosmos        -> LVSA cross-attention HOOK (cosmos3_hook.py), which routes to the
    FlashInfer runner when LVSA_BACKEND=flashinfer.

Usage:
  offline_lvsa.py --family hunyuan --model /data/HunyuanVideo-1.5-Diffusers-480p_t2v \
     --num-frames 193 --height 480 --width 832 --steps 50 --out hv.mp4
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

REF = {"wan": 21, "hunyuan": 33, "cosmos": 48}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["wan", "hunyuan", "cosmos"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="A golden retriever running through a sunlit forest, camera tracking alongside")
    ap.add_argument("--num-frames", type=int, default=193)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--backend", default="flashinfer", choices=["flashinfer", "sdpa", "dense"],
                    help="dense = no LVSA (plain Omni), for baseline latency.")
    ap.add_argument("--flow-shift", type=float, default=None,
                    help="HunyuanVideo flow_shift (example default 5.0); omit to use pipeline default.")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--sparsity-scale", type=float, default=None,
                    help="LVSA_SPARSITY_SCALE: <1 = more sparse, >1 = less sparse "
                         "(scales the auto-keyframe target). Omit = default 1.0. "
                         "Ignored for --backend dense.")
    ap.add_argument("--ref-lat", "--reference-latent-frames", dest="ref_lat",
                    type=int, default=None,
                    help="Override the reference (training) latent-frame count. "
                         "Defaults per family (wan=21, hunyuan=33, cosmos=48); set "
                         "31 for Wan2.2-TI2V-5B (121 frames / new 16x VAE). "
                         "(--reference-latent-frames is an alias matching "
                         "examples/wan_generate.py.)")
    ap.add_argument("--patches-per-frame", type=int, default=None,
                    help="LVSA_PATCHES_PER_FRAME: tokens per latent frame, for the "
                         "plugin's geometry detection. Needed at non-480p / non-8x-VAE "
                         "resolutions (e.g. 880 for Wan2.2-5B 720p, 16x VAE) where the "
                         "default candidate (1560) doesn't match → else LVSA falls back to dense.")
    ap.add_argument("--offload", default="none", choices=["none", "cpu", "layerwise"],
                    help="DiT offload to free VRAM for the VAE decode on long clips.")
    ap.add_argument("--eager", action="store_true",
                    help="enforce_eager: disable torch.compile (lower peak memory, slower/step).")
    ap.add_argument("--init-timeout", type=int, default=1800,
                    help="AsyncOmniEngine orchestrator startup timeout (s). Default 1800 "
                         "(vs the engine's 600) so a slow/contended big-model load — e.g. "
                         "many GPUs loading from one disk in the weekend sweep — doesn't false-timeout.")
    ap.add_argument("--output-dir", type=Path, default=Path("results_plugin_phase"))
    ap.add_argument("--output-name", required=True)
    return ap.parse_args()


def _to_pil_frames(frames):
    import numpy as np, torch
    from PIL import Image

    def one(f):
        if isinstance(f, Image.Image):
            return f.convert("RGB")
        arr = f.detach().cpu().float().numpy() if isinstance(f, torch.Tensor) else f
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 + 1e-3 else arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        return Image.fromarray(arr).convert("RGB")

    if isinstance(frames, (list, tuple)):
        if len(frames) == 1 and hasattr(frames[0], "ndim") and frames[0].ndim >= 4:
            frames = frames[0]
        elif frames and isinstance(frames[0], (list, tuple)):
            frames = frames[0]
    if hasattr(frames, "ndim"):
        while frames.ndim > 4 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim == 4:
            return [one(frames[i]) for i in range(frames.shape[0])]
        if frames.ndim == 3:
            return [one(frames)]
    return [one(f) for f in frames]


def main():
    a = parse_args()
    # MUST be set before any CUDA/torch init
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    lvsa = a.backend != "dense"      # "dense" = no LVSA (baseline)
    # VAE temporal compression factor is 4 for all current families (wan /
    # hunyuan / cosmos); Wan2.2-TI2V-5B's 16x is SPATIAL, temporal stays 4. A
    # wrong t_lat only trips a safe dense geometry-fallback in the hook — it
    # never corrupts output — but revisit this if a 1/2/8x-temporal VAE lands.
    t_lat = (a.num_frames - 1) // 4 + 1
    if lvsa:
        os.environ["LVSA_TOTAL_LATENT_FRAMES"] = str(t_lat)
        os.environ["LVSA_REFERENCE_LATENT_FRAMES"] = str(a.ref_lat if a.ref_lat is not None else REF[a.family])
        os.environ["LVSA_BACKEND"] = a.backend
        os.environ["LVSA_AUTO_KEYFRAMES"] = "1"
        os.environ["LVSA_ROTATE_KEYFRAMES"] = "0" if a.no_rotate else "1"
        if a.sparsity_scale is not None:
            os.environ["LVSA_SPARSITY_SCALE"] = str(a.sparsity_scale)
        if a.patches_per_frame is not None:
            os.environ["LVSA_PATCHES_PER_FRAME"] = str(a.patches_per_frame)
        if a.family == "cosmos":
            os.environ["LVSA_COSMOS3_HOOK"] = "1"     # hook does LVSA on cross-attn
        # wan/hunyuan: no hook -> the LVSA backend handles it (flashinfer LSE-merge)

    from lvsa_vllm_omni.register import register_lvsa_backend
    register_lvsa_backend()   # registers the enum; installs no hook unless LVSA_*_HOOK set

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from diffusers.utils import export_to_video
    import time, torch

    omni_kwargs = {"vae_use_tiling": True, "vae_use_slicing": True, "init_timeout": a.init_timeout}
    if a.eager:
        omni_kwargs["enforce_eager"] = True
    if a.offload == "cpu":
        omni_kwargs["enable_cpu_offload"] = True
    elif a.offload == "layerwise":
        omni_kwargs["enable_layerwise_offload"] = True
    if lvsa and a.family in ("wan", "hunyuan"):
        omni_kwargs["diffusion_attention_config"] = {"per_role": {"self": {"backend": "LVSA"}}}
    if a.family == "cosmos":
        omni_kwargs["model_config"] = {"guardrails": False}

    print(f"[offline_lvsa] family={a.family} backend={a.backend} lvsa={lvsa} T_lat={t_lat} "
          f"frames={a.num_frames} {a.width}x{a.height} steps={a.steps} "
          f"sparsity={a.sparsity_scale if (lvsa and a.sparsity_scale is not None) else 1.0}", flush=True)
    omni = Omni(model=a.model, tensor_parallel_size=1, dtype="bfloat16", **omni_kwargs)
    try:
        pk = dict(
            height=a.height, width=a.width, num_frames=a.num_frames,
            num_inference_steps=a.steps, guidance_scale=a.guidance,
            seed=a.seed, return_frames=True,
        )
        if a.flow_shift is not None:
            pk["extra_step_kwargs"] = {"flow_shift": a.flow_shift}
        params = OmniDiffusionSamplingParams(**pk)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        results = omni.generate(a.prompt, params)   # generation only (excludes model load)
        wall = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        result = results[0] if isinstance(results, (list, tuple)) else results
        if result is None or getattr(result, "images", None) is None:
            raise RuntimeError(
                "omni.generate returned no frames (result.images is None) — "
                "check return_frames=True and that the pipeline produced video output."
            )
        frames = _to_pil_frames(result.images)
        if not frames:
            raise RuntimeError("Generation produced 0 frames — check num_frames / pipeline output.")
        a.output_dir.mkdir(parents=True, exist_ok=True)
        name = a.output_name if a.output_name.endswith(".mp4") else a.output_name + ".mp4"
        out = a.output_dir / name   # str concat, not with_suffix (filenames have dots, e.g. "1.5x")
        export_to_video(frames, str(out), fps=a.fps)
        print(f"[offline_lvsa] wrote {out}  ({len(frames)} frames)", flush=True)
        # Parseable benchmark line (matches bench.py regex):
        print(f"[BENCH] gen_s={wall:.2f} peak_mb={peak_mb:.0f} steps={a.steps} "
              f"frames={len(frames)} s_per_step={wall/max(1,a.steps):.3f}", flush=True)
    finally:
        try:
            omni.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
