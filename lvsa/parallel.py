"""
parallel.py — Context-parallel setup, LVSA installation, and validation utilities.

Model-agnostic: all model-specific logic is delegated to a ``ModelAdapter``
passed in by the caller.
"""

import argparse
from typing import Any, Optional, Tuple

from .adapters.base import ModelAdapter
from .lvsa_processor import (
    DistributedLVSAProcessor,
    _FLASHINFER_AVAILABLE,
)

# Backwards-compatible alias
WanDistributedLVSAProcessor = DistributedLVSAProcessor


# ── Rotary-embedding patch for standard (non-LVSA) mode ───────────────────────


def patch_rotary_emb_for_context_parallel(
    adapter: ModelAdapter,
    rank: int,
    world: int,
) -> None:
    """
    Delegate to the adapter's CP rotary-embedding patch.

    This is only active in standard (non-LVSA) mode.  When LVSA is enabled,
    DistributedLVSAProcessor.__call__ handles rotary slicing internally and
    this patch is a no-op for those blocks (the processor class is different).

    Must be called before from_pretrained so the patch is in place when
    weights are loaded.
    """
    adapter.patch_rotary_for_cp(rank, world)


# ── LVSA processor installation ────────────────────────────────────────────────


def install_lvsa_processors(
    pipe: Any,
    args: argparse.Namespace,
    rank: int,
    world: int,
    adapter: ModelAdapter,
    sparsity_scale: float = 1.0,
    reference_latent_frames: Optional[int] = None,
) -> DistributedLVSAProcessor:
    """
    Compute LVSA parameters from pipeline metadata and generation arguments,
    construct a DistributedLVSAProcessor, and install it on every
    transformer block's self-attention layer.

    All blocks share the same processor instance.  This is safe because the
    processor is stateless between calls (no mutable state written during
    forward — all local variables are stack-allocated inside __call__).

    Parameters are converted from video-frame space to latent-frame space
    using the adapter's geometry methods.
    """
    vae_t = pipe.vae_scale_factor_temporal

    # Use adapter for model-specific geometry
    total_lat_frames = adapter.latent_frames(args.num_frames, pipe)
    num_patches = adapter.patches_per_frame(args.height, args.width, pipe)
    # Caller may override the adapter's training horizon (e.g. Wan2.2-TI2V-5B is
    # 31 latent frames / 121 frames, not the WanAdapter's default 21).
    ref_lat_frames = (
        reference_latent_frames
        if reference_latent_frames is not None
        else adapter.reference_latent_frames(pipe)
    )

    if rank == 0:
        print(
            f"[LVSA] reference_latent_frames={ref_lat_frames}  "
            f"target_latent_frames={total_lat_frames}  "
            f"extension_ratio={total_lat_frames / ref_lat_frames:.2f}x"
        )
        if total_lat_frames <= ref_lat_frames:
            print(
                f"[LVSA] NOTE: target ({total_lat_frames}) <= reference ({ref_lat_frames}) "
                f"latent frames. Full attention may produce better quality at this length. "
                f"LVSA is most beneficial when extending beyond the model's training length."
            )

    window_size = args.window_size // vae_t
    n_first_frames = args.n_first_frames // vae_t

    # ── Auto-scale LVSA parameters based on reference frame count ──────────
    # If window covers the entire sequence, LVSA degenerates to full attention
    # with overhead.  Cap window + globals to at most ref_lat_frames so LVSA
    # always provides a meaningful reduction.
    effective_window = window_size * 2 + 1 + n_first_frames
    if effective_window >= total_lat_frames and total_lat_frames > ref_lat_frames:
        # Window covers everything — scale down to reference proportions
        # Default: window = ref/2, n_first = ref/4  (reasonable coverage)
        window_size = max(ref_lat_frames // 2, 1)
        n_first_frames = max(ref_lat_frames // 4, 1)
        if rank == 0:
            print(
                f"[LVSA] auto-scaled: window_size={window_size} "
                f"n_first_frames={n_first_frames} "
                f"(based on ref_lat_frames={ref_lat_frames})"
            )

    use_auto_kfi = getattr(args, "auto_keyframes", False) or getattr(args, "rotate_keyframes", False)
    if use_auto_kfi:
        auto_kfi = DistributedLVSAProcessor._compute_auto_kfi(
            total_lat_frames, window_size, n_first_frames,
            reference_frames=ref_lat_frames,
            sparsity_scale=sparsity_scale,
        )
        if rank == 0:
            flag = "--rotate-keyframes" if getattr(args, "rotate_keyframes", False) else "--auto-keyframes"
            print(f"[LVSA] {flag}: computed key_frame_interval={auto_kfi} (latent frames)")
        key_frame_interval = auto_kfi
    else:
        key_frame_interval = (
            args.key_frame_interval // vae_t
            if args.key_frame_interval
            else None
        )

    processor = DistributedLVSAProcessor(
        total_num_latent_frames=total_lat_frames,
        num_patches=num_patches,
        window_size=window_size,
        n_first_frames=n_first_frames,
        key_frame_interval=key_frame_interval,
        rank=rank,
        world=world,
        adapter=adapter,
        sparsity_scale=sparsity_scale,
        reference_frames=ref_lat_frames,
    )

    processor._original_window_size = window_size

    # Override kernel selection based on CLI flags
    use_fi = getattr(args, "flashinfer", False)

    if use_fi and not _FLASHINFER_AVAILABLE:
        if rank == 0:
            print(
                "[LVSA] WARNING: --flashinfer requested but flashinfer is not installed. "
                "Install with: pip install flashinfer-python flashinfer-cubin. "
                "Falling back to SDPA."
            )
        use_fi = False

    processor._use_flashinfer = use_fi

    # Install processor on all self-attention layers (model-specific)
    num_blocks = adapter.install_processor(pipe, processor)

    if rank == 0:
        backend = "FlashInfer" if use_fi else "SDPA"
        fi_info = ""
        if use_fi:
            nnz = len(processor._fi_indices)
            cn = processor._fi_compact_n
            density = nnz / (processor._fi_MB * cn) * 100 if cn > 0 else 0
            fi_info = (
                f"  CSR: MB={processor._fi_MB} nnz={nnz} "
                f"density={density:.1f}% block_size={num_patches} "
                f"compact={cn}/{total_lat_frames}frames"
            )
        print(
            f"[LVSA] installed on {num_blocks} blocks  "
            f"num_patches={num_patches}  "
            f"total_lat_frames={total_lat_frames}  "
            f"backend={backend}{fi_info}"
        )

        if getattr(args, "show_mask", False):
            processor.print_attention_mask()
        show_compact = getattr(args, "show_mask_compact", None)
        if show_compact == "once":
            processor.print_attention_mask_compact()

    return processor


# ── Sequence-length validation ────────────────────────────────────────────────


def compute_and_validate_seq_len(
    num_frames: int,
    height: int,
    width: int,
    transformer_config,
    vae_scale_t: int,
    vae_scale_s: int,
    world: int,
    rank: int,
) -> Tuple[int, int, int]:
    """
    Compute the transformer sequence length and the latent frame count, validate
    divisibility by world_size, and return (seq_len, total_lat_frames, num_patches).

    Only seq_len % world == 0 is required; T_lat divisibility is not needed.
    """
    patch_size_cfg = getattr(transformer_config, "patch_size", [1, 2, 2])
    if isinstance(patch_size_cfg, (list, tuple)):
        patch_size_t = int(patch_size_cfg[0])
        patch_size_s = int(patch_size_cfg[1])
    else:
        patch_size_t = 1
        patch_size_s = int(patch_size_cfg)

    T_lat = (num_frames - 1) // vae_scale_t + 1
    T_patch = (T_lat + patch_size_t - 1) // patch_size_t
    H_patch = height // (vae_scale_s * patch_size_s)
    W_patch = width // (vae_scale_s * patch_size_s)
    seq_len = T_patch * H_patch * W_patch
    num_patches = H_patch * W_patch

    if world > 1:
        assert seq_len % world == 0, (
            f"\n[seq_len divisibility check failed]\n"
            f"  seq_len = T_patch({T_patch}) × H_patch({H_patch}) × W_patch({W_patch}) = {seq_len}\n"
            f"  world_size = {world}\n"
            f"  {seq_len} % {world} = {seq_len % world}  (must be 0)\n"
            f"  Hint: choose num_frames so that the product above is divisible by {world}."
        )

    if rank == 0:
        print(
            f"[seq_len] patch_size={patch_size_cfg}  "
            f"T_lat={T_lat}  T_patch={T_patch}  "
            f"H_patch={H_patch}  W_patch={W_patch}  "
            f"seq_len={seq_len}  per_rank={seq_len // world}"
        )

    return seq_len, T_lat, num_patches


# ── Context-parallel setup ────────────────────────────────────────────────────


def setup_context_parallel(
    adapter: ModelAdapter,
    transformer: Any,
    world: int,
) -> None:
    """
    Delegate context-parallel plan setup to the model adapter.

    The adapter knows the correct layer names (e.g. ``"blocks.0"`` for Wan,
    ``"single_transformer_blocks.0"`` for HunyuanVideo) and sets up the
    _cp_plan accordingly.
    """
    adapter.setup_context_parallel(transformer, world)
