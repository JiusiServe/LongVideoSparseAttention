"""LVSA AttentionImpl for vllm-omni.

Implements ``forward_cuda()`` with sparse windowed attention for self-attention
and dense SDPA fallback for cross-attention and warmup runs.

Key design: vllm-omni concatenates [video_tokens; encoder_tokens] before calling
the attention backend. We split them using the known geometry:
  video_seq = T_lat * P  (where P = H_lat * W_lat = 1560 for HunyuanVideo 480p)
  encoder_seq = total_seq - video_seq

The warmup dummy run uses different spatial dims (64x64 → 4096 tokens). We detect
this by checking if seq_len is compatible with T_lat * P for the expected P.
"""

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

from lvsa.sparse_attention import (
    LVSAMetadata,
    compute_auto_kfi,
    print_attention_mask_compact,
    sparse_windowed_attention,
)

from . import step_tracker
from .config import LVSAConfig
from .global_kv import build_global_kv


class _StepCounter:
    """Infer denoising step by counting self-attention forward calls.

    Auto-calibrates n_blocks by detecting when the same layer ID repeats.
    Resets when seq_len changes (new generation or warmup→real transition).

    Concurrency: state is intentionally shared across all attention blocks
    within one generation — the *whole point* is to observe a single shared
    counter. We do not protect it with ``threading.local()`` because every
    vllm-omni diffusion worker is a single process and routes requests
    through a single asyncio loop (one OS thread); the warmup and every
    user request hit this counter sequentially. ``threading.local()`` would
    also not help future in-worker batching (coroutines share an OS thread)
    — that scenario would need a request-scoped key.
    """

    def __init__(self) -> None:
        self._call_count: int = 0
        self._n_blocks: Optional[int] = int(os.environ.get("LVSA_N_BLOCKS", 0)) or None
        # Forward passes per denoising step. With CFG (classifier-free guidance)
        # vllm-omni runs 2 forward passes per denoising step (cond + uncond).
        # Without CFG (guidance_scale == 1) it is 1. Default 2 matches the
        # common case (guidance > 1).
        self._cfg_passes: int = max(1, int(os.environ.get("LVSA_CFG_PASSES", 2)))
        self._step: int = 0
        self._generation_seq_len: Optional[int] = None
        self._seen_ids: set = set()
        self._last_step_time: Optional[float] = None
        if self._n_blocks is not None:
            print(f"[LVSA] Step counter: n_blocks={self._n_blocks} (from env)")
        if self._cfg_passes != 2:
            print(f"[LVSA] Step counter: cfg_passes={self._cfg_passes} (from env)")

    def tick(self, layer_id: int, seq_len: int) -> int:
        # Detect seq_len change → reset (warmup→real or new request)
        if self._generation_seq_len is not None and seq_len != self._generation_seq_len:
            self._generation_seq_len = seq_len
            self._call_count = 0
            self._step = 0
            self._seen_ids.clear()
            self._last_step_time = None
            step_tracker.set_step(0)

        if self._generation_seq_len is None:
            self._generation_seq_len = seq_len

        self._call_count += 1

        # Auto-calibrate n_blocks on the first repeated layer_id (= start of
        # the second forward pass over the same set of attention blocks).
        step_boundary = False
        if self._n_blocks is None:
            if layer_id in self._seen_ids:
                self._n_blocks = len(self._seen_ids)
                self._seen_ids.clear()
                print(
                    f"[LVSA] Step counter auto-calibrated: "
                    f"n_blocks={self._n_blocks} cfg_passes={self._cfg_passes}"
                )
            else:
                self._seen_ids.add(layer_id)

        # Compute step_idx from total call count. One denoising step =
        # n_blocks * cfg_passes attention forwards (cfg_passes=2 with CFG).
        # _call_count is the running total since the last seq_len reset.
        if self._n_blocks is not None:
            threshold = self._n_blocks * self._cfg_passes
            new_step = (self._call_count - 1) // threshold
            if new_step > self._step:
                self._step = new_step
                step_boundary = True

        # Opt-in per-step memory log (device-agnostic: CUDA + Ascend NPU).
        if step_boundary and os.environ.get("LVSA_MEM_LOG", "0") == "1":
            from lvsa.device import memory_stats
            stats = memory_stats()
            if stats is not None:
                kind, dev, alloc, reserved, peak = stats
                print(
                    f"[LVSA-MEM] step={self._step} {kind}={dev} "
                    f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB",
                    flush=True,
                )

        # Opt-in per-step wall-clock timing. Logs the time spent on the step
        # that JUST completed (self._step - 1).
        if step_boundary and os.environ.get("LVSA_STEP_TIME_LOG", "0") == "1":
            import time as _time
            now = _time.perf_counter()
            if self._last_step_time is not None:
                dt = now - self._last_step_time
                print(
                    f"[LVSA-TIME] step={self._step - 1} dt={dt:.3f}s",
                    flush=True,
                )
            self._last_step_time = now

        step_tracker.set_step(self._step)
        return self._step

    @property
    def step(self) -> int:
        return self._step


_step_counter: Optional[_StepCounter] = None


def _get_step_counter() -> _StepCounter:
    global _step_counter
    if _step_counter is None:
        _step_counter = _StepCounter()
    return _step_counter


def _reset_step_counter() -> None:
    """Drop the module-level step counter so the next access rebuilds fresh.

    Called from ``step_tracker.reset()`` so test fixtures that reset the
    thread-local tracker also clear this counter's auto-calibrated block
    count and seen-id set. Without this, counter state leaked across tests
    and made `step_tracker.set_step(N)` non-authoritative.
    """
    global _step_counter, _mask_log_last_step
    _step_counter = None
    _mask_log_last_step = -1


# Patches-per-frame candidate set: resolved from env + defaults. See
# ``config.candidate_patches_per_frame`` for the resolution rules. Default
# covers Wan / HunyuanVideo at 480p (P=1560); override via
# ``LVSA_PATCHES_PER_FRAME`` or the resolution env vars for other configs.
from .config import candidate_patches_per_frame as _ppf_candidates


# Module-level dedup for LVSA_MASK_LOG: print the compact mask once per step
# boundary, not once per attention layer. Reset whenever the step counter is
# reset (new generation request, warmup→real transition).
#
# Concurrency: see _StepCounter docstring — single-process, single-thread
# vllm-omni worker; the dedup is supposed to be shared across all blocks.
_mask_log_last_step: int = -1


# One-time warning when LVSA_BACKEND=flashinfer is requested but FlashInfer is
# not installed — degrade to the SDPA per-frame loop with a legible message
# instead of dereferencing the missing module deep inside the runner.
_fi_unavailable_warned: bool = False


def _warn_flashinfer_unavailable_once() -> None:
    global _fi_unavailable_warned
    if not _fi_unavailable_warned:
        _fi_unavailable_warned = True
        print(
            "[LVSA] LVSA_BACKEND=flashinfer requested but flashinfer is not "
            "installed -> falling back to the SDPA per-frame loop. Install with "
            "`pip install flashinfer-python flashinfer-cubin` to enable it.",
            flush=True,
        )


def _mask_log_should_fire(step_idx: int) -> bool:
    """Return True if LVSA_MASK_LOG env requests printing at this step
    AND we have not yet printed for this step. Dedups across layers.

    Supported env values:
      LVSA_MASK_LOG=1        → every step
      LVSA_MASK_LOG=once     → first LVSA step only
      LVSA_MASK_LOG=N        → step N only
      LVSA_MASK_LOG=N-M      → inclusive step range [N, M]
      LVSA_MASK_LOG=N,M,K    → specific steps
    """
    global _mask_log_last_step
    spec = os.environ.get("LVSA_MASK_LOG", "")
    if not spec or spec == "0":
        return False
    if step_idx == _mask_log_last_step:
        return False  # already printed for this step

    fire = False
    if spec == "1":
        fire = True
    elif spec == "once":
        fire = _mask_log_last_step == -1
    elif "-" in spec:
        try:
            lo, hi = spec.split("-", 1)
            fire = int(lo) <= step_idx <= int(hi)
        except ValueError:
            fire = False
    else:
        try:
            wanted = {int(x.strip()) for x in spec.split(",") if x.strip()}
            fire = step_idx in wanted
        except ValueError:
            fire = False

    if fire:
        _mask_log_last_step = step_idx
    return fire


class LVSAAttentionImpl:
    """Sparse windowed attention backend for vllm-omni.

    Handles dual-stream models (HunyuanVideo) where the attention receives
    concatenated [video; encoder] tokens. Detects and skips warmup calls
    where the geometry doesn't match the expected video frame structure.
    """

    _total_instances: int = 0
    _logged_geometry: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        qkv_layout: Optional[str] = None,
        backend_kwargs: Optional[dict] = None,
        **extra_impl_args: Any,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.num_kv_heads = num_kv_heads or num_heads
        # vllm-omni 0.22 passes these two to every AttentionImpl (by keyword,
        # from diffusion/attention/layer.py). ``backend_kwargs`` carries the
        # per-role ``AttentionSpec.extra`` dict, available for future per-role
        # LVSA overrides; today LVSA config is sourced from LVSA_* env vars.
        self.qkv_layout = qkv_layout
        self.backend_kwargs = backend_kwargs or {}
        self.config = LVSAConfig.from_env()

        self._lvsa_metadata: Optional[LVSAMetadata] = None
        self._cached_total_frames: Optional[int] = None
        self._cached_num_patches: Optional[int] = None
        self._cached_step: int = -1

        LVSAAttentionImpl._total_instances += 1

    # ── Public interface ─────────────────────────────────────────────────

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        return self.forward_cuda(query, key, value, attn_metadata)

    # Run the WHOLE LVSA attention eager. vllm-omni torch.compiles the transformer,
    # and this entry's geometry-detect + ``counter.tick(id(self), seq_len)`` (varying
    # seq_len + per-instance id) makes dynamo recompile up to recompile_limit (8).
    # Each cached graph pins GPU memory, so on a big model (HunyuanVideo: ~34 GB
    # loaded) the 8-graph bloat (~45 GB) blows past 80 GB even at 1×/1-step — while
    # the eager standalone fits the same workload at ~47 GB. Disabling compile here
    # graph-breaks at the attention (the transformer still compiles around it; the
    # FlashInfer/SDPA attention is a fused kernel, so nothing is lost) → no recompiles,
    # no bloat. Supersedes the narrower @disable on _lvsa_attention below.
    @torch.compiler.disable
    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        # Cross-attention: different seq lengths → dense (benign, no warning)
        if query.shape[1] != key.shape[1]:
            return self._dense_attention(query, key, value)

        # Resolve T_lat from config/env
        total_frames = self._resolve_total_frames()
        if total_frames is None:
            from ._fallback import warn_fallback
            warn_fallback(
                origin="forward_cuda",
                reason="no_t_lat",
                seq_len=query.shape[1],
                extra={"hint": "set LVSA_TOTAL_LATENT_FRAMES"},
            )
            return self._dense_attention(query, key, value)

        # Detect video geometry: try to find P such that seq = T*P + text
        seq_len = query.shape[1]
        P, text_tokens = self._detect_geometry(seq_len, total_frames)
        if P is None:
            # Geometry doesn't match (warmup, pre-sharded seq, or unknown model).
            # First call per model is almost always warmup → dedup handles it.
            from ._fallback import warn_fallback
            warn_fallback(
                origin="forward_cuda",
                reason="geometry_detect",
                seq_len=seq_len,
                extra={"T_lat": total_frames, "known_ppf": _ppf_candidates()},
            )
            return self._dense_attention(query, key, value)

        # Track denoising step
        counter = _get_step_counter()
        step_idx = counter.tick(id(self), seq_len)

        return self._lvsa_attention(query, key, value, total_frames, P, text_tokens)

    # ── LVSA path ─────────────────────────────────────────────────────────

    # Run the LVSA path eager. The metadata/CSR build is data-dependent Python
    # (``.item()``, list construction) and is rebuilt every denoising step under
    # rotating keyframes, so under vllm-omni's mandatory ``torch.compile`` it
    # graph-breaks and recompiles each step (the first step exceeded the
    # server's request timeout → 504). Marking the LVSA path compiler-disabled
    # keeps the rest of the transformer compiled while the already-fused
    # FlashInfer / SDPA attention runs eagerly — as it does in the standalone.
    @torch.compiler.disable
    def _lvsa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        total_latent_frames: int,
        P: int,
        text_tokens: int,
    ) -> torch.Tensor:
        B, seq_len, H, D = query.shape
        video_seq = total_latent_frames * P

        # Split video and encoder tokens
        q_video = query[:, :video_seq]
        k_video = key[:, :video_seq]
        v_video = value[:, :video_seq]

        if text_tokens > 0:
            k_text = key[:, video_seq:]
            v_text = value[:, video_seq:]
            q_text = query[:, video_seq:]
        else:
            k_text = v_text = q_text = None

        step_idx = step_tracker.get_step()

        # Build or rebuild LVSAMetadata
        needs_rebuild = (
            self._lvsa_metadata is None
            or self._cached_total_frames != total_latent_frames
            or self._cached_num_patches != P
            or (self.config.rotate_keyframes and self._cached_step != step_idx)
        )

        if needs_rebuild:
            cfg = self.config
            W = cfg.latent_window_size
            n_first = cfg.latent_n_first_frames
            kfi = cfg.latent_key_frame_interval

            if cfg.auto_keyframes:
                kfi = compute_auto_kfi(
                    total_latent_frames, W, n_first,
                    reference_frames=cfg.reference_latent_frames,
                    sparsity_scale=cfg.sparsity_scale,
                )

            offset = 0
            if cfg.rotate_keyframes and kfi > 0:
                offset = step_idx % kfi

            self._lvsa_metadata = LVSAMetadata.build(
                total_latent_frames=total_latent_frames,
                num_patches=P,
                window_size=W,
                n_first_frames=n_first,
                key_frame_interval=kfi,
                rank=0,
                world=1,
                expand_window=True,
                keyframe_offset=offset,
                reference_frames=cfg.reference_latent_frames,
                sparsity_scale=cfg.sparsity_scale,
            )
            self._lvsa_metadata.ensure_device(query.device)
            self._cached_total_frames = total_latent_frames
            self._cached_num_patches = P
            self._cached_step = step_idx

        # Opt-in compact attention-mask log (LVSA_MASK_LOG env var).
        # Dedups across layers via module-level _mask_log_last_step so we
        # print once per step boundary, not once per attention layer.
        if _mask_log_should_fire(step_idx):
            print(f"[LVSA-MASK] step={step_idx}  "
                  f"T_lat={total_latent_frames}  W={self._lvsa_metadata.window_size}  "
                  f"|G|={len(self._lvsa_metadata.global_set)}  "
                  f"kfi={self._lvsa_metadata.key_frame_interval}", flush=True)
            print_attention_mask_compact(
                total_frames=total_latent_frames,
                window_size=self._lvsa_metadata.window_size,
                global_set=self._lvsa_metadata.global_set,
                expand_window=self._lvsa_metadata.expand_window,
            )

        # Build global K/V from video + append encoder K/V
        k_global, v_global = build_global_kv(
            k_video, v_video, self._lvsa_metadata.global_indices, P,
        )
        if k_text is not None:
            k_global = torch.cat([k_global, k_text], dim=1)
            v_global = torch.cat([v_global, v_text], dim=1)

        # Sparse attention on video tokens.
        # FlashInfer goes through the LSE-merge runner: gen block-sparse + a
        # SEPARATE dense encoder/text term combined via log-sum-exp. This avoids
        # the old zero-padded-encoder-block bug (a single fused block-sparse call
        # attended to phantom zero keys when the text length wasn't a multiple
        # of P — diluting every video query). SDPA keeps its exact per-frame loop.
        if self.config.backend == "flashinfer":
            # Shared singleton across ALL layers: the runner's workspace +
            # compact K/V are per-call scratch, so one instance suffices. A
            # per-layer instance (the old `self._fi_runner`) duplicated ~1 GB of
            # persistent scratch per layer (~30 GB for a 40-layer model).
            from .flashinfer_runner import FLASHINFER_AVAILABLE, get_shared_runner
            if FLASHINFER_AVAILABLE:
                out_video = get_shared_runner().run(
                    q_video, k_video, v_video, k_global, v_global, self._lvsa_metadata,
                )
            else:
                # flashinfer absent → legible degrade to SDPA (not an opaque
                # AttributeError on the missing BlockSparseAttentionWrapper).
                _warn_flashinfer_unavailable_once()
                out_video = sparse_windowed_attention(
                    q_video, k_video, v_video, k_global, v_global,
                    self._lvsa_metadata, backend="sdpa",
                )
        else:
            out_video = sparse_windowed_attention(
                q_video, k_video, v_video, k_global, v_global,
                self._lvsa_metadata, backend=self.config.backend,
            )

        if q_text is not None:
            # Encoder queries attend to all K/V (dense)
            k_all = torch.cat([k_video, k_text], dim=1)
            v_all = torch.cat([v_video, v_text], dim=1)
            out_text = self._dense_attention(q_text, k_all, v_all)
            return torch.cat([out_video, out_text], dim=1)

        return out_video

    # ── Dense fallback ───────────────────────────────────────────────────

    def _dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.softmax_scale,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=(key.shape[2] != query.shape[2]),   # GQA models (e.g. HunyuanVideo)
        )
        return out.transpose(1, 2)

    # ── Geometry detection ───────────────────────────────────────────────

    def _detect_geometry(self, seq_len: int, total_frames: int) -> tuple:
        """Detect video/encoder split from sequence length.

        Returns (P, text_tokens) if geometry matches, or (None, None) for
        warmup/dummy runs where the sequence doesn't match expected video frames.
        """
        # Try each candidate patches-per-frame value in turn
        for p in _ppf_candidates():
            video_seq = total_frames * p
            text = seq_len - video_seq
            if text >= 0 and text < video_seq:
                if not LVSAAttentionImpl._logged_geometry:
                    LVSAAttentionImpl._logged_geometry = True
                    print(f"[LVSA] Geometry detected: T_lat={total_frames} P={p} "
                          f"video_seq={video_seq} text={text} total={seq_len}")
                return p, text

        # No known P matched → skip LVSA (warmup/dummy run or unknown model)
        return None, None

    def _resolve_total_frames(self) -> Optional[int]:
        if self.config.total_latent_frames is not None:
            return self.config.total_latent_frames
        tracked = step_tracker.get_total_latent_frames()
        if tracked is not None:
            return tracked
        return None
