"""
base.py — ModelAdapter abstract base class.

Each video diffusion model that wants to use the LVSA LVSA engine must
implement this interface.  The adapter bridges model-specific attention
layer internals (QKV projections, RoPE format, output projection, layer
naming) to the generic DistributedLVSAProcessor.

The LVSA engine itself (window math, Triton kernel, FlashInfer CSR, ring
attention) is completely model-agnostic and never imported here.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch


class ModelAdapter(ABC):
    """Bridge between a diffusion model's attention layer and the LVSA engine.

    Implementations live in ``lvsa.adapters.<model>.py`` — one file per model
    family (e.g. ``wan.py``, ``hunyuan_video.py``).
    """

    # ── Geometry ──────────────────────────────────────────────────────────────

    @abstractmethod
    def patches_per_frame(self, height: int, width: int, pipe: Any) -> int:
        """Number of spatial tokens per latent temporal frame.

        Depends on VAE spatial compression and patchification.
        """

    @abstractmethod
    def latent_frames(self, num_frames: int, pipe: Any) -> int:
        """Convert raw video frame count to latent temporal frame count.

        Depends on the VAE temporal compression factor.
        """

    @abstractmethod
    def reference_latent_frames(self, pipe: Any) -> int:
        """Native latent frame count the model was trained on.

        This is the number of latent temporal frames the model produces by
        default (e.g. Wan: 81 raw → 21 latent, CogVideoX: 49 raw → 13 latent).
        Used to determine when LVSA is beneficial (only for sequences longer
        than the reference) and to auto-tune LVSA parameters.

        Parameters
        ----------
        pipe : the loaded pipeline (used to read config / VAE parameters)

        Returns
        -------
        int
            The reference latent frame count.
        """

    # ── QKV extraction ────────────────────────────────────────────────────────

    @abstractmethod
    def extract_qkv(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into Q, K, V.

        Returns
        -------
        query, key, value : each ``[B, seq, H, D]``
            Already QK-normed and unflattened to per-head layout.
        """

    @abstractmethod
    def extract_cross_attn_kv(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract K/V for cross-attention (e.g. I2V image context).

        Returns ``(key_cross, value_cross)`` each ``[B, ctx_len, H, D]``,
        or ``None`` if the model / layer has no cross-attention path.

        When not None, the caller also needs the *trimmed* encoder hidden
        states (text-only, after splitting off image tokens).  Use
        :meth:`split_encoder_for_cross_attn` for that.
        """

    @abstractmethod
    def split_encoder_for_cross_attn(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split encoder_hidden_states into (text, image_ctx) components.

        Returns
        -------
        encoder_text : [B, text_len, C]
            Text tokens passed to self-attention QKV extraction.
        encoder_image : [B, img_len, C] or None
            Image tokens for cross-attention, or None if no I2V.
        """

    # ── Rotary position embeddings ────────────────────────────────────────────

    @abstractmethod
    def apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rotary_emb: Any,
        local_seq: int,
        rank: int,
        world: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply (and optionally slice) rotary position embeddings to Q and K.

        Parameters
        ----------
        query, key : [B, local_seq, H, D]
        rotary_emb : model-specific format (e.g. (cos, sin) tuple or 3D freq tensor)
        local_seq  : number of tokens on this rank
        rank, world: distributed info for slicing full-sequence RoPE
        """

    # ── Output projection ─────────────────────────────────────────────────────

    @abstractmethod
    def output_projection(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        query_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project attention output back to hidden dimension.

        Parameters
        ----------
        attn : the attention layer
        hidden_states : [B, seq, H, D]
        query_dtype : dtype to cast to before projection

        Returns
        -------
        [B, seq, C]
        """

    # ── Encoder projections (dual-stream models) ─────────────────────────────

    def extract_encoder_qkv(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Project encoder_hidden_states into Q, K, V for dual-stream models.

        Returns
        -------
        (encoder_q, encoder_k, encoder_v) each ``[B, text_seq, H, D]``,
        or ``None`` if this model doesn't use dual-stream attention.

        When not None, the LVSA processor concatenates these with video Q/K/V
        for joint attention, then splits the output back.
        """
        return None

    # ── Output formatting ──────────────────────────────────────────────────────

    def format_output(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_output: Optional[torch.Tensor],
        encoder_seq_len: int,
        query_dtype: torch.dtype,
    ) -> Any:
        """Format the final output for the model's block forward().

        Parameters
        ----------
        attn : the attention layer
        hidden_states : [B, video_seq, C] — video output, already projected
        encoder_output : [B, enc_seq, H, D] or None — raw encoder attention output
        encoder_seq_len : number of encoder tokens (0 if single-stream)
        query_dtype : original query dtype

        Returns
        -------
        A single tensor for single-stream models (Wan),
        or a tuple ``(hidden_states, encoder_hidden_states)`` for dual-stream (HunyuanVideo).
        """
        return hidden_states

    # ── Cross-attention dispatch ──────────────────────────────────────────────

    @abstractmethod
    def cross_attention(
        self,
        attn: Any,
        query: torch.Tensor,
        encoder_image: torch.Tensor,
        attention_backend: Any,
    ) -> Optional[torch.Tensor]:
        """Compute cross-attention with image context (I2V).

        Returns the cross-attention output ``[B, seq, H*D]`` to be *added*
        to the self-attention output, or ``None`` if not applicable.
        """

    # ── Pipeline integration ──────────────────────────────────────────────────

    @abstractmethod
    def install_processor(self, pipe: Any, processor: Any) -> int:
        """Install the LVSA processor on all relevant self-attention layers.

        Returns the number of blocks the processor was installed on.
        """

    @abstractmethod
    def setup_context_parallel(self, transformer: Any, world: int) -> None:
        """Attach a _cp_plan and enable_parallelism on the transformer.

        Layer names (e.g. ``"blocks.0"`` vs ``"single_transformer_blocks.0"``)
        differ per model.
        """

    @abstractmethod
    def patch_rotary_for_cp(self, rank: int, world: int) -> None:
        """Patch the model's standard attention processor for CP rotary slicing.

        Called *before* ``from_pretrained`` so the patch is in place when
        weights are loaded.  Only needed when ``world > 1``.
        """
