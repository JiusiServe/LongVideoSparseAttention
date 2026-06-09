"""Stateful FlashInfer block-sparse runner for dual-stream LVSA hooks (LSE-merge).

The gen video tokens are computed by a fused
``flashinfer.BlockSparseAttentionWrapper`` over the sparse (global + window)
pattern; the understanding/encoder (und/text) tokens are computed by a SEPARATE
dense ``single_prefill`` term; the two partial attentions are combined exactly via
log-sum-exp (the flash-attention block-merge). This avoids appending the und K/V
as a zero-padded block inside the block-sparse kernel — that padding (the und
length is essentially never a multiple of the patches-per-frame block size P)
would otherwise be attended as phantom zero-value keys, diluting every gen query
and corrupting output. Splitting the und out is exact, scales to any horizon
(no giant per-block mask / int32 overflow), and keeps the gen block-sparse fast.

GQA is native (plan ``num_kv_heads = Hkv``; ``single_prefill`` infers it from the
und K/V). The wrapper / workspace / compact-KV buffers are reused across all
attention layers within a generation; the plan is rebuilt only when the sparse
pattern (CSR) changes (so rotating keyframes re-plan correctly).
"""

import torch

try:
    import flashinfer  # noqa: F401
    FLASHINFER_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    flashinfer = None
    FLASHINFER_AVAILABLE = False

_DTYPE_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


_SHARED_RUNNER = None


def get_shared_runner():
    """Process-wide singleton runner shared across ALL attention layers.

    Each layer's attention is computed sequentially on one stream and the
    runner's K/V/workspace are scratch buffers that are fully produced and
    consumed within a single ``run()`` call, so a single shared instance is
    safe. Sharing avoids duplicating the ~1 GB of persistent scratch
    (workspace + compact K/V) per layer — for a 40-layer model that is ~30 GB
    of otherwise-wasted resident memory. All layers have identical attention
    geometry, so the cached buffers never need resizing and the plan is built
    once per step instead of once per layer. Mirrors the standalone
    DistributedSWAProcessor, which shares one processor across all layers.
    """
    global _SHARED_RUNNER
    if _SHARED_RUNNER is None:
        _SHARED_RUNNER = FlashInferDualStreamRunner()
    return _SHARED_RUNNER


class FlashInferDualStreamRunner:
    """Gen block-sparse + separate dense und term, merged via log-sum-exp."""

    def __init__(self) -> None:
        self._wrapper = None
        self._workspace = None
        self._compact_k = None
        self._compact_v = None
        self._q_pad = None
        self._sig = None          # scalar plan signature (shapes + dtype/device)
        self._sig_indptr = None   # cached CSR row pointers from the last plan
        self._sig_indices = None  # cached CSR column indices from the last plan

    def _plan(self, metadata, H, Hkv, D, device, q_dtype) -> None:
        """Plan the GEN-ONLY block-sparse CSR (no und block, no mask)."""
        indptr = metadata.fi_indptr.to(device)
        indices = metadata.fi_indices.to(device)
        if self._workspace is None:
            self._workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device,
            )
        self._wrapper = flashinfer.BlockSparseAttentionWrapper(self._workspace)
        dtype_str = _DTYPE_STR.get(q_dtype, "bfloat16")
        self._wrapper.plan(
            indptr=indptr,
            indices=indices,
            M=metadata.fi_M,
            N=metadata.fi_N,
            R=metadata.num_patches,
            C=metadata.num_patches,
            num_qo_heads=H,
            num_kv_heads=Hkv,
            head_dim=D,
            q_data_type=dtype_str,
            kv_data_type=dtype_str,
            o_data_type=dtype_str,
        )

    def run(self, query, key, value, k_global, v_global, metadata):
        """Sparse gen attention + dense und attention, exact LSE-merge.

        ``k_global``/``v_global`` are the gen global-frame K/V (from
        ``build_global_kv``) with the und/encoder K/V concatenated after them;
        we split the und back out and handle it as a separate dense term.
        """
        B, local_seq, H, D = query.shape
        Hkv = key.shape[2]
        P = metadata.num_patches
        M = metadata.fi_M

        num_global_video = len(metadata.global_indices) * P
        k_gen_g = k_global[:, :num_global_video]      # gen global frames
        v_gen_g = v_global[:, :num_global_video]
        k_und = k_global[:, num_global_video:]        # appended und/text K/V
        v_und = v_global[:, num_global_video:]
        S_und = k_und.shape[1]

        # Replan when the plan-defining geometry changes. We compare the CSR
        # tensors EXACTLY (torch.equal) rather than a sum-based hash: two
        # different patterns can share the same indptr.sum()/indices.sum() (e.g.
        # an index permutation), which would silently reuse a stale plan. The
        # scalar tuple also carries dtype+device. CSR tensors are tiny int
        # arrays, so the exact compare is negligible next to the kernel.
        sig = (
            int(metadata.fi_M), int(metadata.fi_compact_n), int(H), int(Hkv),
            int(metadata.fi_indptr.numel()), int(metadata.fi_indices.numel()),
            str(query.dtype), str(query.device),
        )
        need_replan = (
            sig != self._sig
            or self._sig_indptr is None
            or not torch.equal(self._sig_indptr, metadata.fi_indptr)
            or not torch.equal(self._sig_indices, metadata.fi_indices)
        )
        if need_replan:
            self._plan(metadata, H, Hkv, D, query.device, query.dtype)
            self._sig = sig
            self._sig_indptr = metadata.fi_indptr
            self._sig_indices = metadata.fi_indices
            # Invariant for the compact-KV reuse below: the global+local copy
            # destination frames must tile [0, compact_n) EXACTLY. The compact
            # buffer is only zero-filled on (re)allocation, so if a future
            # metadata-builder change left a gap, stale K/V from a prior call
            # would leak into the kernel. Validate once per geometry (cheap —
            # metadata is constant for a given sig).
            _dst_frames = sorted(
                [d // P for _, d in metadata.fi_global_copies]
                + [d // P for _, d in metadata.fi_local_copies]
            )
            assert _dst_frames == list(range(metadata.fi_compact_n)), (
                f"compact-KV copies must tile [0, {metadata.fi_compact_n}) exactly "
                f"(got {len(_dst_frames)} frames); a gap would leak stale K/V"
            )

        # ── gen compact KV buffer (gen frames only) ──
        # Reused across calls without re-zeroing when the shape matches; correct
        # only because the copies below fully tile [0, compact_N) (asserted above).
        compact_N = metadata.fi_compact_n * P
        shape = (B, compact_N, Hkv, D)
        if self._compact_k is None or self._compact_k.shape != shape:
            self._compact_k = query.new_zeros(*shape)
            self._compact_v = query.new_zeros(*shape)
        ck, cv = self._compact_k, self._compact_v
        for src_s, dst_s in metadata.fi_global_copies:
            ck[:, dst_s:dst_s + P] = k_gen_g[:, src_s:src_s + P]
            cv[:, dst_s:dst_s + P] = v_gen_g[:, src_s:src_s + P]
        for src_s, dst_s in metadata.fi_local_copies:
            ck[:, dst_s:dst_s + P] = key[:, src_s:src_s + P]
            cv[:, dst_s:dst_s + P] = value[:, src_s:src_s + P]

        # ── q padding to M = MB*P (block-aligned) ──
        if local_seq < M:
            if self._q_pad is None or self._q_pad.shape != (B, M, H, D):
                self._q_pad = query.new_zeros(B, M, H, D)
            self._q_pad[:, :local_seq] = query
            q_pad = self._q_pad
        else:
            q_pad = query.contiguous()

        out = query.new_empty(B, local_seq, H, D)
        for b in range(B):
            # gen sparse attention (with LSE)
            o_gen, lse_gen = self._wrapper.run(
                q_pad[b], ck[b], cv[b], return_lse=True)     # [M,H,D], [M,H]
            o_gen = o_gen[:local_seq].float()                # [S,H,D]
            lse_gen = lse_gen[:local_seq]                    # [S,H] fp32
            if S_und > 0:
                # dense und attention (with LSE), same 1/sqrt(D) scale
                o_und, lse_und = flashinfer.single_prefill_with_kv_cache(
                    query[b].contiguous(), k_und[b].contiguous(), v_und[b].contiguous(),
                    causal=False, return_lse=True)            # [S,H,D], [S,H]
                o_und = o_und.float()
                # exact flash-merge of two disjoint key sets. NOTE: FlashInfer
                # returns LSE in log2 (= log2(sum exp)), so weights use exp2.
                m = torch.maximum(lse_gen, lse_und)
                w_gen = torch.exp2(lse_gen - m).unsqueeze(-1)
                w_und = torch.exp2(lse_und - m).unsqueeze(-1)
                out[b] = ((o_gen * w_gen + o_und * w_und) / (w_gen + w_und)).to(query.dtype)
            else:
                out[b] = o_gen.to(query.dtype)
        return out
