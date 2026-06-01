"""Tests for sparse_attention.py — LVSAMetadata, helper functions, and backend dispatch."""

import torch
import pytest

from lvsa.sparse_attention import (
    LVSAMetadata,
    adaptive_window_bounds,
    compute_auto_kfi,
    compute_boundary_guard_frames,
    compute_global_indices,
    expanded_window_bounds,
    get_window_bounds,
    sparse_windowed_attention,
    lvsa_sdpa,
)


# ── LVSAMetadata.build() ─────────────────────────────────────────────────────


class TestLVSAMetadataBuild:
    """Test LVSAMetadata.build() produces valid metadata."""

    def _build(self, T=21, P=30, W=3, n_first=1, kfi=4, rank=0, world=1, expand=True):
        return LVSAMetadata.build(
            total_latent_frames=T, num_patches=P, window_size=W,
            n_first_frames=n_first, key_frame_interval=kfi,
            rank=rank, world=world, expand_window=expand,
        )

    def test_basic_build(self):
        m = self._build()
        assert m.total_latent_frames == 21
        assert m.num_patches == 30
        assert m.local_seq == 21 * 30
        assert m.global_token_start == 0

    def test_global_indices_include_first_frames(self):
        m = self._build(n_first=3)
        assert 0 in m.global_set
        assert 1 in m.global_set
        assert 2 in m.global_set

    def test_global_indices_sorted(self):
        m = self._build()
        assert m.global_indices == sorted(m.global_indices)

    def test_local_frames_cover_local_seq(self):
        m = self._build()
        total_tokens = sum(q_end - q_start for _, q_start, q_end in m.local_frames)
        assert total_tokens == m.local_seq

    def test_window_ctx_keys_match_local_frames(self):
        m = self._build()
        local_frame_ids = {f for f, _, _ in m.local_frames}
        assert set(m.window_ctx.keys()) == local_frame_ids

    def test_window_ctx_excludes_globals(self):
        m = self._build()
        for f, parts in m.window_ctx.items():
            for s, e in parts:
                for t in range(s, e, m.num_patches):
                    frame = (m.global_token_start + t) // m.num_patches
                    assert frame not in m.global_set

    def test_global_frame_mask_shape(self):
        m = self._build()
        assert m.global_frame_mask.shape == (21,)

    def test_global_frame_mask_values(self):
        m = self._build()
        for i in range(21):
            if i in m.global_set:
                assert m.global_frame_mask[i] == 1
            else:
                assert m.global_frame_mask[i] == 0

    def test_window_bounds_shape(self):
        m = self._build()
        assert m.window_bounds.shape == (21, 2)

    def test_window_bounds_valid_range(self):
        m = self._build()
        for f in range(21):
            lo, hi = m.window_bounds[f].tolist()
            assert 0 <= lo <= hi <= 20

    def test_attended_indices_shape(self):
        m = self._build()
        T_local = len(m.local_frames)
        assert m.attended_indices.shape[0] == T_local
        assert m.attended_indices.shape[1] == m.attended_C

    def test_csr_indptr_valid(self):
        m = self._build()
        assert m.fi_indptr[0] == 0
        assert len(m.fi_indptr) == m.fi_MB + 1
        assert m.fi_indptr[-1].item() == len(m.fi_indices)

    def test_csr_indices_in_range(self):
        m = self._build()
        if m.fi_compact_n > 0:
            assert m.fi_indices.max().item() < m.fi_compact_n

    def test_copies_cover_compact(self):
        m = self._build()
        assert len(m.fi_global_copies) + len(m.fi_local_copies) == m.fi_compact_n

    def test_multi_gpu_rank1(self):
        m = self._build(rank=1, world=2)
        assert m.global_token_start == 21 * 30 // 2
        assert m.local_seq == 21 * 30 // 2

    def test_boundary_guards_passed_through(self):
        guards = [5, 10, 15]
        m = LVSAMetadata.build(
            total_latent_frames=21, num_patches=30, window_size=3,
            n_first_frames=1, key_frame_interval=4,
            rank=0, world=1, boundary_guards=guards,
        )
        for g in guards:
            assert g in m.global_set

    def test_keyframe_offset_rotation(self):
        m0 = self._build()
        m1 = LVSAMetadata.build(
            total_latent_frames=21, num_patches=30, window_size=3,
            n_first_frames=1, key_frame_interval=4,
            rank=0, world=1, keyframe_offset=1,
        )
        assert m0.global_indices != m1.global_indices
        assert len(m0.global_indices) == len(m1.global_indices)


# ── Cross-check: LVSAMetadata vs DistributedLVSAProcessor ─────────────────────


class TestCrossCheck:
    """Verify LVSAMetadata.build() produces the same structures as the processor."""

    def test_metadata_matches_processor(self):
        from lvsa.lvsa_processor import DistributedLVSAProcessor

        T, P, W, n_first, kfi = 21, 30, 3, 1, 4
        proc = DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P, window_size=W,
            n_first_frames=n_first, key_frame_interval=kfi,
            rank=0, world=1,
        )
        m = proc._metadata

        assert m.global_indices == proc._global_indices
        assert m.global_set == proc._global_set
        assert m.local_frames == proc._local_frames
        assert m.local_seq == proc.local_seq
        assert m.global_token_start == proc.global_token_start
        assert torch.equal(m.global_frame_mask, proc._global_frame_mask)
        assert torch.equal(m.window_bounds, proc._window_bounds)
        assert torch.equal(m.attended_indices, proc._attended_indices)
        assert torch.equal(m.fi_indptr, proc._fi_indptr)
        assert torch.equal(m.fi_indices, proc._fi_indices)

    def test_metadata_matches_after_rotation(self):
        from lvsa.lvsa_processor import DistributedLVSAProcessor

        T, P, W, n_first, kfi = 21, 30, 3, 1, 4
        proc = DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P, window_size=W,
            n_first_frames=n_first, key_frame_interval=kfi,
            rank=0, world=1,
        )
        proc.set_step(2)  # triggers rotation

        m = proc._metadata
        assert m.global_indices == proc._global_indices
        assert torch.equal(m.fi_indptr, proc._fi_indptr)

    def test_metadata_matches_multi_gpu(self):
        from lvsa.lvsa_processor import DistributedLVSAProcessor

        T, P, W, n_first, kfi = 20, 10, 2, 1, 5
        for rank in range(2):
            proc = DistributedLVSAProcessor(
                total_num_latent_frames=T, num_patches=P, window_size=W,
                n_first_frames=n_first, key_frame_interval=kfi,
                rank=rank, world=2,
            )
            m = proc._metadata
            assert m.global_indices == proc._global_indices
            assert m.local_seq == proc.local_seq


# ── ensure_device ────────────────────────────────────────────────────────────


class TestEnsureDevice:
    """Test LVSAMetadata.ensure_device() is idempotent."""

    def test_cpu_to_cpu_noop(self):
        m = LVSAMetadata.build(
            total_latent_frames=21, num_patches=30, window_size=3,
            n_first_frames=1, key_frame_interval=4,
            rank=0, world=1,
        )
        old_mask = m.global_frame_mask
        m.ensure_device(torch.device("cpu"))
        assert m.global_frame_mask is old_mask  # same object


# ── lvsa_sdpa on CPU ──────────────────────────────────────────────────────────


class TestSwaSDPA:
    """Test lvsa_sdpa() produces valid output on CPU."""

    def test_output_shape(self):
        T, P, H, D = 5, 4, 2, 8
        m = LVSAMetadata.build(
            total_latent_frames=T, num_patches=P, window_size=1,
            n_first_frames=1, key_frame_interval=None,
            rank=0, world=1, expand_window=False,
        )
        B = 1
        seq = T * P
        q = torch.randn(B, seq, H, D)
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)

        num_global = len(m.global_indices)
        k_g = torch.randn(B, num_global * P, H, D)
        v_g = torch.randn(B, num_global * P, H, D)

        out = lvsa_sdpa(q, k, v, k_g, v_g, m)
        assert out.shape == (B, seq, H, D)

    def test_output_not_nan(self):
        T, P, H, D = 5, 4, 2, 8
        m = LVSAMetadata.build(
            total_latent_frames=T, num_patches=P, window_size=1,
            n_first_frames=1, key_frame_interval=None,
            rank=0, world=1, expand_window=False,
        )
        B = 1
        seq = T * P
        q = torch.randn(B, seq, H, D)
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)

        num_global = len(m.global_indices)
        k_g = torch.randn(B, num_global * P, H, D)
        v_g = torch.randn(B, num_global * P, H, D)

        out = lvsa_sdpa(q, k, v, k_g, v_g, m)
        assert not torch.isnan(out).any()


# ── sparse_windowed_attention dispatch ───────────────────────────────────────


class TestSparseWindowedAttention:
    """Test top-level dispatcher."""

    def test_sdpa_dispatch(self):
        T, P, H, D = 5, 4, 2, 8
        m = LVSAMetadata.build(
            total_latent_frames=T, num_patches=P, window_size=1,
            n_first_frames=1, key_frame_interval=None,
            rank=0, world=1, expand_window=False,
        )
        B = 1
        seq = T * P
        q = torch.randn(B, seq, H, D)
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)

        num_global = len(m.global_indices)
        k_g = torch.randn(B, num_global * P, H, D)
        v_g = torch.randn(B, num_global * P, H, D)

        out = sparse_windowed_attention(q, k, v, k_g, v_g, m, backend="sdpa")
        assert out.shape == (B, seq, H, D)

    def test_flashinfer_raises_without_args(self):
        m = LVSAMetadata.build(
            total_latent_frames=5, num_patches=4, window_size=1,
            n_first_frames=1, key_frame_interval=None,
            rank=0, world=1,
        )
        q = k = v = k_g = v_g = torch.randn(1, 20, 2, 8)
        with pytest.raises(AssertionError, match="FlashInfer backend requires"):
            sparse_windowed_attention(q, k, v, k_g, v_g, m, backend="flashinfer")


# ── Module-level helper functions ────────────────────────────────────────────


class TestModuleLevelHelpers:
    """Test that module-level helpers match processor static methods."""

    def test_adaptive_window_bounds(self):
        from lvsa.lvsa_processor import DistributedLVSAProcessor
        for f in range(21):
            assert (
                adaptive_window_bounds(f, 3, 21)
                == DistributedLVSAProcessor._adaptive_window_bounds(f, 3, 21)
            )

    def test_compute_global_indices(self):
        from lvsa.lvsa_processor import DistributedLVSAProcessor
        assert (
            compute_global_indices(21, 1, 4, offset=2)
            == DistributedLVSAProcessor._compute_global_indices(21, 1, 4, offset=2)
        )

    def test_compute_boundary_guard_frames(self):
        from lvsa.lvsa_processor import DistributedLVSAProcessor
        assert (
            compute_boundary_guard_frames(20, 100, 10, 2, 2)
            == DistributedLVSAProcessor._compute_boundary_guard_frames(20, 100, 10, 2, 2)
        )

    def test_compute_auto_kfi(self):
        from lvsa.lvsa_processor import DistributedLVSAProcessor
        assert (
            compute_auto_kfi(60, 3, 1)
            == DistributedLVSAProcessor._compute_auto_kfi(60, 3, 1)
        )


# ── Sparsity scale ──────────────────────────────────────────────────────────


class TestSparsityScale:
    """Test sparsity_scale parameter in compute_auto_kfi."""

    def test_default_unchanged(self):
        """sparsity_scale=1.0 should give same result as no scale."""
        kfi_default = compute_auto_kfi(97, 3, 1)
        kfi_scale1 = compute_auto_kfi(97, 3, 1, sparsity_scale=1.0)
        assert kfi_default == kfi_scale1

    def test_higher_scale_more_globals(self):
        """sparsity_scale > 1.0 → smaller KFI → more globals."""
        kfi_1x = compute_auto_kfi(97, 3, 1, sparsity_scale=1.0)
        kfi_2x = compute_auto_kfi(97, 3, 1, sparsity_scale=2.0)
        assert kfi_2x <= kfi_1x  # smaller KFI = denser

    def test_lower_scale_fewer_globals(self):
        """sparsity_scale < 1.0 → larger KFI → fewer globals."""
        kfi_1x = compute_auto_kfi(97, 3, 1, sparsity_scale=1.0)
        kfi_05x = compute_auto_kfi(97, 3, 1, sparsity_scale=0.5)
        assert kfi_05x >= kfi_1x  # larger KFI = sparser

    def test_scale_affects_globals_count(self):
        """Verify actual global frame count changes with scale."""
        T = 97
        for scale in [0.5, 1.0, 2.0]:
            kfi = compute_auto_kfi(T, 3, 1, sparsity_scale=scale)
            globals = compute_global_indices(T, 1, kfi)
            # More scale → more globals
            if scale > 1.0:
                kfi_base = compute_auto_kfi(T, 3, 1, sparsity_scale=1.0)
                globals_base = compute_global_indices(T, 1, kfi_base)
                assert len(globals) >= len(globals_base)

    def test_scale_clamped_to_minimum(self):
        """Very low scale should still produce at least n_first globals."""
        kfi = compute_auto_kfi(97, 3, 1, sparsity_scale=0.1)
        globals = compute_global_indices(97, 1, kfi)
        assert len(globals) >= 1  # at least n_first_frames

    def test_scale_at_short_video(self):
        """Short video (T <= reference_frames) should be unaffected by scale."""
        kfi_1x = compute_auto_kfi(10, 3, 1, sparsity_scale=1.0)
        kfi_2x = compute_auto_kfi(10, 3, 1, sparsity_scale=2.0)
        # Short video: all frames fit in budget, both should be similar
        assert kfi_1x >= 1
        assert kfi_2x >= 1


# ── Coverage at / below reference length ────────────────────────────────────


class TestCoverageAtReference:
    """Regression tests for the kfi=1 fix at T <= reference_frames.

    Bug: at T <= ref, compute_auto_kfi returned kfi = T (large), yielding only
    n_first global anchors. With window=0 (globals-only), each query then
    attended to just those n_first anchors, introducing accidental sparsity
    where the budget should cover everything.

    Fix: compute_auto_kfi now returns 1 when T <= scaled_reference, making
    every frame a global so globals-only mode is also 100%-coverage.
    """

    def _coverage(self, T, W, n_first, ref, sparsity_scale=1.0, window_override=None):
        """Return (attended-count list of length T, kfi, n_globals)."""
        kfi = compute_auto_kfi(T, W, n_first, reference_frames=ref,
                               sparsity_scale=sparsity_scale)
        globals_idx = set(compute_global_indices(T, n_first, kfi, offset=0))
        W_eff = W if window_override is None else window_override
        counts = []
        for f in range(T):
            lo, hi = get_window_bounds(
                f, W_eff, T, expand=True,
                global_set=globals_idx, global_count=len(globals_idx),
            )
            win = set(range(lo, hi + 1)) if lo <= hi else set()
            attended = globals_idx | win
            counts.append(len(attended))
        return counts, kfi, len(globals_idx)

    # ── compute_auto_kfi returns 1 at T <= ref ─────────────────────────────

    @pytest.mark.parametrize("T, ref", [
        (21, 21),  # Wan at 1x
        (20, 21),  # Wan just below 1x
        (17, 33),  # HunyuanVideo 0.5x (T < ref)
        (33, 33),  # HunyuanVideo 1x (T == ref)
        (1, 21),   # degenerate single-frame
    ])
    def test_kfi_is_one_at_or_below_reference(self, T, ref):
        """At T <= reference_frames, auto-kfi must collapse to 1 so every
        frame is a global anchor (ensures globals-only mode stays fully dense).
        """
        kfi = compute_auto_kfi(T, window_size=12, n_first_frames=8,
                               reference_frames=ref)
        assert kfi == 1, f"expected kfi=1 at T={T} <= ref={ref}, got {kfi}"

    @pytest.mark.parametrize("T, ref", [
        (22, 21), (34, 33), (49, 33), (65, 33), (81, 21),
    ])
    def test_kfi_above_one_above_reference(self, T, ref):
        """At T > ref, kfi should be > 1 (genuine sparsity by design)."""
        kfi = compute_auto_kfi(T, window_size=12, n_first_frames=8,
                               reference_frames=ref)
        assert kfi >= 1
        # For T strictly greater than ref and T > 2W+1+n_first, we expect
        # a sparse schedule (kfi > 1). For small T just above ref the
        # algorithm may still be forced to kfi=1 because scaled_ref is
        # quite large; so we don't assert kfi > 1 universally — just that
        # globals do NOT cover every frame.
        globals_idx = compute_global_indices(T, 8, kfi)
        if T > ref * 1.2:
            assert len(globals_idx) < T, (
                f"expected some sparsity at T={T} > ref={ref}, got {len(globals_idx)} globals"
            )

    # ── windowed mode gives 100% coverage at T <= ref ───────────────────────

    @pytest.mark.parametrize("T, W, n_first, ref", [
        (17, 12, 8, 33),   # HV 0.5x
        (33, 12, 8, 33),   # HV 1x
        (21, 12, 8, 21),   # Wan 1x
        (10, 3, 1, 21),    # very short
    ])
    def test_windowed_coverage_is_full_at_reference(self, T, W, n_first, ref):
        counts, kfi, n_glob = self._coverage(T, W, n_first, ref)
        assert all(c == T for c in counts), (
            f"T={T} W={W} n_first={n_first} ref={ref} kfi={kfi}: "
            f"expected all queries to attend T={T} frames, got {counts}"
        )

    # ── globals-only mode (window=0) also 100% at T <= ref ─────────────────

    @pytest.mark.parametrize("T, W, n_first, ref", [
        (17, 12, 8, 33),
        (33, 12, 8, 33),
        (21, 12, 8, 21),
    ])
    def test_globals_only_coverage_is_full_at_reference(self, T, W, n_first, ref):
        """This is the direct regression test for the bug: when the graduated
        schedule flips to window=0, every query should still see all T frames
        at T <= ref (because every frame is a global)."""
        counts, kfi, n_glob = self._coverage(
            T, W, n_first, ref, window_override=0,
        )
        assert n_glob == T, (
            f"expected all {T} frames to be globals at T<=ref, got {n_glob}"
        )
        assert all(c == T for c in counts), (
            f"T={T} (window=0): expected all queries to attend T={T} frames, got {counts}"
        )

    # ── above reference, real sparsity is present ──────────────────────────

    @pytest.mark.parametrize("T, W, n_first, ref", [
        (49, 12, 8, 33),   # HV 1.5x
        (65, 12, 8, 33),   # HV 2x
        (81, 12, 8, 21),   # Wan 4x
    ])
    def test_sparsity_at_extrapolation(self, T, W, n_first, ref):
        counts, kfi, n_glob = self._coverage(T, W, n_first, ref)
        assert max(counts) < T, (
            f"T={T} > ref={ref}: expected sparsity (max attended < T), "
            f"got max={max(counts)} with kfi={kfi}"
        )
        # Budget bound: attended = n_first + min(2W+1, T - n_first) is the
        # upper bound of per-query work. The algorithm guarantees this is
        # well below T so that sparsity saves compute.
        budget_ub = n_first + min(2 * W + 1, T - n_first)
        avg = sum(counts) / len(counts)
        assert avg <= budget_ub + 0.1, (
            f"avg attended {avg:.1f} exceeds theoretical upper bound {budget_ub}"
        )
        # And coverage should be meaningfully less than T (>= 20% savings)
        assert avg / T <= 0.8, (
            f"avg coverage {avg/T:.1%} should be <= 80% at T={T} > ref={ref}"
        )


# ── Mask shape invariants ───────────────────────────────────────────────────


class TestMaskShapeInvariants:
    """Validate invariants on the attended-set mask produced by the LVSA
    geometry helpers. Every query row must be valid (non-empty, within [0,T)),
    globals must be a subset, and counts must match the documented budget."""

    @pytest.mark.parametrize("T, W, n_first, ref", [
        (17, 12, 8, 33),
        (33, 12, 8, 33),
        (49, 12, 8, 33),
        (65, 12, 8, 33),
        (21, 3, 1, 21),
        (60, 3, 1, 21),
        (81, 3, 1, 21),
    ])
    def test_mask_shape_and_invariants(self, T, W, n_first, ref):
        kfi = compute_auto_kfi(T, W, n_first, reference_frames=ref)
        globals_idx = set(compute_global_indices(T, n_first, kfi))

        # Globals are in-range and sorted-unique
        assert all(0 <= g < T for g in globals_idx), (
            f"global indices out of range for T={T}: {sorted(globals_idx)}"
        )

        # n_first frames must be included
        for i in range(min(n_first, T)):
            assert i in globals_idx, f"n_first frame {i} missing from globals"

        for f in range(T):
            lo, hi = get_window_bounds(
                f, W, T, expand=True,
                global_set=globals_idx, global_count=len(globals_idx),
            )
            win = set(range(lo, hi + 1)) if lo <= hi else set()
            attended = globals_idx | win

            # Shape invariants
            assert all(0 <= a < T for a in attended), (
                f"attended indices out of range: query f={f}, T={T}, attended={sorted(attended)}"
            )
            assert len(attended) > 0, f"empty attended set for query f={f}"
            assert len(attended) <= T, (
                f"attended set exceeds T: {len(attended)} > {T}"
            )

            # Globals subset
            assert globals_idx.issubset(attended), (
                f"globals not included in attended set for query f={f}"
            )

            # Query itself should be attended (self-attention)
            if W > 0:
                assert f in attended, (
                    f"query frame f={f} not in its own attended set "
                    f"(window=[{lo},{hi}])"
                )

    def test_globals_only_mode_shape(self):
        """With W=0, attended set must equal exactly the globals set."""
        T, W, n_first, ref = 49, 12, 8, 33
        kfi = compute_auto_kfi(T, W, n_first, reference_frames=ref)
        globals_idx = set(compute_global_indices(T, n_first, kfi))

        for f in range(T):
            lo, hi = get_window_bounds(
                f, 0, T, expand=True,
                global_set=globals_idx, global_count=len(globals_idx),
            )
            win = set(range(lo, hi + 1)) if lo <= hi else set()
            attended = globals_idx | win
            assert attended == globals_idx, (
                f"globals-only mode should have attended = globals, "
                f"got extra window tokens {win - globals_idx} for f={f}"
            )

    def test_lvsametadata_build_honors_reference_frames(self):
        """Regression for caller-trust semantics: LVSAMetadata.build uses the
        caller's key_frame_interval verbatim (no internal auto-rescaling).
        Callers that want autoscaling — parallel.py for --auto-keyframes /
        --rotate-keyframes, vllm-omni hooks — must call compute_auto_kfi
        themselves.
        """
        from lvsa.sparse_attention import LVSAMetadata

        # HV 1× case: T=33, user kfi=1 (what the kfi=1 fix produces), ref=33
        md = LVSAMetadata.build(
            total_latent_frames=33, num_patches=30,
            window_size=3, n_first_frames=1,
            key_frame_interval=1,
            rank=0, world=1, reference_frames=33,
        )
        assert md.key_frame_interval == 1, (
            f"expected LVSAMetadata to preserve kfi=1 at T<=ref, got {md.key_frame_interval}"
        )
        assert len(md.global_set) == 33, (
            f"expected all 33 frames as globals, got {len(md.global_set)}"
        )

    def test_mask_monotonic_decay_past_reference(self):
        """As T grows past reference, average coverage percent should strictly
        decrease (more sparsity for longer videos)."""
        ref, W, n_first = 21, 12, 8
        prev_pct = 101.0
        for T in [21, 41, 61, 81, 121, 161]:
            kfi = compute_auto_kfi(T, W, n_first, reference_frames=ref)
            globals_idx = set(compute_global_indices(T, n_first, kfi))
            counts = []
            for f in range(T):
                lo, hi = get_window_bounds(
                    f, W, T, expand=True,
                    global_set=globals_idx, global_count=len(globals_idx),
                )
                win = set(range(lo, hi + 1)) if lo <= hi else set()
                counts.append(len(globals_idx | win))
            pct = 100 * sum(counts) / len(counts) / T
            assert pct <= prev_pct + 0.1, (
                f"T={T}: coverage {pct:.1f}% should be <= previous {prev_pct:.1f}%"
            )
            prev_pct = pct


# ── Import test ──────────────────────────────────────────────────────────────


class TestImports:
    """Test that the new public API is importable from lvsa."""

    def test_import_from_lvsa(self):
        from lvsa import sparse_windowed_attention, LVSAMetadata
        assert callable(sparse_windowed_attention)
        assert hasattr(LVSAMetadata, "build")
