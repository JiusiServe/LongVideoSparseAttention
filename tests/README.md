# LGAA Test Suite

## Local setup (CPU-only)

```bash
pip install -r tests/requirements.txt
```

This installs the CPU-only build of PyTorch and pytest — no CUDA or GPU needed.

## Running

```bash
python -m pytest tests/ -v
```

## Coverage

| File | Tests | Covers |
|------|-------|--------|
| `test_rope.py` | 15 | `apply_rotary_emb` (shape, dtype, identity, determinism, broadcast), `slice_rotary_emb` (rank slicing, full reconstruction, contiguity, no-match passthrough) |
| `test_lvsa_processor.py` | 40+ | `_adaptive_window_bounds` (center/edge/constant-width/small-T), `_expanded_window_bounds` (non-global target count), `_compute_boundary_guard_frames` (single/multi GPU, sorted, in-range), `_compute_global_indices` (first frames, periodic, offset rotation, constant count, full coverage), `_auto_key_frame_interval`, `_compute_auto_kfi`, `__init__` (local_seq, global_token_start, local_frames, window_ctx, masks, index tensors, divisibility assertion), `set_window_size`/`set_step` (dynamic reconfiguration, idempotency), `_build_flashinfer_csr` (indptr/indices validity, compact coverage, CSR rebuild), `print_attention_mask`/`print_attention_mask_compact`, edge cases (T=1, W>T, all-global, kfi=T) |
| `test_parallel.py` | 7 | `compute_and_validate_seq_len` (standard 480x832, multi-GPU divisibility, assertion on indivisible, scalar patch_size, T_lat formula, spatial independence from frame count) |

All tests are CPU-only (no GPU or `torch.distributed` required).
