"""Tests for lvsa.device backend selection."""

from __future__ import annotations

from unittest import mock

import torch

from lvsa.device import (
    enable_fast_matmul,
    get_device,
    get_distributed_backend,
)


class TestGetDevice:
    def test_returns_torch_device(self):
        d = get_device(0)
        assert isinstance(d, torch.device)

    def test_cuda_path_when_available(self):
        if not torch.cuda.is_available():
            return
        d = get_device(0)
        assert d.type == "cuda"
        assert d.index == 0

    def test_cpu_when_nothing_available(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("lvsa.device._npu_available", return_value=False):
            d = get_device(0)
            assert d.type == "cpu"

    def test_npu_branch_calls_set_device(self):
        # Verify the NPU code path is taken when only NPU is available.
        # Constructing torch.device("npu", ...) only works after torch_npu
        # has registered the device type, so on a CUDA-only host we just
        # confirm set_device was dispatched and let the construction
        # raise — the surrounding hardware test covers the real device.
        npu_module = mock.MagicMock()
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("lvsa.device._npu_available", return_value=True), \
             mock.patch.object(torch, "npu", npu_module, create=True):
            try:
                get_device(2)
            except RuntimeError:
                pass  # torch.device("npu", ...) unavailable on CUDA host
            npu_module.set_device.assert_called_once_with(2)


class TestGetDistributedBackend:
    def test_nccl_on_cuda(self):
        if not torch.cuda.is_available():
            return
        assert get_distributed_backend() == "nccl"

    def test_hccl_on_npu_only(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("lvsa.device._npu_available", return_value=True):
            assert get_distributed_backend() == "hccl"

    def test_gloo_on_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("lvsa.device._npu_available", return_value=False):
            assert get_distributed_backend() == "gloo"


class TestEnableFastMatmul:
    def test_noop_on_cpu(self):
        # Just ensure the call doesn't blow up when CUDA is unavailable.
        with mock.patch("torch.cuda.is_available", return_value=False):
            enable_fast_matmul()  # no exception
