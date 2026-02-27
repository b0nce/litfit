"""Tests for device utilities."""

import numpy as np
import torch

from litfit.device import DEVICE, DTYPE, _eye, _normalize, to_numpy, to_torch


class TestToTorch:
    def test_numpy_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = to_torch(arr)
        assert isinstance(t, torch.Tensor)
        assert t.device == DEVICE
        assert t.dtype == DTYPE
        assert t.shape == (2, 2)
        np.testing.assert_allclose(t.cpu().numpy(), arr)

    def test_numpy_float64_converts_dtype(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        t = to_torch(arr)
        assert t.dtype == DTYPE

    def test_torch_tensor_on_device(self):
        t_in = torch.tensor([1.0, 2.0])
        t_out = to_torch(t_in)
        assert t_out.device == DEVICE
        assert t_out.dtype == DTYPE

    def test_torch_tensor_wrong_dtype(self):
        t_in = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t_out = to_torch(t_in)
        assert t_out.dtype == DTYPE

    def test_list_input(self):
        t = to_torch([1.0, 2.0, 3.0])
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)


class TestToNumpy:
    def test_from_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0], device=DEVICE, dtype=DTYPE)
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_from_numpy(self):
        arr_in = np.array([1.0, 2.0])
        arr_out = to_numpy(arr_in)
        assert isinstance(arr_out, np.ndarray)
        np.testing.assert_array_equal(arr_in, arr_out)

    def test_on_cpu(self):
        t = torch.tensor([1.0], device=DEVICE)
        arr = to_numpy(t)
        # Should always return a CPU numpy array
        assert isinstance(arr, np.ndarray)


class TestEye:
    def test_shape(self):
        I = _eye(5)  # noqa: E741
        assert I.shape == (5, 5)

    def test_identity(self):
        I = _eye(3)  # noqa: E741
        expected = torch.eye(3, device=DEVICE, dtype=DTYPE)
        assert torch.equal(I, expected)

    def test_device_and_dtype(self):
        I = _eye(4)  # noqa: E741
        assert I.device == DEVICE
        assert I.dtype == DTYPE


class TestNormalize:
    def test_unit_norms(self):
        x = torch.randn(10, 5, device=DEVICE, dtype=DTYPE)
        xn = _normalize(x)
        norms = xn.norm(dim=1)
        assert torch.allclose(norms, torch.ones(10, device=DEVICE, dtype=DTYPE), atol=1e-6)

    def test_shape_preserved(self):
        x = torch.randn(3, 7, device=DEVICE, dtype=DTYPE)
        xn = _normalize(x)
        assert xn.shape == (3, 7)

    def test_zero_row_handled(self):
        x = torch.zeros(1, 4, device=DEVICE, dtype=DTYPE)
        xn = _normalize(x)
        # Should not produce NaN due to clamp(min=1e-10)
        assert torch.isfinite(xn).all()

    def test_direction_preserved(self):
        x = torch.tensor([[3.0, 4.0]], device=DEVICE, dtype=DTYPE)
        xn = _normalize(x)
        expected = torch.tensor([[0.6, 0.8]], device=DEVICE, dtype=DTYPE)
        assert torch.allclose(xn, expected, atol=1e-6)
