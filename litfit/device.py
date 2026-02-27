from typing import Union

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
DTYPE = torch.float32


def to_torch(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=DEVICE, dtype=DTYPE)
    return torch.tensor(np.asarray(x), device=DEVICE, dtype=DTYPE)


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _eye(d: int) -> torch.Tensor:
    return torch.eye(d, device=DEVICE, dtype=DTYPE)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows of x."""
    norms = x.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return x / norms
