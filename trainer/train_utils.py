"""
Pure utility functions for semi-supervised 3D training.

These functions have no dependencies on trainer state and can be
imported and tested independently.
"""
import numpy as np
import torch


def normalize_slice_to_01(slice_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a 2-D (H, W) tensor to [0, 1] using 1st/99th percentile."""
    arr = slice_tensor.detach().float().cpu().numpy()
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return torch.from_numpy(arr).float()


def sharpening(P: torch.Tensor, T: float = 10.0) -> torch.Tensor:
    """Apply sharpening to a probability tensor (temperature T = 1/0.1 = 10)."""
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen
