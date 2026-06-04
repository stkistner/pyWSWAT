from __future__ import annotations

import numpy as np


def swap_last_two(arr: np.ndarray) -> np.ndarray:
    """Swap the last two axes: (..., a, b) → (..., b, a)."""
    axes = list(range(arr.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def arr_or_none(value: object) -> np.ndarray | None:
    """Convert *value* to a float numpy array, or return None if empty/None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.size > 0 else None
