"""Core FIR algorithms."""

from __future__ import annotations

from warnings import warn

import numpy as np
from scipy.signal import firwin, lfilter


def fir_filter(
    data: np.ndarray,
    dt: float,
    window_size: int = 1025,
    fmax: float | None = None,
    fmin: float | None = None,
    window: str = "hamming",
    axis: int = 0,
) -> tuple[np.ndarray, float]:
    """Generate and apply a Finite Impulse Response filter to an array.

    Uses ``scipy.signal.firwin`` to design the filter and ``scipy.signal.lfilter``
    to apply it.  Supports low-pass, high-pass, band-pass, and band-stop modes.

    Parameters
    ----------
    data : np.ndarray
        Input array.  The time axis is selected by *axis*.
    dt : float
        Timestep in seconds.
    window_size : int
        Number of FIR taps.  An odd value avoids a half-sample time offset.
    fmax : float or None
        Low-pass cutoff frequency (Hz).  ``None`` means no low-pass.
    fmin : float or None
        High-pass cutoff frequency (Hz).  ``None`` means no high-pass.
    window : str
        Scipy window type passed to :func:`scipy.signal.firwin`.
        Default ``"hamming"``.
    axis : int
        Axis of *data* along which to apply the filter (the time axis).
        Default ``0``.

    Returns
    -------
    filtered : np.ndarray
        Filtered array with the first ``window_size - 1`` samples removed
        along *axis* to discard the filter startup transient.
    delay : float
        Phase delay in seconds (``0.5 * (window_size - 1) * dt``).  Add this
        to the original time axis after slicing off the first
        ``window_size - 1`` steps to obtain the corrected time axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(2048) * 0.5          # 2048 steps, dt=0.5 s
    >>> x = np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.1, t.size)
    >>> filtered, delay = fir_filter(x, dt=0.5, window_size=65, fmax=0.2)
    >>> filtered.shape[0] == x.shape[0] - 64
    True
    >>> round(delay, 4)
    16.0
    """
    if window_size % 2 == 0:
        warn("window_size is even; time axis will be offset by half a timestep")

    if window_size >= data.shape[axis]:
        raise ValueError(
            f"window_size ({window_size}) must be less than the length of the "
            f"time axis ({data.shape[axis]})"
        )

    sample_rate = 1.0 / dt
    nyq_rate = sample_rate / 2.0

    if fmax is not None and (fmax <= 0 or fmax >= nyq_rate):
        raise ValueError(f"fmax must be in (0, {nyq_rate}); got {fmax}")
    if fmin is not None and (fmin <= 0 or fmin >= nyq_rate):
        raise ValueError(f"fmin must be in (0, {nyq_rate}); got {fmin}")

    if fmax is None and fmin is None:
        warn("No cutoff frequencies specified; returning data unchanged")
        return data, 0.0

    if fmax is not None and fmin is None:
        f_: float | np.ndarray = fmax
        pass_zero = True
    elif fmax is None and fmin is not None:
        f_ = fmin
        pass_zero = False
    else:
        assert fmax is not None and fmin is not None
        if fmax > fmin:  # band-pass
            f_ = np.array([fmin, fmax])
            pass_zero = False
        else:  # band-stop
            f_ = np.array([fmax, fmin])
            pass_zero = True

    taps = firwin(window_size, f_ / nyq_rate, window=window, pass_zero=pass_zero)
    delay = 0.5 * (window_size - 1) * dt

    filtered = np.apply_along_axis(lambda x: lfilter(taps, 1, x), axis=axis, arr=data)
    truncated = np.take(filtered, range(window_size - 1, data.shape[axis]), axis=axis)

    return truncated, delay
