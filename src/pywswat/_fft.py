"""Core FFT and cross-spectral density algorithms."""

from __future__ import annotations

import numpy as np
from scipy.signal import csd, periodogram, welch

import mikeio


def _dt_seconds(da: mikeio.DataArray) -> float:
    """Return the timestep in seconds from a mikeio DataArray."""
    t = da.time
    if len(t) < 2:
        raise ValueError("DataArray must have at least 2 time steps.")
    return float((t[1] - t[0]).total_seconds())


def _fft_spectrum(
    values: np.ndarray,
    dt: float,
    truncate_f0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD via a single periodogram (rectangular window).

    Parameters
    ----------
    values : (N,) float
        Time series with the mean already removed (or detrend handled here).
    dt : float
        Timestep in seconds.
    truncate_f0 : bool
        Drop the zero-frequency component.

    Returns
    -------
    freqs : (n_freqs,) Hz
    psd   : (n_freqs,) m²/Hz
    """
    freqs, psd = periodogram(values, fs=1.0 / dt, window="boxcar", detrend="constant")
    if truncate_f0:
        freqs, psd = freqs[1:], psd[1:]
    return freqs, psd


def _welch_spectrum(
    values: np.ndarray,
    dt: float,
    window_size: int,
    overlap: float,
    window: str,
    truncate_f0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD via Welch's averaged periodogram.

    Parameters
    ----------
    values : (N,) float
    dt : float, seconds
    window_size : int
    overlap : float  0 ≤ overlap < 1
    window : str  window function name for scipy

    Returns
    -------
    freqs : (n_freqs,) Hz
    psd   : (n_freqs,) m²/Hz
    """
    noverlap = int(window_size * overlap)
    freqs, psd = welch(
        values,
        fs=1.0 / dt,
        window=window,
        nperseg=window_size,
        noverlap=noverlap,
        detrend="constant",
    )
    if truncate_f0:
        freqs, psd = freqs[1:], psd[1:]
    return freqs, psd


def _csd_welch(
    vals1: np.ndarray,
    vals2: np.ndarray,
    dt: float,
    window_size: int,
    overlap: float,
    window: str,
    truncate_f0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-spectral density via Welch's method.

    Returns
    -------
    freqs : (n_freqs,) Hz
    cross : (n_freqs,) complex  m²/Hz
    """
    noverlap = int(window_size * overlap)
    freqs, cross = csd(
        vals1,
        vals2,
        fs=1.0 / dt,
        window=window,
        nperseg=window_size,
        noverlap=noverlap,
        detrend="constant",
    )
    if truncate_f0:
        freqs, cross = freqs[1:], cross[1:]
    return freqs, cross


def directional_fourier_coeffs(
    eta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    window_size: int = 512,
    overlap: float = 0.5,
    window: str = "hann",
    meteo_dir=True,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalised directional Fourier coefficients from η, u, v.

    Following Kuik et al. (1988):

    * ``a1 = Re(C_{ηu}) · (ω/g) / S_{ηη}``
    * ``b1 = Re(C_{ηv}) · (ω/g) / S_{ηη}``
    * ``a2 = (S_{uu} − S_{vv}) · (ω/g)² / S_{ηη}``
    * ``b2 = 2 · Re(C_{uv}) · (ω/g)² / S_{ηη}``

    Parameters
    ----------
    eta, u, v : (N,) float
        Surface elevation [m] and depth-averaged horizontal velocities [m/s].
    dt : float
        Timestep [s].
    window_size, overlap, window :
        Welch parameters (same as :func:`_welch_spectrum`).
    meteo_dir : bool
        If True, return coefficients for meteorological convention (0° = N, +ve clockwise).
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    freqs : (n_freqs,) Hz
    S_ee  : (n_freqs,) m²/Hz  — surface elevation auto-spectrum
    a1, b1, a2, b2 : (n_freqs,)  — directional Fourier moments ∈ [−1, 1]
    """
    # Auto-spectra
    freqs, S_ee = _welch_spectrum(eta, dt, window_size, overlap, window)  # C00
    _, S_uu = _welch_spectrum(u, dt, window_size, overlap, window)  # C11
    _, S_vv = _welch_spectrum(v, dt, window_size, overlap, window)  # C22

    # Cross-spectra (same frequency axis, truncate_f0=True by default)
    _, C_eu = _csd_welch(eta, u, dt, window_size, overlap, window)  # C01
    _, C_ev = _csd_welch(eta, v, dt, window_size, overlap, window)  # C02
    _, C_uv = _csd_welch(u, v, dt, window_size, overlap, window)  # C12

    # omega = 2.0 * np.pi * freqs
    # k_star = omega / g          # deep-water transfer factor (ω/g)
    k_star = 1 / (((S_uu + S_vv) / S_ee) ** 0.5)
    k_star_sq = k_star**2

    # Guard division by zero at very low S_ee
    S_safe = np.where(S_ee > 0, S_ee, np.nan)

    # First directional moments
    a1 = np.real(C_eu) * k_star / S_safe  # C01.real * k_star / S00
    b1 = np.real(C_ev) * k_star / S_safe  # C02.real * k_star / S00

    # Second directional moments
    a2 = (S_uu - S_vv) * k_star_sq / S_safe  # (C11 - C22) * k_star^2 / S00
    b2 = 2.0 * np.real(C_uv) * k_star_sq / S_safe  # 2 * C12.real * k_star^2 / S00

    # Physical bounds
    a1 = np.clip(a1, -1.0, 1.0)
    b1 = np.clip(b1, -1.0, 1.0)
    r2 = np.sqrt(a2**2 + b2**2)
    excess = r2 > 1.0
    a2 = np.where(excess, a2 / r2, a2)
    b2 = np.where(excess, b2 / r2, b2)

    if meteo_dir:
        return freqs, S_ee, -b1, -a1, -a2, b2
    else:
        return freqs, S_ee, a1, b1, a2, b2
