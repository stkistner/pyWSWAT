"""Spectral estimation from time series.

Public functions
----------------
:func:`spectrum`
    One-sided PSD via a single FFT of the entire record.
:func:`spectrum_welch`
    One-sided PSD via Welch's averaged periodogram.
:func:`spectrum_mem`
    2-D directional spectrum via cross-spectral analysis and MEM.
    Supports point, Grid1D, Grid2D, area (element), and line (node) input
    geometries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mikeio
from tqdm import tqdm
from typing import TYPE_CHECKING

from pywswat.spectral._fft import (
    _dt_seconds,
    _fft_spectrum,
    _welch_spectrum,
    directional_fourier_coeffs,
)
from pywswat.spectral._mem import mem_lk86, mem_newton
from pywswat.core import SpectraArray
from pywswat.mikeio._adapters import (
    _is_grid_geom,
    _grid_spatial_shape,
    _grid_to_spectral_geom,
    _to_spectral_geom,
    _extract_coords,
)

if TYPE_CHECKING:
    from pywswat.collection import SpectraSet

# Dimension names that indicate an FM-mesh spatial (location) axis
_LOC_DIMS = frozenset({"element", "node"})


# ──────────────────────────────────────────────────────────────────────────────
# FM-mesh spatial helpers
# ──────────────────────────────────────────────────────────────────────────────


def _is_spatial(da: mikeio.DataArray) -> bool:
    """Return True when *da* has a recognised FM-mesh spatial dimension."""
    return bool(_LOC_DIMS & set(da.dims))


def _loc_axis_info(da: mikeio.DataArray) -> tuple[str, int, int]:
    """Return ``(dim_name, axis_index, n_locs)`` for the FM location axis."""
    dims = list(da.dims)
    for d in dims:
        if d in _LOC_DIMS:
            ax = dims.index(d)
            return d, ax, da.values.shape[ax]
    raise ValueError(
        f"No spatial dimension found in dims {da.dims!r}. Expected one of {_LOC_DIMS}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────────────────────────────────────


def _setup_time(da: mikeio.DataArray) -> pd.DatetimeIndex:
    """Setup timestamp of a DataArray's time axis."""
    t = da.time
    # return pd.DatetimeIndex([t[0] + (t[-1] - t[0]) / 2])
    return [t[0]]


# ──────────────────────────────────────────────────────────────────────────────
# MEM helper
# ──────────────────────────────────────────────────────────────────────────────


def _apply_mem(
    theta_rad: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply MEM estimator and return D (n_freqs, n_dirs) in rad⁻¹."""
    if method == "lk86":
        return mem_lk86(theta_rad, a1, b1, a2, b2)
    if method == "newton":
        return mem_newton(theta_rad, a1, b1, a2, b2)
    raise ValueError(f"Unknown MEM method {method!r}. Use 'lk86' or 'newton'.")


# ──────────────────────────────────────────────────────────────────────────────
# Core builders — produce SpectraArray directly in pywswat axis order
# ──────────────────────────────────────────────────────────────────────────────


def _build_point_spectrum(
    psd: np.ndarray,
    freqs: np.ndarray,
    time: pd.DatetimeIndex,
    name: str,
    directions: np.ndarray | None = None,
    io_geom: object | None = None,
) -> SpectraArray:
    """Build a *point* SpectraArray."""
    return SpectraArray(
        psd,
        freq=freqs,
        direction=directions,
        time=time,
        name=name,
        _io_geom=io_geom,
    )


def _build_spatial_spectrum(
    spectra: np.ndarray,
    freqs: np.ndarray,
    directions: np.ndarray | None,
    time: pd.DatetimeIndex,
    io_geom: object,
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: "tuple[int, ...] | None" = None,
) -> SpectraArray:
    """Build a spatial SpectraArray.

    Parameters
    ----------
    grid_shape :
        Original spatial grid dimensions (e.g. ``(ny, nx)`` for Grid2D).
        When set, :attr:`~pywswat.core.SpectraArray.data` and
        :meth:`~pywswat.core.SpectraArray.to_params` return arrays reshaped
        to ``(*grid_shape, ...)`` instead of ``(nloc, ...)``.
    """
    return SpectraArray(
        spectra,
        freq=freqs,
        direction=directions,
        time=time,
        x=x,
        y=y,
        name="Energy density",
        _io_geom=io_geom,
        _grid_shape=grid_shape,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Point-only spectrum helpers
# ──────────────────────────────────────────────────────────────────────────────


def _single_spectrum(da: mikeio.DataArray, truncate_f0: bool) -> SpectraArray:
    if _is_grid_geom(da):
        return _grid_spectrum(da, truncate_f0)
    if da.values.ndim != 1:
        raise ValueError(
            f"spectrum() requires a 1-D (time-only) DataArray, "
            f"got dims {da.dims!r}. "
            "Use .isel() to select a single location first (FM mesh), "
            "or pass a Grid1D/Grid2D DataArray for all-location computation."
        )
    dt = _dt_seconds(da)
    freqs, psd = _fft_spectrum(np.asarray(da.values, dtype=float), dt, truncate_f0)
    io_geom = mikeio.spatial.GeometryFMPointSpectrum(frequencies=freqs)
    return _build_point_spectrum(
        psd, freqs, _setup_time(da), name=da.name, io_geom=io_geom
    )


def _single_welch(
    da: mikeio.DataArray,
    window_size: int,
    overlap: float,
    window: str,
    truncate_f0: bool,
) -> SpectraArray:
    if _is_grid_geom(da):
        return _grid_welch(da, window_size, overlap, window, truncate_f0)
    if da.values.ndim != 1:
        raise ValueError(
            f"spectrum_welch() requires a 1-D (time-only) DataArray, "
            f"got dims {da.dims!r}. "
            "Use .isel() to select a single location first (FM mesh), "
            "or pass a Grid1D/Grid2D DataArray for all-location computation."
        )
    dt = _dt_seconds(da)
    freqs, psd = _welch_spectrum(
        np.asarray(da.values, dtype=float),
        dt,
        window_size,
        overlap,
        window,
        truncate_f0,
    )
    io_geom = mikeio.spatial.GeometryFMPointSpectrum(frequencies=freqs)
    return _build_point_spectrum(
        psd, freqs, _setup_time(da), name=da.name, io_geom=io_geom
    )


# ──────────────────────────────────────────────────────────────────────────────
# Grid spectrum helpers (Grid1D / Grid2D)
# ──────────────────────────────────────────────────────────────────────────────


def _grid_spectrum(da: mikeio.DataArray, truncate_f0: bool) -> SpectraArray:
    """Compute 1-D FFT spectrum at every Grid1D/Grid2D location."""
    geom = da.geometry
    spatial_shape = _grid_spatial_shape(geom)
    n_locs = int(np.prod(spatial_shape))
    dt = _dt_seconds(da)

    vals = np.asarray(da.values, dtype=float).reshape(-1, n_locs).T

    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    for i in tqdm(range(n_locs), desc="Computing spectrum at grid locations"):
        f, psd = _fft_spectrum(vals[i], dt, truncate_f0)
        if freqs is None:
            freqs = f
        psds.append(psd)

    assert freqs is not None
    S_stack = np.stack(psds, axis=0)
    line_geom = _grid_to_spectral_geom(geom, freqs, None)
    x, y = line_geom.node_coordinates[:, 0], line_geom.node_coordinates[:, 1]
    return _build_spatial_spectrum(
        S_stack, freqs, None, _setup_time(da), geom, x, y, grid_shape=spatial_shape
    )


def _grid_welch(
    da: mikeio.DataArray,
    window_size: int,
    overlap: float,
    window: str,
    truncate_f0: bool,
) -> SpectraArray:
    """Compute Welch spectrum at every Grid1D/Grid2D location."""
    geom = da.geometry
    spatial_shape = _grid_spatial_shape(geom)
    n_locs = int(np.prod(spatial_shape))
    dt = _dt_seconds(da)

    vals = np.asarray(da.values, dtype=float).reshape(-1, n_locs).T

    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    for i in tqdm(range(n_locs), desc="Computing Welch spectrum at grid locations"):
        f, psd = _welch_spectrum(vals[i], dt, window_size, overlap, window, truncate_f0)
        if freqs is None:
            freqs = f
        psds.append(psd)

    assert freqs is not None
    S_stack = np.stack(psds, axis=0)
    line_geom = _grid_to_spectral_geom(geom, freqs, None)
    x, y = line_geom.node_coordinates[:, 0], line_geom.node_coordinates[:, 1]
    return _build_spatial_spectrum(
        S_stack, freqs, None, _setup_time(da), geom, x, y, grid_shape=spatial_shape
    )


def _mem_spectrum_grid(
    eta_da: mikeio.DataArray,
    u_da: mikeio.DataArray,
    v_da: mikeio.DataArray,
    theta_rad: np.ndarray,
    directions: np.ndarray,
    window_size: int,
    overlap: float,
    window: str,
    method: str,
    g: float,
) -> SpectraArray:
    """Compute MEM spectrum at every Grid1D/Grid2D location."""
    geom = eta_da.geometry
    spatial_shape = _grid_spatial_shape(geom)
    n_locs = int(np.prod(spatial_shape))
    dt = _dt_seconds(eta_da)

    eta_vals = np.asarray(eta_da.values, dtype=float).reshape(-1, n_locs)
    u_vals = np.asarray(u_da.values, dtype=float).reshape(-1, n_locs)
    v_vals = np.asarray(v_da.values, dtype=float).reshape(-1, n_locs)

    spectra: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    for i in tqdm(range(n_locs), desc="Computing MEM spectrum at grid locations"):
        f, S_ee, a1, b1, a2, b2 = directional_fourier_coeffs(
            eta_vals[:, i],
            u_vals[:, i],
            v_vals[:, i],
            dt,
            window_size,
            overlap,
            window,
            g=g,
        )
        if freqs is None:
            freqs = f
        D = _apply_mem(theta_rad, a1, b1, a2, b2, method)
        spectra.append(S_ee[:, None] * D)

    S_stack = np.stack(spectra, axis=0)

    assert freqs is not None
    line_geom = _grid_to_spectral_geom(geom, freqs, directions)
    x, y = line_geom.node_coordinates[:, 0], line_geom.node_coordinates[:, 1]
    return _build_spatial_spectrum(
        S_stack,
        freqs,
        directions,
        _setup_time(eta_da),
        geom,
        x,
        y,
        grid_shape=spatial_shape,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MEM spectrum helpers (point and FM spatial)
# ──────────────────────────────────────────────────────────────────────────────


def _mem_spectrum_point(
    eta_da: mikeio.DataArray,
    u_da: mikeio.DataArray,
    v_da: mikeio.DataArray,
    theta_rad: np.ndarray,
    directions: np.ndarray,
    window_size: int,
    overlap: float,
    window: str,
    method: str,
    g: float,
) -> SpectraArray:
    """Compute MEM spectrum at a single (point) location."""
    dt = _dt_seconds(eta_da)
    eta = np.asarray(eta_da.values, dtype=float).ravel()
    u = np.asarray(u_da.values, dtype=float).ravel()
    v = np.asarray(v_da.values, dtype=float).ravel()

    freqs, S_ee, a1, b1, a2, b2 = directional_fourier_coeffs(
        eta, u, v, dt, window_size, overlap, window, g=g
    )
    D = _apply_mem(theta_rad, a1, b1, a2, b2, method)
    S_pywswat = S_ee[:, None] * D

    io_geom = mikeio.spatial.GeometryFMPointSpectrum(
        frequencies=freqs, directions=directions
    )
    return _build_point_spectrum(
        S_pywswat,
        freqs,
        _setup_time(eta_da),
        name="Energy density",
        directions=directions,
        io_geom=io_geom,
    )


def _mem_spectrum_spatial(
    eta_da: mikeio.DataArray,
    u_da: mikeio.DataArray,
    v_da: mikeio.DataArray,
    theta_rad: np.ndarray,
    directions: np.ndarray,
    window_size: int,
    overlap: float,
    window: str,
    method: str,
    g: float,
) -> SpectraArray:
    """Compute MEM spectrum at every FM-mesh spatial location independently."""
    _, loc_axis, n_locs = _loc_axis_info(eta_da)
    dt = _dt_seconds(eta_da)

    eta_vals = np.asarray(eta_da.values, dtype=float)
    u_vals = np.asarray(u_da.values, dtype=float)
    v_vals = np.asarray(v_da.values, dtype=float)

    spectra: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    for i in tqdm(range(n_locs), desc="Computing MEM spectrum at spatial locations"):
        eta_i = np.take(eta_vals, i, axis=loc_axis)
        u_i = np.take(u_vals, i, axis=loc_axis)
        v_i = np.take(v_vals, i, axis=loc_axis)

        f, S_ee, a1, b1, a2, b2 = directional_fourier_coeffs(
            eta_i, u_i, v_i, dt, window_size, overlap, window, g=g
        )
        if freqs is None:
            freqs = f

        D = _apply_mem(theta_rad, a1, b1, a2, b2, method)
        spectra.append(S_ee[:, None] * D)

    S_stack = np.stack(spectra, axis=0)

    assert freqs is not None
    io_geom = _to_spectral_geom(eta_da.geometry, freqs, directions)
    x, y = _extract_coords(io_geom)
    if x is None or y is None:
        x = np.zeros(n_locs)
        y = np.zeros(n_locs)
    return _build_spatial_spectrum(
        S_stack, freqs, directions, _setup_time(eta_da), io_geom, x, y
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def spectrum(
    da: mikeio.DataArray | mikeio.Dataset,
    truncate_f0: bool = True,
) -> SpectraArray | SpectraSet:
    """One-sided power spectral density via a single FFT.

    Applies a rectangular (no-taper) window over the entire record.  For
    reduced variance at the cost of frequency resolution use
    :func:`spectrum_welch`.

    Parameters
    ----------
    da : mikeio.DataArray or mikeio.Dataset
        Time series.  ``Grid1D`` and ``Grid2D`` DataArrays are supported and
        compute the spectrum at every grid location, returning a multi-location
        :class:`SpectraArray`.  For FM mesh DataArrays with an ``element`` or
        ``node`` dimension use ``.isel()`` to select a single location first.
        A :class:`~mikeio.Dataset` processes each item independently and
        returns a :class:`SpectraSet`.
    truncate_f0 : bool
        Drop the zero-frequency (DC) component (default ``True``).

    Returns
    -------
    SpectraArray
        1-D frequency spectrum.
    SpectraSet
        One :class:`SpectraArray` per item when *da* is a
        :class:`~mikeio.Dataset`.

    Examples
    --------
    >>> import numpy as np, pandas as pd, mikeio
    >>> from pywswat import spectrum
    >>> t = pd.date_range("2000-01-01", periods=512, freq="s")
    >>> sig = np.sin(2 * np.pi * 0.1 * np.arange(512))
    >>> da = mikeio.DataArray(sig, time=t)
    >>> spec = spectrum(da)
    >>> spec.has_freq
    True
    >>> spec.has_dir
    False
    """
    if isinstance(da, mikeio.Dataset):
        from pywswat.collection import SpectraSet  # lazy to avoid circular import

        das = [_single_spectrum(da[name], truncate_f0) for name in da.names]
        return SpectraSet(das)
    return _single_spectrum(da, truncate_f0)


def spectrum_welch(
    da: mikeio.DataArray | mikeio.Dataset,
    window_size: int = 512,
    overlap: float = 0.5,
    window: str = "hann",
    truncate_f0: bool = True,
) -> SpectraArray | SpectraSet:
    """One-sided PSD via Welch's averaged periodogram method.

    Parameters
    ----------
    da : mikeio.DataArray or mikeio.Dataset
        Time series.  ``Grid1D`` and ``Grid2D`` DataArrays are supported and
        compute the spectrum at every grid location, returning a multi-location
        :class:`SpectraArray`.  For FM mesh DataArrays with an ``element`` or
        ``node`` dimension use ``.isel()`` to select a single location first.
        A :class:`~mikeio.Dataset` returns a :class:`SpectraSet`.
    window_size : int
        Samples per segment (default 512).
    overlap : float
        Fractional overlap between consecutive segments, 0 ≤ overlap < 1
        (default 0.5).
    window : str
        Window function name (default ``"hann"``).
    truncate_f0 : bool
        Drop the zero-frequency (DC) component (default ``True``).

    Returns
    -------
    SpectraArray or SpectraSet

    Examples
    --------
    >>> import numpy as np, pandas as pd, mikeio
    >>> from pywswat import spectrum_welch
    >>> t = pd.date_range("2000-01-01", periods=2048, freq="s")
    >>> rng = np.random.default_rng(0)
    >>> da = mikeio.DataArray(rng.standard_normal(2048), time=t)
    >>> spec = spectrum_welch(da, window_size=256)
    >>> spec.has_freq
    True
    """
    if isinstance(da, mikeio.Dataset):
        from pywswat.collection import SpectraSet  # lazy to avoid circular import

        das = [
            _single_welch(da[name], window_size, overlap, window, truncate_f0)
            for name in da.names
        ]
        return SpectraSet(das)
    return _single_welch(da, window_size, overlap, window, truncate_f0)


def spectrum_mem(
    ds: mikeio.Dataset,
    elevation: str,
    u: str,
    v: str,
    directions: np.ndarray | None = None,
    window_size: int = 512,
    overlap: float = 0.5,
    window: str = "hann",
    method: str = "lk86",
    g: float = 9.81,
) -> SpectraArray:
    """Estimate the directional wave spectrum using cross-spectral analysis and MEM.

    Parameters
    ----------
    ds : mikeio.Dataset
        Dataset containing the time series items.
    elevation : str
        Item name of the surface elevation [m].
    u : str
        Item name of the depth-averaged east / x velocity [m/s].
    v : str
        Item name of the depth-averaged north / y velocity [m/s].
    directions : array-like of float, optional
        Wave direction bins in **degrees**.  Defaults to 0–350° in 10° steps.
    window_size : int
        Samples per Welch segment (default 512).
    overlap : float
        Fractional segment overlap (default 0.5).
    window : str
        Window function (default ``"hann"``).
    method : str
        MEM algorithm: ``"lk86"`` or ``"newton"``.
    g : float
        Gravitational acceleration [m/s²] (default 9.81).

    Returns
    -------
    SpectraArray

    Examples
    --------
    >>> import numpy as np, pandas as pd, mikeio
    >>> from pywswat import spectrum_mem
    >>> N, dt, f0, theta0 = 4096, 0.5, 0.1, 45.0
    >>> t = pd.date_range("2000-01-01", periods=N, freq=f"{dt}s")
    >>> rng = np.random.default_rng(0)
    >>> eta = np.sin(2 * np.pi * f0 * np.arange(N) * dt) + rng.standard_normal(N) * 0.05
    >>> g_val, omega0 = 9.81, 2 * np.pi * f0
    >>> u_arr = (g_val / omega0) * np.cos(np.deg2rad(theta0)) * eta + rng.standard_normal(N) * 0.01
    >>> v_arr = (g_val / omega0) * np.sin(np.deg2rad(theta0)) * eta + rng.standard_normal(N) * 0.01
    >>> ds = mikeio.Dataset([
    ...     mikeio.DataArray(eta,   time=t, name="eta"),
    ...     mikeio.DataArray(u_arr, time=t, name="u"),
    ...     mikeio.DataArray(v_arr, time=t, name="v"),
    ... ])
    >>> spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    >>> spec.has_freq and spec.has_dir
    True
    >>> spec.has_location
    False
    """
    if directions is None:
        directions = np.arange(0, 360, 10, dtype=float)
    directions = np.asarray(directions, dtype=float)
    theta_rad = np.deg2rad(directions)

    eta_da = ds[elevation]
    u_da = ds[u]
    v_da = ds[v]

    kwargs = dict(
        theta_rad=theta_rad,
        directions=directions,
        window_size=window_size,
        overlap=overlap,
        window=window,
        method=method,
        g=g,
    )

    if _is_grid_geom(eta_da):
        return _mem_spectrum_grid(eta_da, u_da, v_da, **kwargs)
    if _is_spatial(eta_da):
        return _mem_spectrum_spatial(eta_da, u_da, v_da, **kwargs)
    return _mem_spectrum_point(eta_da, u_da, v_da, **kwargs)
