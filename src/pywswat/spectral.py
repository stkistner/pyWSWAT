"""Spectral estimation from time series.

Public functions
----------------
:func:`spectrum`
    One-sided PSD via a single FFT of the entire record.
:func:`spectrum_welch`
    One-sided PSD via Welch's averaged periodogram.
:func:`spectrum_mem`
    2-D directional spectrum via cross-spectral analysis and MEM.
    Supports point, area (element), and line (node) input geometries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mikeio
from mikeio.spatial._FM_geometry_spectral import (
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
)
from tqdm import tqdm

from pywswat._fft import (
    _dt_seconds,
    _fft_spectrum,
    _welch_spectrum,
    directional_fourier_coeffs,
)
from pywswat._mem import mem_lk86, mem_newton
from pywswat.wave_spectrum import SpectraArray
from pywswat.collection import SpectraSet

_SPEC_EUM_TYPE = mikeio.EUMType.Wave_energy_density
_UNIT_1D = mikeio.EUMUnit.meter_pow_2_sec  # m²·s = m²/Hz
_UNIT_2D = mikeio.EUMUnit.meter_pow_2_sec_per_deg  # stored as m²·s/rad by convention

# Dimension names that indicate a spatial (location) axis
_LOC_DIMS = frozenset({"element", "node"})


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────


def _is_spatial(da: mikeio.DataArray) -> bool:
    """Return True when *da* has a recognised spatial location dimension."""
    return bool(_LOC_DIMS & set(da.dims))


def _loc_axis_info(da: mikeio.DataArray) -> tuple[str, int, int]:
    """Return ``(dim_name, axis_index, n_locs)`` for the location axis."""
    dims = list(da.dims)
    for d in dims:
        if d in _LOC_DIMS:
            ax = dims.index(d)
            return d, ax, da.values.shape[ax]
    raise ValueError(
        f"No spatial dimension found in dims {da.dims!r}. Expected one of {_LOC_DIMS}."
    )


def _to_spectral_geom(
    input_geom: object,
    freqs: np.ndarray,
    directions: np.ndarray,
) -> GeometryFMAreaSpectrum | GeometryFMLineSpectrum:
    """Map an FM spatial geometry to its spectral counterpart.

    ============================================= ===========================
    Input geometry type                           Output spectral geometry
    ============================================= ===========================
    ``GeometryFM2D`` (Dfsu2D)                    ``GeometryFMAreaSpectrum``
    ``GeometryFM2D`` (Dfsu1D)                    ``GeometryFMLineSpectrum``
    ============================================= ===========================

    Raises
    ------
    TypeError
        If *input_geom* is not a supported ``GeometryFM2D`` sub-type.
    """
    from mikecore.DfsuFile import DfsuFileType
    from mikeio.spatial._FM_geometry import GeometryFM2D
    from mikeio.spatial._FM_geometry_spectral import (
        GeometryFMAreaSpectrum,
        GeometryFMLineSpectrum,
    )

    if not isinstance(input_geom, GeometryFM2D):
        raise TypeError(
            f"Spatial spectrum_mem requires a GeometryFM2D input geometry, "
            f"got {type(input_geom).__name__!r}. "
            "For other geometry types (e.g. regular grids), select individual "
            "locations with .isel() before calling spectrum_mem."
        )

    mesh_kwargs = dict(
        node_coordinates=input_geom.node_coordinates,
        element_table=input_geom.element_table,
        codes=input_geom.codes,
        projection=input_geom.projection_string,
        frequencies=freqs,
        directions=directions,
    )
    dfsu_type = input_geom._type

    if dfsu_type == DfsuFileType.Dfsu2D:
        return GeometryFMAreaSpectrum(
            **mesh_kwargs, dfsu_type=DfsuFileType.DfsuSpectral2D
        )
    if dfsu_type == DfsuFileType.Dfsu1D:
        return GeometryFMLineSpectrum(
            **mesh_kwargs, dfsu_type=DfsuFileType.DfsuSpectral1D
        )

    raise TypeError(
        f"Unsupported dfsu_type {dfsu_type!r} for spatial spectral analysis. "
        "Supported: Dfsu2D → GeometryFMAreaSpectrum, Dfsu1D → GeometryFMLineSpectrum."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Spectrum construction helpers
# ──────────────────────────────────────────────────────────────────────────────


def _center_time(da: mikeio.DataArray) -> pd.DatetimeIndex:
    """Midpoint timestamp of a DataArray's time axis."""
    t = da.time
    return pd.DatetimeIndex([t[0] + (t[-1] - t[0]) / 2])


def _build_point_spectrum_da(
    psd: np.ndarray,
    freqs: np.ndarray,
    time: pd.DatetimeIndex,
    name: str,
    directions: np.ndarray | None = None,
) -> mikeio.DataArray:
    """Build a *point* spectral DataArray (GeometryFMPointSpectrum).

    Parameters
    ----------
    psd :
        * 1-D ``(n_freqs,)`` — energy density in m²/Hz
        * 2-D ``(n_dirs, n_freqs)`` — energy density in m²/(Hz·rad), mikeio order
    """
    geom = mikeio.spatial.GeometryFMPointSpectrum(
        frequencies=freqs, directions=directions
    )
    if directions is None:
        data = psd[np.newaxis, :]  # (1, nf)
        unit = _UNIT_1D
    else:
        data = psd[np.newaxis, :, :]  # (1, nd, nf)
        unit = _UNIT_2D
    return mikeio.DataArray(
        data=data,
        time=time,
        geometry=geom,
        type=_SPEC_EUM_TYPE,
        unit=unit,
        name=name,
    )


def _build_spatial_spectrum_da(
    spectra: np.ndarray,
    freqs: np.ndarray,
    directions: np.ndarray,
    time: pd.DatetimeIndex,
    geom: GeometryFMAreaSpectrum | GeometryFMLineSpectrum,
) -> mikeio.DataArray:
    """Build a spatial spectral DataArray (area or line geometry).

    Parameters
    ----------
    spectra : (n_locs, n_dirs, n_freqs) float
        Pre-computed 2-D spectra at each location, in m²/(Hz·rad).
    geom : GeometryFMAreaSpectrum or GeometryFMLineSpectrum
    """
    # mikeio order: (time, loc, direction, frequency)
    data = spectra[np.newaxis, ...]  # (1, n_locs, n_dirs, n_freqs)
    return mikeio.DataArray(
        data=data,
        time=time,
        geometry=geom,
        type=_SPEC_EUM_TYPE,
        unit=_UNIT_2D,
        name="Energy density",
    )


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
# Point-only spectrum helpers (used by spectrum / spectrum_welch)
# ──────────────────────────────────────────────────────────────────────────────


def _single_spectrum(da: mikeio.DataArray, truncate_f0: bool) -> SpectraArray:
    if da.values.ndim != 1:
        raise ValueError(
            f"spectrum() requires a 1-D (time-only) DataArray, "
            f"got dims {da.dims!r}. Use .isel() to select a single location first."
        )
    dt = _dt_seconds(da)
    freqs, psd = _fft_spectrum(np.asarray(da.values, dtype=float), dt, truncate_f0)
    return SpectraArray(
        _build_point_spectrum_da(psd, freqs, _center_time(da), name=da.name)
    )


def _single_welch(
    da: mikeio.DataArray,
    window_size: int,
    overlap: float,
    window: str,
    truncate_f0: bool,
) -> SpectraArray:
    if da.values.ndim != 1:
        raise ValueError(
            f"spectrum_welch() requires a 1-D (time-only) DataArray, "
            f"got dims {da.dims!r}. Use .isel() to select a single location first."
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
    return SpectraArray(
        _build_point_spectrum_da(psd, freqs, _center_time(da), name=da.name)
    )


# ──────────────────────────────────────────────────────────────────────────────
# MEM spectrum helpers (point and spatial)
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
    D = _apply_mem(theta_rad, a1, b1, a2, b2, method)  # (n_freqs, n_dirs)
    S_2d_mikeio = (S_ee[:, None] * D).T  # (n_dirs, n_freqs)

    t = _center_time(eta_da)
    da = _build_point_spectrum_da(
        S_2d_mikeio, freqs, t, name="Energy density", directions=directions
    )
    return SpectraArray(da)


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
    """Compute MEM spectrum at every spatial location independently."""
    _, loc_axis, n_locs = _loc_axis_info(eta_da)
    dt = _dt_seconds(eta_da)

    eta_vals = np.asarray(eta_da.values, dtype=float)
    u_vals = np.asarray(u_da.values, dtype=float)
    v_vals = np.asarray(v_da.values, dtype=float)

    spectra: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    for i in tqdm(range(n_locs), desc="Computing MEM spectrum at spatial locations"):
        eta_i = np.take(eta_vals, i, axis=loc_axis)  # (n_time,)
        u_i = np.take(u_vals, i, axis=loc_axis)
        v_i = np.take(v_vals, i, axis=loc_axis)

        f, S_ee, a1, b1, a2, b2 = directional_fourier_coeffs(
            eta_i, u_i, v_i, dt, window_size, overlap, window, g=g
        )
        if freqs is None:
            freqs = f

        # Same meteo→sci conversion as in _mem_spectrum_point
        D = _apply_mem(theta_rad, -b1, -a1, -a2, b2, method)  # (n_freqs, n_dirs)
        spectra.append((S_ee[:, None] * D).T)  # (n_dirs, n_freqs)

    # Stack: (n_locs, n_dirs, n_freqs)
    S_stack = np.stack(spectra, axis=0)

    assert freqs is not None
    out_geom = _to_spectral_geom(eta_da.geometry, freqs, directions)
    t = _center_time(eta_da)
    return SpectraArray(
        _build_spatial_spectrum_da(S_stack, freqs, directions, t, out_geom)
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
        Time series.  Spatial dims are **not** supported — use ``.isel()``
        to extract a single location first.  A :class:`~mikeio.Dataset`
        processes each item independently and returns a :class:`SpectraSet`.
    truncate_f0 : bool
        Drop the zero-frequency (DC) component (default ``True``).

    Returns
    -------
    SpectraArray
        1-D frequency spectrum backed by a ``GeometryFMPointSpectrum``.
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
        das = [_single_spectrum(da[name], truncate_f0) for name in da.names]
        return SpectraSet(mikeio.Dataset([s.da for s in das]))
    return _single_spectrum(da, truncate_f0)


def spectrum_welch(
    da: mikeio.DataArray | mikeio.Dataset,
    window_size: int = 512,
    overlap: float = 0.5,
    window: str = "hann",
    truncate_f0: bool = True,
) -> SpectraArray | SpectraSet:
    """One-sided PSD via Welch's averaged periodogram method.

    Divides the time series into overlapping segments, applies a window
    function, FFT-transforms each segment, and averages the periodograms.
    Reduces variance compared to a single FFT at the cost of halved
    frequency resolution relative to the full record length.

    Parameters
    ----------
    da : mikeio.DataArray or mikeio.Dataset
        Time series.  Spatial dims are **not** supported — use ``.isel()``
        to select a single location first.  A :class:`~mikeio.Dataset`
        returns a :class:`SpectraSet`.
    window_size : int
        Samples per segment (default 512).
    overlap : float
        Fractional overlap between consecutive segments, 0 ≤ overlap < 1
        (default 0.5; a common choice is 0.5–0.67).
    window : str
        Window function name accepted by :func:`scipy.signal.welch`
        (default ``"hann"``).
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
        das = [
            _single_welch(da[name], window_size, overlap, window, truncate_f0)
            for name in da.names
        ]
        return SpectraSet(mikeio.Dataset([s.da for s in das]))
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

    Computes cross-spectral densities between surface elevation η and the
    depth-averaged horizontal velocity components u, v using Welch's method,
    extracts the first four normalised directional Fourier moments
    (a1, b1, a2, b2), and applies a Maximum Entropy Method estimator to
    recover the full directional distribution D(f, θ).

    The output spectrum is::

        S(f, θ) = S_{ηη}(f) · D(f, θ)   [m²·s/rad]

    **Multi-dimensional input** is supported when the Dataset items carry an
    FM mesh geometry.  The spectrum is computed independently at each spatial
    location and the result is returned as a spatially-resolved
    :class:`SpectraArray`:

    ========================= =================== ==============================
    Input geometry            Output geometry     Output dims
    ========================= =================== ==============================
    ``GeometryFMPointSpectrum`` / no spatial dim ``GeometryFMPointSpectrum``   ``(time:1, dir, freq)``
    ``GeometryFM2D`` (Dfsu2D) ``GeometryFMAreaSpectrum``   ``(time:1, element, dir, freq)``
    ``GeometryFM2D`` (Dfsu1D) ``GeometryFMLineSpectrum``   ``(time:1, element, dir, freq)``
    ========================= =================== ==============================

    Parameters
    ----------
    ds : mikeio.Dataset
        Dataset containing the time series items.  All three items must share
        the same geometry and time axis.
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
        MEM algorithm:

        * ``"lk86"`` — Lygre & Krogstad (1986) closed-form AR(2). Fast.
        * ``"newton"`` — Kim et al. (1995) MEM2 Newton-Raphson. More
          accurate for narrow or multi-modal distributions.

    g : float
        Gravitational acceleration [m/s²] (default 9.81).

    Returns
    -------
    SpectraArray
        Directional spectrum.  Geometry matches the input (see table above).

    References
    ----------
    Kuik, A. J., G. Ph. van Vledder, and L. H. Holthuijsen (1988).
    *J. Phys. Oceanogr.*, 18, 1020–1034.

    Lygre, A., and H. E. Krogstad (1986). *J. Phys. Oceanogr.*, 16, 2052–2060.

    Kim, T., L. E. Borgman, and B. Mehaute (1995).
    *J. Waterway, Port, Coastal, and Ocean Engineering*, 121, 275–285.

    Examples
    --------
    **Point input** (single time series):

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

    if _is_spatial(eta_da):
        return _mem_spectrum_spatial(eta_da, u_da, v_da, **kwargs)
    return _mem_spectrum_point(eta_da, u_da, v_da, **kwargs)
