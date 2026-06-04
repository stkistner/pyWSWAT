"""Tests for spectral estimation: spectrum, spectrum_welch, spectrum_mem."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import mikeio

from pywswat import SpectraArray, SpectraSet, spectrum, spectrum_welch, spectrum_mem
from pywswat._mem import mem_lk86, mem_newton

# ---------------------------------------------------------------------------
# Shared synthetic signal helpers
# ---------------------------------------------------------------------------

G = 9.81
DT = 0.5           # 2 Hz sampling
N = 4096
F0 = 0.1           # Hz — peak frequency
OMEGA0 = 2 * np.pi * F0
THETA0_DEG = 225.0  # degrees — wave direction
THETA0_RAD = np.deg2rad(THETA0_DEG)

_TIME = pd.date_range("2000-01-01", periods=N, freq=f"{DT}s")


def _sine_da(f0=F0, dt=DT, n=N, name="eta") -> mikeio.DataArray:
    """Clean sinusoidal time series at frequency f0."""
    t_arr = np.arange(n) * dt
    sig = np.sin(2 * np.pi * f0 * t_arr)
    return mikeio.DataArray(sig, time=pd.date_range("2000-01-01", periods=n, freq=f"{dt}s"), name=name)


def _noisy_mono_dataset(
    f0=F0, theta_deg=THETA0_DEG, dt=DT, n=N, seed=42, noise_scale=0.02
) -> mikeio.Dataset:
    """
    Monochromatic wave dataset: eta + correlated u, v + small noise.

    Uses the deep-water depth-averaged relationship:
        u = (g/ω) cos(θ) η
        v = (g/ω) sin(θ) η
    """
    rng = np.random.default_rng(seed)
    omega = 2 * np.pi * f0
    t_arr = np.arange(n) * dt
    time = pd.date_range("2000-01-01", periods=n, freq=f"{dt}s")
    eta = np.sin(2 * np.pi * f0 * t_arr) + rng.standard_normal(n) * noise_scale
    u = (G / omega) * np.cos(np.deg2rad(theta_deg)) * eta + rng.standard_normal(n) * noise_scale
    v = (G / omega) * np.sin(np.deg2rad(theta_deg)) * eta + rng.standard_normal(n) * noise_scale
    return mikeio.Dataset([
        mikeio.DataArray(eta, time=time, name="eta"),
        mikeio.DataArray(u, time=time, name="u"),
        mikeio.DataArray(v, time=time, name="v"),
    ])


# ---------------------------------------------------------------------------
# spectrum() — single FFT
# ---------------------------------------------------------------------------


def test_spectrum_returns_spectraarray():
    da = _sine_da()
    result = spectrum(da)
    assert isinstance(result, SpectraArray)


def test_spectrum_has_freq_no_dir():
    da = _sine_da()
    result = spectrum(da)
    assert result.has_freq
    assert not result.has_dir


def test_spectrum_peak_at_correct_frequency():
    da = _sine_da(f0=0.1)
    spec = spectrum(da)
    peak_idx = np.argmax(spec.da.values)
    peak_freq = spec.freq.flat[peak_idx]
    assert abs(peak_freq - 0.1) < 0.005


def test_spectrum_hm0_from_sine():
    """For a unit-amplitude sine, Hm0 ≈ 4*sqrt(m0) ≈ 4*sqrt(0.5) ≈ 2.83."""
    da = _sine_da()
    spec = spectrum(da)
    ds = spec.to_params()
    hm0 = float(ds["Hm0"].values[0])
    # Unit sinusoid: variance = 0.5 → Hm0 = 4*sqrt(0.5) ≈ 2.83
    assert hm0 == pytest.approx(4.0 * np.sqrt(0.5), rel=0.05)


def test_spectrum_truncate_f0():
    """With truncate_f0=True, zero-frequency bin is absent."""
    da = _sine_da()
    spec = spectrum(da, truncate_f0=True)
    assert spec.freq[0] > 0.0


def test_spectrum_dataset_returns_spectraset():
    ds = mikeio.Dataset([_sine_da(name="a"), _sine_da(f0=0.2, name="b")])
    result = spectrum(ds)
    assert isinstance(result, SpectraSet)
    assert result.n_items == 2


def test_spectrum_dataset_items_independent():
    """Spectrum of each item should peak at its own frequency."""
    ds = mikeio.Dataset([_sine_da(f0=0.1, name="a"), _sine_da(f0=0.2, name="b")])
    ss = spectrum(ds)
    spec_a = ss["a"]
    spec_b = ss["b"]
    peak_a = spec_a.freq[np.argmax(spec_a.da.values.ravel())]
    peak_b = spec_b.freq[np.argmax(spec_b.da.values.ravel())]
    assert abs(peak_a - 0.1) < 0.01
    assert abs(peak_b - 0.2) < 0.01


def test_spectrum_raises_for_spatial_da():
    """spectrum() must reject DataArrays with spatial dimensions."""
    da = mikeio.read("tests/testdata/spectra/area_spectra.dfsu")[0]
    pt_da = da.isel(element=0)  # still has (time, direction, frequency) dims
    with pytest.raises((ValueError, Exception)):
        spectrum(pt_da)


# ---------------------------------------------------------------------------
# spectrum_welch()
# ---------------------------------------------------------------------------


def test_spectrum_welch_returns_spectraarray():
    da = _sine_da()
    result = spectrum_welch(da, window_size=512)
    assert isinstance(result, SpectraArray)


def test_spectrum_welch_peak_correct():
    da = _sine_da(f0=0.1)
    spec = spectrum_welch(da, window_size=512)
    peak_idx = np.argmax(spec.da.values.ravel())
    peak_freq = spec.freq.ravel()[peak_idx]
    assert abs(peak_freq - 0.1) < 0.01


def test_spectrum_welch_hm0_from_sine():
    da = _sine_da()
    spec = spectrum_welch(da, window_size=512)
    ds = spec.to_params()
    hm0 = float(ds["Hm0"].values[0])
    assert hm0 == pytest.approx(4.0 * np.sqrt(0.5), rel=0.10)


def test_spectrum_welch_dataset_returns_spectraset():
    ds = mikeio.Dataset([_sine_da(name="a"), _sine_da(f0=0.2, name="b")])
    result = spectrum_welch(ds, window_size=256)
    assert isinstance(result, SpectraSet)
    assert result.n_items == 2


def test_spectrum_welch_fewer_freqs_than_fft():
    """Welch with window_size < N should yield fewer frequencies than FFT."""
    da = _sine_da()
    spec_fft = spectrum(da)
    spec_welch = spectrum_welch(da, window_size=512)
    assert spec_welch.nf < spec_fft.nf


def test_spectrum_welch_geometry_type():
    from mikeio.spatial._FM_geometry_spectral import GeometryFMPointSpectrum
    da = _sine_da()
    spec = spectrum_welch(da)
    assert isinstance(spec.geometry, GeometryFMPointSpectrum)


# ---------------------------------------------------------------------------
# mem_lk86 and mem_newton — unit tests
# ---------------------------------------------------------------------------


def _make_moments(theta_deg: float, r1: float = 0.8, r2: float = 0.5):
    """
    Directional moments for a distribution centred at theta_deg.

    r1 ∈ (0, 1) controls the width of the main lobe (r1→1 = very narrow).
    Using r1 < 1 avoids the degenerate unidirectional case that the LK86
    AR(2) cannot handle (denominator → 0 when |c1| = 1).
    """
    theta_rad = np.deg2rad(theta_deg)
    a1 = np.array([r1 * np.cos(theta_rad)])
    b1 = np.array([r1 * np.sin(theta_rad)])
    a2 = np.array([r2 * np.cos(2 * theta_rad)])
    b2 = np.array([r2 * np.sin(2 * theta_rad)])
    return a1, b1, a2, b2


def test_mem_lk86_shape():
    theta = np.deg2rad(np.arange(0, 360, 10, dtype=float))
    a1, b1, a2, b2 = _make_moments(90.0)
    D = mem_lk86(theta, a1, b1, a2, b2)
    assert D.shape == (1, len(theta))


def test_mem_lk86_normalized():
    theta = np.deg2rad(np.arange(0, 360, 10, dtype=float))
    a1, b1, a2, b2 = _make_moments(90.0)
    D = mem_lk86(theta, a1, b1, a2, b2)
    dtheta = np.diff(theta).mean()
    integ = float(D[0] @ np.full(len(theta), dtheta))
    assert integ == pytest.approx(1.0, abs=0.01)


def test_mem_lk86_peak_direction():
    """Peak of D should be close to the target direction."""
    theta_deg = np.arange(0, 360, 5, dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    a1, b1, a2, b2 = _make_moments(135.0)
    D = mem_lk86(theta_rad, a1, b1, a2, b2)
    peak_dir = float(theta_deg[np.argmax(D[0])])
    assert abs(peak_dir - 135.0) <= 10.0


def test_mem_newton_shape():
    theta = np.deg2rad(np.arange(0, 360, 10, dtype=float))
    a1, b1, a2, b2 = _make_moments(90.0)
    D = mem_newton(theta, a1, b1, a2, b2)
    assert D.shape == (1, len(theta))


def test_mem_newton_normalized():
    theta = np.deg2rad(np.arange(0, 360, 10, dtype=float))
    a1, b1, a2, b2 = _make_moments(90.0)
    D = mem_newton(theta, a1, b1, a2, b2)
    dtheta = np.diff(theta).mean()
    integ = float(D[0] @ np.full(len(theta), dtheta))
    assert integ == pytest.approx(1.0, abs=0.01)


def test_mem_newton_peak_direction():
    theta_deg = np.arange(0, 360, 5, dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    a1, b1, a2, b2 = _make_moments(135.0)
    D = mem_newton(theta_rad, a1, b1, a2, b2)
    peak_dir = float(theta_deg[np.argmax(D[0])])
    assert abs(peak_dir - 135.0) <= 10.0


def test_mem_lk86_multi_freq():
    """LK86 must handle multiple frequency bins at once."""
    theta = np.deg2rad(np.arange(0, 360, 10, dtype=float))
    nf = 20
    a1 = np.full(nf, 0.5)
    b1 = np.full(nf, 0.5)
    a2 = np.full(nf, 0.1)
    b2 = np.full(nf, 0.0)
    D = mem_lk86(theta, a1, b1, a2, b2)
    assert D.shape == (nf, len(theta))
    # All rows should integrate to ~1
    dtheta = np.diff(theta).mean()
    integ = (D * dtheta).sum(axis=-1)
    np.testing.assert_allclose(integ, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# spectrum_mem()
# ---------------------------------------------------------------------------


def test_spectrum_mem_returns_spectraarray():
    ds = _noisy_mono_dataset()
    result = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    assert isinstance(result, SpectraArray)


def test_spectrum_mem_has_freq_and_dir():
    ds = _noisy_mono_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    assert spec.has_freq
    assert spec.has_dir


def test_spectrum_mem_dims_mikeio_order():
    """Output DataArray dims must be (time, direction, frequency)."""
    ds = _noisy_mono_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    assert spec.da.dims == ("time", "direction", "frequency")


def test_spectrum_mem_default_directions():
    ds = _noisy_mono_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    assert spec.nd == 36   # 0–350 in 10° steps


def test_spectrum_mem_custom_directions():
    ds = _noisy_mono_dataset()
    dirs = np.arange(0, 360, 15, dtype=float)
    spec = spectrum_mem(ds, "eta", "u", "v", directions=dirs, window_size=512)
    assert spec.nd == len(dirs)
    np.testing.assert_array_equal(spec.direction, dirs)


def test_spectrum_mem_hm0_matches_eta_spectrum():
    """
    m0 from the 2-D MEM spectrum must be close to m0 from the 1-D η spectrum.
    """
    ds = _noisy_mono_dataset(noise_scale=0.0)  # no noise for cleaner test
    spec_2d = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    spec_1d = spectrum_welch(ds["eta"], window_size=512)

    hm0_2d = float(spec_2d.to_params()["Hm0"].values[0])
    hm0_1d = float(spec_1d.to_params()["Hm0"].values[0])
    assert hm0_2d == pytest.approx(hm0_1d, rel=0.10)


def test_spectrum_mem_mwd_near_input_direction():
    """
    MWD from the 2-D spectrum should be close to the meteorological
    "coming-from" direction: (270 - THETA0_DEG) % 360 = 45° for THETA0=225°.
    directional_fourier_coeffs defaults to meteo_dir=True.
    """
    ds = _noisy_mono_dataset(theta_deg=THETA0_DEG, noise_scale=0.02)
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    mwd = float(spec.to_params()["MWD"].values[0])
    expected_meteo = (270.0 - THETA0_DEG) % 360.0   # 45.0° for THETA0=225°
    diff = abs((mwd - expected_meteo + 180) % 360 - 180)  # circular distance
    assert diff < 30.0   # within 30°


def test_spectrum_mem_lk86_vs_newton_similar():
    """LK86 and Newton-Raphson should give similar Hm0."""
    ds = _noisy_mono_dataset()
    s_lk = spectrum_mem(ds, "eta", "u", "v", window_size=512, method="lk86")
    s_nw = spectrum_mem(ds, "eta", "u", "v", window_size=512, method="newton")
    hm0_lk = float(s_lk.to_params()["Hm0"].values[0])
    hm0_nw = float(s_nw.to_params()["Hm0"].values[0])
    assert hm0_lk == pytest.approx(hm0_nw, rel=0.05)


def test_spectrum_mem_invalid_method_raises():
    ds = _noisy_mono_dataset()
    with pytest.raises(ValueError, match="method"):
        spectrum_mem(ds, "eta", "u", "v", method="bad_method")


def test_spectrum_mem_to_params_returns_dataset():
    ds = _noisy_mono_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    params = spec.to_params()
    assert isinstance(params, mikeio.Dataset)
    for key in ["Hm0", "Tp", "MWD", "DSD"]:
        assert key in params.names


def test_spectrum_mem_geometry_type():
    from mikeio.spatial._FM_geometry_spectral import GeometryFMPointSpectrum
    ds = _noisy_mono_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    assert isinstance(spec.geometry, GeometryFMPointSpectrum)


# ---------------------------------------------------------------------------
# Integration: spectrum → SpectraArray operations
# ---------------------------------------------------------------------------


def test_spectrum_then_sel_freq():
    da = _sine_da()
    spec = spectrum(da)
    spec2 = spec.sel_freq(fmin=0.05, fmax=0.2)
    assert isinstance(spec2, SpectraArray)
    assert spec2.nf < spec.nf


def test_spectrum_welch_then_to_params():
    da = _sine_da()
    spec = spectrum_welch(da, window_size=512)
    params = spec.to_params()
    assert isinstance(params, mikeio.Dataset)
    assert "Hm0" in params.names


def test_spectrum_mem_then_integrate_dir():
    ds = _noisy_mono_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=512)
    spec_1d = spec.integrate_dir()
    assert isinstance(spec_1d, SpectraArray)
    assert not spec_1d.has_dir
    assert spec_1d.has_freq


# ---------------------------------------------------------------------------
# spectrum_mem — spatial (multi-location) inputs
# ---------------------------------------------------------------------------


def _spatial_dataset(n_elem: int = 5, seed: int = 7) -> tuple[mikeio.Dataset, object]:
    """
    Build a synthetic (time, element) Dataset with GeometryFM2D geometry,
    reusing the real mesh from the area spectral test file.

    Returns (ds, geom_2d).
    """
    from pywswat import read as pyw_read

    spec_area = pyw_read("tests/testdata/spectra/area_spectra.dfsu", item=0)
    geom_2d = spec_area.to_params()[0].geometry   # GeometryFM2D(Dfsu2D)

    N_TIME = 2048
    DT = 0.5
    time = pd.date_range("2000-01-01", periods=N_TIME, freq=f"{DT}s")
    rng = np.random.default_rng(seed)

    n_locs = geom_2d.n_elements
    eta_data = rng.standard_normal((N_TIME, n_locs))
    u_data = rng.standard_normal((N_TIME, n_locs)) * 0.1
    v_data = rng.standard_normal((N_TIME, n_locs)) * 0.1

    ds = mikeio.Dataset([
        mikeio.DataArray(eta_data, time=time, geometry=geom_2d, name="eta"),
        mikeio.DataArray(u_data,   time=time, geometry=geom_2d, name="u"),
        mikeio.DataArray(v_data,   time=time, geometry=geom_2d, name="v"),
    ])
    return ds, geom_2d


def test_spectrum_mem_spatial_returns_spectraarray():
    ds, _ = _spatial_dataset()
    result = spectrum_mem(ds, "eta", "u", "v", window_size=256)
    assert isinstance(result, SpectraArray)


def test_spectrum_mem_spatial_has_location():
    ds, geom_2d = _spatial_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=256)
    assert spec.has_location
    assert spec.nloc == geom_2d.n_elements


def test_spectrum_mem_spatial_geometry_is_area_spectrum():
    from mikeio.spatial._FM_geometry_spectral import GeometryFMAreaSpectrum
    ds, _ = _spatial_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=256)
    assert isinstance(spec.geometry, GeometryFMAreaSpectrum)


def test_spectrum_mem_spatial_dims_mikeio_order():
    """dims must be (time, element, direction, frequency)."""
    ds, _ = _spatial_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=256)
    assert spec.da.dims == ("time", "element", "direction", "frequency")


def test_spectrum_mem_spatial_has_freq_and_dir():
    ds, _ = _spatial_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=256)
    assert spec.has_freq and spec.has_dir


def test_spectrum_mem_spatial_to_params_geometry():
    """to_params on a spatial spectrum should return GeometryFM2D (Dfsu2D)."""
    from mikeio.spatial._FM_geometry import GeometryFM2D
    ds, _ = _spatial_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=256)
    ds_params = spec.to_params()
    assert isinstance(ds_params["Hm0"].geometry, GeometryFM2D)
    assert ds_params["Hm0"].values.shape[1] == spec.nloc


def test_spectrum_mem_spatial_to_params_shape():
    ds, geom_2d = _spatial_dataset()
    spec = spectrum_mem(ds, "eta", "u", "v", window_size=256)
    ds_params = spec.to_params()
    n_elem = geom_2d.n_elements
    assert ds_params["Hm0"].values.shape == (1, n_elem)
    assert ds_params["MWD"].values.shape == (1, n_elem)


def test_spectrum_mem_spatial_isel_matches_point():
    """
    spectrum_mem on a spatial dataset, then .isel(), should give the same
    result as spectrum_mem computed directly on the point time series.
    """
    ds, geom_2d = _spatial_dataset()
    spec_area = spectrum_mem(ds, "eta", "u", "v", window_size=256)

    # Extract element 3
    loc = 3
    spec_pt_from_area = spec_area.isel(location=loc)

    # Compute directly on the point time series
    eta_loc = ds["eta"].isel(element=loc)
    u_loc   = ds["u"].isel(element=loc)
    v_loc   = ds["v"].isel(element=loc)
    ds_pt = mikeio.Dataset([eta_loc, u_loc, v_loc])
    spec_pt_direct = spectrum_mem(ds_pt, "eta", "u", "v", window_size=256)

    np.testing.assert_allclose(
        spec_pt_from_area.to_params()["Hm0"].values,
        spec_pt_direct.to_params()["Hm0"].values,
        rtol=1e-10,
    )


def test_spectrum_mem_spatial_unsupported_geometry_raises():
    """Non-FM geometry should raise TypeError."""
    from pywswat.spectral import _to_spectral_geom
    from mikeio.spatial import GeometryPoint2D
    geom_pt = GeometryPoint2D(x=0.0, y=0.0)
    with pytest.raises(TypeError, match="GeometryFM2D"):
        _to_spectral_geom(geom_pt, np.linspace(0.05, 0.5, 10), np.arange(0, 360, 10.0))
