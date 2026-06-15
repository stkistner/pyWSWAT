"""Tests for SpectraArray and SpectraSet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import mikeio

from pywswat import SpectraArray, SpectraSet, read

# ---------------------------------------------------------------------------
# Helpers — build test DataArrays without reading files
# ---------------------------------------------------------------------------

FREQ = np.linspace(0.05, 0.5, 32)
DIRS = np.arange(0, 360, 10, dtype=float)
_TIME1 = pd.DatetimeIndex(["2000-01-01"])
_TIME5 = pd.date_range("2000-01-01", periods=5, freq="h")


def _make_geom_pt(freq=None, dirs=None):
    freq = FREQ if freq is None else freq
    return mikeio.spatial.GeometryFMPointSpectrum(frequencies=freq, directions=dirs)


def _make_da_pt_2d(nt=1, data=None):
    """Point spectrum, freq+dir, mikeio order (nt, nd, nf)."""
    nf, nd = len(FREQ), len(DIRS)
    geom = _make_geom_pt(dirs=DIRS)
    if data is None:
        data = np.ones((nt, nd, nf))
    time = _TIME1 if nt == 1 else pd.date_range("2000-01-01", periods=nt, freq="h")
    return mikeio.DataArray(data=data, time=time, geometry=geom)


def _make_da_pt_1d(nt=1, data=None):
    """Point spectrum, freq only, mikeio order (nt, nf)."""
    nf = len(FREQ)
    geom = _make_geom_pt()
    if data is None:
        data = np.ones((nt, nf))
    time = _TIME1 if nt == 1 else pd.date_range("2000-01-01", periods=nt, freq="h")
    return mikeio.DataArray(data=data, time=time, geometry=geom)


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PT_DFSU = "tests/testdata/spectra/pt_spectra.dfsu"
AREA_DFSU = "tests/testdata/spectra/area_spectra.dfsu"
LINE_DFSU = "tests/testdata/spectra/line_spectra.dfsu"

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_construct_from_mikeio_da():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    # .da reconstructs a mikeio DataArray (no longer the identical object)
    assert isinstance(spec.da, mikeio.DataArray)
    assert spec.has_freq
    assert spec.has_dir
    assert spec.has_time


def test_construct_freq_only():
    da = _make_da_pt_1d(nt=1)
    spec = SpectraArray(da)
    assert spec.has_freq
    assert not spec.has_dir
    assert not spec.has_time


def test_construct_2d_single_step():
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    assert spec.has_freq and spec.has_dir
    assert not spec.has_time
    # shape follows mikeio order: (time, direction, frequency)
    assert spec.shape == (1, len(DIRS), len(FREQ))


def test_construct_2d_multi_step():
    da = _make_da_pt_2d(nt=5)
    spec = SpectraArray(da)
    assert spec.has_time
    # shape follows mikeio order: (time, direction, frequency)
    assert spec.shape == (5, len(DIRS), len(FREQ))


def test_invalid_geometry_raises():
    """DataArray with non-spectral geometry must fail."""
    da = mikeio.DataArray(
        data=np.ones((3,)),
        time=_TIME1,
    )
    with pytest.raises(ValueError, match="spectral"):
        SpectraArray(da)


def test_spectra_alias():
    from pywswat import Spectra

    assert Spectra is SpectraArray


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_freq_property_is_copy():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    f = spec.freq
    assert f is not None
    f[:] = 0
    assert spec.freq[0] != 0


def test_nf_nd_nt():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    assert spec.nf == da.geometry.n_frequencies
    assert spec.nd == da.geometry.n_directions
    assert spec.nt == len(da.time)


def test_name_property():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    assert spec.name == da.name


def test_geometry_property():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    assert isinstance(spec.geometry, type(da.geometry))


# ---------------------------------------------------------------------------
# _raw_data ordering
# ---------------------------------------------------------------------------


def test_raw_data_shape_2d_single_step():
    """Without time: shape is (nf, nd)."""
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    assert spec._raw_data.shape == (len(FREQ), len(DIRS))


def test_raw_data_shape_2d_multi_step():
    """With time: shape is (nt, nf, nd)."""
    da = _make_da_pt_2d(nt=4)
    spec = SpectraArray(da)
    assert spec._raw_data.shape == (4, len(FREQ), len(DIRS))


def test_raw_data_axes_swapped():
    """_raw_data must swap mikeio's (..., dir, freq) to (..., freq, dir)."""
    nf, nd = 4, 8
    freq = np.linspace(0.1, 0.4, nf)
    dirs = np.arange(0, 360, 45, dtype=float)
    geom = mikeio.spatial.GeometryFMPointSpectrum(frequencies=freq, directions=dirs)
    # mikeio order: (time=1, direction=nd, frequency=nf)
    raw = np.arange(nd * nf, dtype=float).reshape(1, nd, nf)
    da = mikeio.DataArray(data=raw, time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    # After swap and time-drop: shape (nf, nd), row i = freq i
    assert spec._raw_data.shape == (nf, nd)
    # row 0 of _raw_data corresponds to freq=0 across all dirs
    np.testing.assert_array_equal(spec._raw_data[0, :], raw[0, :, 0])


# ---------------------------------------------------------------------------
# _delta_freq
# ---------------------------------------------------------------------------


def test_delta_freq_uniform():
    freq = np.array([0.1, 0.2, 0.3, 0.4])
    geom = _make_geom_pt(freq=freq)
    da = mikeio.DataArray(data=np.ones((1, len(freq))), time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    df = spec._delta_freq()
    np.testing.assert_allclose(df, [0.1, 0.1, 0.1, 0.1])


def test_delta_freq_single_bin():
    freq = np.array([0.1])
    geom = _make_geom_pt(freq=freq)
    da = mikeio.DataArray(data=np.ones((1, 1)), time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    assert spec._delta_freq()[0] == pytest.approx(0.1)


def test_delta_freq_nonuniform():
    freq = np.array([0.05, 0.10, 0.30, 0.50])
    geom = _make_geom_pt(freq=freq)
    da = mikeio.DataArray(data=np.ones((1, 4)), time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    df = spec._delta_freq()
    assert df[1] == pytest.approx((0.30 - 0.05) / 2.0)


# ---------------------------------------------------------------------------
# Spectral moments
# ---------------------------------------------------------------------------


def test_m0_constant_spectrum():
    freq = np.linspace(0.1, 0.5, 200)
    C = 2.0
    geom = _make_geom_pt(freq=freq)
    da = mikeio.DataArray(data=np.full((1, 200), C), time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    m = spec._moments([0])
    expected = C * (freq[-1] - freq[0])
    assert abs(m[0] - expected) / expected < 0.01


def test_moments_with_time_returns_array():
    nt = 4
    da = _make_da_pt_1d(nt=nt)
    spec = SpectraArray(da)
    m = spec._moments([0])
    assert m[0].shape == (nt,)


def test_moments_2d_integrates_dir():
    """m0 from 2-D uniform spectrum == m0 from integrated 1-D."""
    da_2d = _make_da_pt_2d(nt=1)
    spec_2d = SpectraArray(da_2d)
    spec_1d = spec_2d.integrate_dir()
    m0_2d = spec_2d._moments([0])[0]
    m0_1d = spec_1d._moments([0])[0]
    assert m0_2d == pytest.approx(m0_1d, rel=1e-10)


# ---------------------------------------------------------------------------
# to_params — returns mikeio.Dataset
# ---------------------------------------------------------------------------


def test_to_params_returns_dict():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    params = spec.to_params()
    assert isinstance(params, dict)


def test_to_params_keys_freq_only():
    da = _make_da_pt_1d(nt=1)
    spec = SpectraArray(da)
    params = spec.to_params()
    for k in ["Hm0", "Tp", "T01", "T02", "Tm10"]:
        assert k in params
    for k in ["MWD", "PWD", "DSD"]:
        assert k not in params


def test_to_params_keys_2d():
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    params = spec.to_params()
    for k in ["Hm0", "Tp", "T01", "T02", "Tm10", "MWD", "PWD", "DSD"]:
        assert k in params


def test_to_params_single_step_scalar():
    """Single time step point spectrum → parameters are scalars (float)."""
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    params = spec.to_params()
    assert isinstance(params["Hm0"], (float, np.floating))


def test_to_params_multi_step_shape():
    """Multi time step → each parameter is an ndarray of shape (nt,)."""
    nt = 6
    da = _make_da_pt_2d(nt=nt)
    spec = SpectraArray(da)
    params = spec.to_params()
    for k in ["Hm0", "Tp", "T01", "T02", "Tm10", "MWD", "PWD", "DSD"]:
        assert np.asarray(params[k]).shape == (nt,)


def test_hm0_matches_formula():
    nf = 32
    freq = np.linspace(0.05, 0.5, nf)
    data_1d = np.exp(-0.5 * ((freq - 0.15) / 0.02) ** 2)
    geom = _make_geom_pt(freq=freq)
    da = mikeio.DataArray(data=data_1d[np.newaxis, :], time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    params = spec.to_params()
    m0 = float(spec._moments([0])[0])
    hm0_expected = 4.0 * np.sqrt(m0)
    assert float(params["Hm0"]) == pytest.approx(hm0_expected, rel=1e-6)


def test_tp_near_peak_frequency():
    fp = 0.2
    data_1d = np.exp(-0.5 * ((FREQ - fp) / 0.005) ** 2)
    geom = _make_geom_pt(freq=FREQ)
    da = mikeio.DataArray(data=data_1d[np.newaxis, :], time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    params = spec.to_params()
    nearest_bin = FREQ[np.argmin(np.abs(FREQ - fp))]
    assert float(params["Tp"]) == pytest.approx(1.0 / nearest_bin, rel=1e-6)


def test_period_ordering():
    data_1d = np.exp(-0.5 * ((FREQ - 0.15) / 0.03) ** 2)
    geom = _make_geom_pt(freq=FREQ)
    da = mikeio.DataArray(data=data_1d[np.newaxis, :], time=_TIME1, geometry=geom)
    spec = SpectraArray(da)
    params = spec.to_params()
    t02 = float(params["T02"])
    t01 = float(params["T01"])
    tm10 = float(params["Tm10"])
    assert t02 <= t01 + 1e-10
    assert t01 <= tm10 + 1e-10


# ---------------------------------------------------------------------------
# to_params — directional parameters
# ---------------------------------------------------------------------------


def test_mwd_single_direction_bin():
    nf, nd = len(FREQ), len(DIRS)
    # All energy at direction index 9 → 90 degrees
    raw = np.zeros((1, nd, nf))
    raw[0, 9, :] = 1.0
    da = _make_da_pt_2d(nt=1, data=raw)
    spec = SpectraArray(da)
    params = spec.to_params()
    assert float(params["MWD"]) == pytest.approx(90.0, abs=2.0)


def test_dsd_isotropic():
    da = _make_da_pt_2d(nt=1)  # uniform energy in all dirs
    spec = SpectraArray(da)
    params = spec.to_params()
    assert float(params["DSD"]) == pytest.approx(np.sqrt(2.0), rel=0.01)


def test_dsd_nearly_unidirectional():
    nf, nd = len(FREQ), len(DIRS)
    raw = np.zeros((1, nd, nf))
    raw[0, 5, :] = 1.0  # concentrated at one direction
    da = _make_da_pt_2d(nt=1, data=raw)
    spec = SpectraArray(da)
    params = spec.to_params()
    assert float(params["DSD"]) < 0.2


def test_pwd_returns_direction_at_peak():
    nf, nd = len(FREQ), len(DIRS)
    raw = np.zeros((1, nd, nf))
    # Peak energy at dir index 18 (180°), freq index 10
    raw[0, 18, 10] = 10.0
    da = _make_da_pt_2d(nt=1, data=raw)
    spec = SpectraArray(da)
    params = spec.to_params()
    assert float(params["PWD"]) == pytest.approx(180.0, abs=1.0)


# ---------------------------------------------------------------------------
# to_params — consistency across time steps
# ---------------------------------------------------------------------------


def test_params_time_consistent_with_single_step():
    rng = np.random.default_rng(42)
    nt = 4
    nf, nd = len(FREQ), len(DIRS)
    time = pd.date_range("2000-01-01", periods=nt, freq="h")
    raw = rng.exponential(scale=1.0, size=(nt, nd, nf))
    geom = _make_geom_pt(dirs=DIRS)
    da_t = mikeio.DataArray(data=raw, time=time, geometry=geom)
    spec_t = SpectraArray(da_t)
    ds_t = spec_t.to_params()

    for i in range(nt):
        da_i = mikeio.DataArray(data=raw[[i]], time=time[[i]], geometry=geom)
        spec_i = SpectraArray(da_i)
        ds_i = spec_i.to_params()
        assert np.asarray(ds_t["Hm0"])[i] == pytest.approx(
            float(ds_i["Hm0"]), rel=1e-10
        )
        assert np.asarray(ds_t["Tp"])[i] == pytest.approx(float(ds_i["Tp"]), rel=1e-10)


# ---------------------------------------------------------------------------
# integrate_dir
# ---------------------------------------------------------------------------


def test_integrate_dir_removes_dir_axis():
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    spec_f = spec.integrate_dir()
    assert not spec_f.has_dir
    assert spec_f.has_freq
    # mikeio shape: (time=1, frequency)
    assert spec_f.shape == (1, len(FREQ))


def test_integrate_dir_preserves_time():
    da = _make_da_pt_2d(nt=3)
    spec = SpectraArray(da)
    spec_f = spec.integrate_dir()
    assert spec_f.has_time
    assert spec_f.nt == 3
    # mikeio shape: (time, frequency)
    assert spec_f.shape == (3, len(FREQ))


def test_integrate_dir_returns_spectraarray():
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    result = spec.integrate_dir()
    assert isinstance(result, SpectraArray)


def test_integrate_dir_raises_without_dir():
    da = _make_da_pt_1d(nt=1)
    spec = SpectraArray(da)
    with pytest.raises(ValueError, match="direction"):
        spec.integrate_dir()


def test_integrate_dir_geometry_is_spectral():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    spec_f = spec.integrate_dir()
    from mikeio.spatial._FM_geometry_spectral import GeometryFMPointSpectrum

    assert isinstance(spec_f.geometry, GeometryFMPointSpectrum)


# ---------------------------------------------------------------------------
# sel_freq
# ---------------------------------------------------------------------------


def test_sel_freq_limits_range():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    spec2 = spec.sel_freq(fmin=0.1, fmax=0.3)
    assert spec2.freq is not None
    assert spec2.freq[0] >= 0.1 - 1e-10
    assert spec2.freq[-1] <= 0.3 + 1e-10


def test_sel_freq_preserves_dir_axis():
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    spec2 = spec.sel_freq(0.1, 0.3)
    assert spec2.has_dir
    assert spec2.nd == len(DIRS)


def test_sel_freq_preserves_time_axis():
    da = _make_da_pt_2d(nt=3)
    spec = SpectraArray(da)
    spec2 = spec.sel_freq(0.1, 0.4)
    assert spec2.has_time
    assert spec2.nt == 3


def test_sel_freq_raises_without_freq():
    dirs_only_geom = mikeio.spatial.GeometryFMPointSpectrum(
        frequencies=None, directions=DIRS
    )
    da = mikeio.DataArray(
        data=np.ones((1, len(DIRS))), time=_TIME1, geometry=dirs_only_geom
    )
    spec = SpectraArray(da)
    with pytest.raises(ValueError, match="frequency"):
        spec.sel_freq(0.1, 0.3)


def test_sel_freq_returns_spectraarray():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    result = spec.sel_freq(0.1, 0.3)
    assert isinstance(result, SpectraArray)


def test_sel_freq_geometry_preserved():
    """sel_freq must preserve the spectral geometry type and direction count."""
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    spec2 = spec.sel_freq(0.1, 0.3)
    from mikeio.spatial._FM_geometry_spectral import GeometryFMPointSpectrum

    assert isinstance(spec2.geometry, GeometryFMPointSpectrum)
    assert spec2.nd == spec.nd


# ---------------------------------------------------------------------------
# isel — multi-location spectra
# ---------------------------------------------------------------------------


def test_isel_area_removes_location():
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    pt = spec.isel(location=5)
    assert not pt.has_location
    assert pt.nloc is None


def test_isel_area_shape():
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    pt = spec.isel(location=5)
    # shape matches resulting mikeio DA: (time, direction, frequency)
    assert pt.shape == da.isel(element=5).values.shape


def test_isel_area_geometry_is_point_spectrum():
    from mikeio.spatial._FM_geometry_spectral import GeometryFMPointSpectrum

    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da).isel(location=5)
    assert isinstance(spec.geometry, GeometryFMPointSpectrum)


def test_isel_no_location_raises():
    da = _make_da_pt_2d(nt=1)
    spec = SpectraArray(da)
    with pytest.raises(ValueError, match="location"):
        spec.isel(location=0)


def test_isel_matches_mikeio_isel():
    """isel() must give same result as mikeio's native da.isel(element=i)."""
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    pt_pywswat = spec.isel(location=5)
    pt_mikeio = SpectraArray(da.isel(element=5))
    np.testing.assert_allclose(
        np.asarray(pt_pywswat.to_params()["Hm0"]),
        np.asarray(pt_mikeio.to_params()["Hm0"]),
    )


# ---------------------------------------------------------------------------
# Area spectrum properties
# ---------------------------------------------------------------------------


def test_area_has_location():
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    assert spec.has_location
    assert spec.nloc == 40


def test_area_shape():
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    # shape matches mikeio DA shape: (time, element, direction, frequency)
    assert spec.shape == da.values.shape


def test_area_to_params_shape():
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    params = spec.to_params()
    nt = len(da.time)
    assert np.asarray(params["Hm0"]).shape == (nt, 40)
    assert np.asarray(params["MWD"]).shape == (nt, 40)


def test_area_integrate_dir():
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    spec_f = spec.integrate_dir()
    assert not spec_f.has_dir
    assert spec_f.has_freq
    assert spec_f.has_location
    assert spec_f.nloc == 40


def test_area_sel_freq():
    da = mikeio.read(AREA_DFSU)[0]
    spec = SpectraArray(da)
    spec2 = spec.sel_freq(0.1, 0.3)
    assert spec2.has_location
    assert spec2.nloc == 40
    assert spec2.freq is not None
    assert spec2.freq[0] >= 0.1 - 1e-10
    assert spec2.freq[-1] <= 0.3 + 1e-10


# ---------------------------------------------------------------------------
# Line spectrum
# ---------------------------------------------------------------------------


def test_line_has_location():
    da = mikeio.read(LINE_DFSU)[0]
    spec = SpectraArray(da)
    assert spec.has_location
    assert spec.nloc == 10


def test_line_isel():
    da = mikeio.read(LINE_DFSU)[0]
    spec = SpectraArray(da)
    pt = spec.isel(location=3)
    assert not pt.has_location
    assert pt.shape == da.isel(node=3).values.shape


def test_line_integrate_dir():
    da = mikeio.read(LINE_DFSU)[0]
    spec = SpectraArray(da)
    spec_f = spec.integrate_dir()
    assert not spec_f.has_dir
    assert spec_f.has_location
    assert spec_f.nloc == 10


def test_line_sel_freq():
    da = mikeio.read(LINE_DFSU)[0]
    spec = SpectraArray(da)
    spec2 = spec.sel_freq(0.1, 0.3)
    assert spec2.has_location
    assert spec2.nloc == 10


# ---------------------------------------------------------------------------
# read() function
# ---------------------------------------------------------------------------


def test_read_returns_spectrasset_default():
    result = read(PT_DFSU)
    assert isinstance(result, SpectraSet)


def test_read_item_returns_spectraarray():
    result = read(PT_DFSU, item=0)
    assert isinstance(result, SpectraArray)


def test_read_item_string():
    da = mikeio.read(PT_DFSU)[0]
    result = read(PT_DFSU, item=da.name)
    assert isinstance(result, SpectraArray)


def test_read_area():
    ss = read(AREA_DFSU)
    assert isinstance(ss, SpectraSet)
    assert ss[0].has_location


# ---------------------------------------------------------------------------
# SpectraSet
# ---------------------------------------------------------------------------


def test_spectraset_from_dataset():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    assert ss.n_items == len(ds.names)


def test_spectraset_getitem_str():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    name = ss.names[0]
    assert isinstance(ss[name], SpectraArray)


def test_spectraset_getitem_int():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    assert isinstance(ss[0], SpectraArray)


def test_spectraset_getitem_slice():
    ds = mikeio.read(AREA_DFSU)
    ss = SpectraSet(ds)
    result = ss[0:1]
    assert isinstance(result, SpectraSet)


def test_spectraset_to_params_single_item():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    result = ss.to_params()
    assert isinstance(result, mikeio.Dataset)
    assert "Hm0" in result.names


def test_spectraset_to_params_multi_item():
    """Multi-item SpectraSet prefixes parameter names with item name."""
    da = mikeio.read(AREA_DFSU)[0]
    # Build a 2-item dataset by wrapping the same DA with different names
    da1 = mikeio.DataArray(da.values, time=da.time, geometry=da.geometry, name="comp1")
    da2 = mikeio.DataArray(da.values, time=da.time, geometry=da.geometry, name="comp2")
    ss = SpectraSet(mikeio.Dataset([da1, da2]))
    params = ss.to_params()
    assert isinstance(params, mikeio.Dataset)
    assert "comp1_Hm0" in params.names
    assert "comp2_Hm0" in params.names


def test_spectraset_integrate_dir():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    ss_f = ss.integrate_dir()
    assert isinstance(ss_f, SpectraSet)
    assert not ss_f.has_dir


def test_spectraset_sel_freq():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    ss2 = ss.sel_freq(0.1, 0.3)
    assert isinstance(ss2, SpectraSet)
    assert ss2.freq is not None
    assert ss2.freq[0] >= 0.1 - 1e-10


def test_spectraset_isel_area():
    ds = mikeio.read(AREA_DFSU)
    ss = SpectraSet(ds)
    ss_pt = ss.isel(location=5)
    assert isinstance(ss_pt, SpectraSet)
    assert not ss_pt.has_location


def test_spectraset_ds_property():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    assert isinstance(ss.ds, mikeio.Dataset)


def test_spectraset_iter():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    items = list(ss)
    assert all(isinstance(i, SpectraArray) for i in items)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_spectraarray_repr():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    r = repr(spec)
    assert "SpectraArray" in r
    assert "frequency:" in r
    assert "direction:" in r
    assert "time:" in r


def test_spectraset_repr():
    ds = mikeio.read(PT_DFSU)
    ss = SpectraSet(ds)
    r = repr(ss)
    assert "SpectraSet" in r
    assert "frequency:" in r
    assert "direction:" in r


# ---------------------------------------------------------------------------
# Mikeio axes consistency (axes match geometry)
# ---------------------------------------------------------------------------


def test_freq_matches_geometry():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    np.testing.assert_array_equal(spec.freq, da.geometry.frequencies)


def test_direction_matches_geometry():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    np.testing.assert_array_equal(spec.direction, da.geometry.directions)


def test_time_matches_da():
    da = mikeio.read(PT_DFSU)[0]
    spec = SpectraArray(da)
    np.testing.assert_array_equal(spec.time, np.asarray(da.time))
