from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import mikeio
from mikeio.spatial._FM_geometry_spectral import (
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
    GeometryFMPointSpectrum,
)

from pywswat._utils import arr_or_none, swap_last_two

if TYPE_CHECKING:
    from pywswat._plot import SpectraPlotter

_LOCATION_DIMS = frozenset({"element", "node"})

EUM_mapping = {
    "Hm0": (mikeio.EUMType.Significant_wave_height, mikeio.EUMUnit.meter),
    "Tp": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "T01": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "T02": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "Tm10": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "MWD": (mikeio.EUMType.Mean_Wave_Direction, mikeio.EUMUnit.degree),
    "PWD": (mikeio.EUMType.Wave_direction, mikeio.EUMUnit.degree),
    "DSD": (mikeio.EUMType.Directional_Standard_Deviation, mikeio.EUMUnit.degree),
}


def _make_spectral_geom(
    geom: object,
    frequencies: np.ndarray | None,
    directions: np.ndarray | None,
) -> GeometryFMPointSpectrum | GeometryFMAreaSpectrum | GeometryFMLineSpectrum:
    """Return a new spectral geometry with the given freq/dir arrays.

    Pass *None* for an axis to omit it from the new geometry.
    Supports :class:`~mikeio.spatial.GeometryFMPointSpectrum`,
    :class:`~mikeio.spatial.GeometryFMAreaSpectrum`, and
    :class:`~mikeio.spatial.GeometryFMLineSpectrum`.
    """
    if isinstance(geom, GeometryFMPointSpectrum):
        return GeometryFMPointSpectrum(frequencies=frequencies, directions=directions)
    if isinstance(geom, (GeometryFMAreaSpectrum, GeometryFMLineSpectrum)):
        cls = type(geom)
        return cls(
            node_coordinates=geom.node_coordinates,  # type: ignore[union-attr]
            element_table=geom.element_table,  # type: ignore[union-attr]
            codes=geom.codes,  # type: ignore[union-attr]
            projection=geom.projection_string,  # type: ignore[union-attr]
            dfsu_type=geom._type,  # type: ignore[union-attr]
            frequencies=frequencies,
            directions=directions,
        )
    raise TypeError(
        f"Unsupported spectral geometry: {type(geom).__name__!r}. "
        "Expected GeometryFMPointSpectrum, GeometryFMAreaSpectrum, or "
        "GeometryFMLineSpectrum."
    )


def _make_params_geometry(geom: object) -> object | None:
    """Return the non-spectral output geometry for wave-parameter DataArrays.

    * ``GeometryFMAreaSpectrum`` → :class:`~mikeio.spatial.GeometryFM2D` (Dfsu2D)
    * ``GeometryFMLineSpectrum`` → :class:`~mikeio.spatial.GeometryFM2D` (Dfsu1D)
    * ``GeometryFMPointSpectrum`` → ``None`` (time-series only, no spatial dims)
    """
    from mikecore.DfsuFile import DfsuFileType

    if isinstance(geom, GeometryFMAreaSpectrum):
        return mikeio.spatial.GeometryFM2D(
            node_coordinates=geom.node_coordinates,  # type: ignore[union-attr]
            element_table=geom.element_table,  # type: ignore[union-attr]
            codes=geom.codes,  # type: ignore[union-attr]
            projection=geom.projection_string,  # type: ignore[union-attr]
            dfsu_type=DfsuFileType.Dfsu2D,
        )
    if isinstance(geom, GeometryFMLineSpectrum):
        return mikeio.spatial.GeometryFM2D(
            node_coordinates=geom.node_coordinates,  # type: ignore[union-attr]
            element_table=geom.element_table,  # type: ignore[union-attr]
            codes=geom.codes,  # type: ignore[union-attr]
            projection=geom.projection_string,  # type: ignore[union-attr]
            dfsu_type=DfsuFileType.Dfsu1D,
        )
    return None  # GeometryFMPointSpectrum — time series, no spatial geometry


def _params_to_dataset(
    params: dict[str, float | np.ndarray],
    time: pd.DatetimeIndex,
    has_time: bool,
    geometry: object | None = None,
) -> mikeio.Dataset:
    """Convert a wave-parameter dict to a :class:`mikeio.Dataset`.

    Parameters
    ----------
    params :
        Output of :meth:`SpectraArray._compute_params`.
    time :
        Time axis from the underlying DataArray.
    has_time :
        Whether the spectrum has more than one time step.
    geometry :
        Non-spectral output geometry (from :func:`_make_params_geometry`).
        ``None`` produces a time-series-only Dataset (point spectra).
    """
    da_list = []
    for key, val in params.items():
        if key not in EUM_mapping:
            continue
        eum_type, eum_unit = EUM_mapping[key]
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            # Scalar (point, single time step) — add time dim
            arr = arr.reshape(1)
        elif not has_time:
            # Multi-location, single time step: (nloc,) → (1, nloc)
            arr = arr[np.newaxis]
        kwargs: dict = dict(data=arr, time=time, type=eum_type, unit=eum_unit, name=key)
        if geometry is not None:
            kwargs["geometry"] = geometry
        da_list.append(mikeio.DataArray(**kwargs))
    return mikeio.Dataset(da_list)


class SpectraArray:
    """Wave spectral energy density, wrapping a :class:`mikeio.DataArray`.

    Supports all mikeio spectral geometry types:

    * ``GeometryFMPointSpectrum``  — single point, no location axis
    * ``GeometryFMLineSpectrum``   — spectra along a transect (node axis)
    * ``GeometryFMAreaSpectrum``   — spectra over a mesh area (element axis)

    Parameters
    ----------
    da : mikeio.DataArray
        A DataArray with spectral geometry.

    Examples
    --------
    >>> import mikeio
    >>> da = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")[0]
    >>> spec = SpectraArray(da)
    >>> spec.has_freq
    True
    >>> spec.has_dir
    True
    """

    def __init__(self, da: mikeio.DataArray) -> None:
        self._da = da
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        geom = self._da.geometry
        has_f = arr_or_none(getattr(geom, "frequencies", None)) is not None
        has_d = arr_or_none(getattr(geom, "directions", None)) is not None
        if not has_f and not has_d:
            raise ValueError(
                "The DataArray geometry has neither a frequency nor a direction axis. "
                "Expected a spectral DataArray with GeometryFMPointSpectrum, "
                "GeometryFMAreaSpectrum, or GeometryFMLineSpectrum geometry."
            )

    # ------------------------------------------------------------------
    # Internal axis helpers
    # ------------------------------------------------------------------

    @property
    def _raw_freq(self) -> np.ndarray | None:
        return arr_or_none(getattr(self._da.geometry, "frequencies", None))

    @property
    def _raw_direction(self) -> np.ndarray | None:
        return arr_or_none(getattr(self._da.geometry, "directions", None))

    @property
    def _raw_time(self) -> np.ndarray | None:
        t = np.asarray(self._da.time)
        return t if len(t) > 1 else None

    @property
    def _raw_data(self) -> np.ndarray:
        """Data in pywswat axis order ``(time?, location?, freq?, dir?)``.

        mikeio stores spectral data as ``(..., direction, frequency)``.
        This property swaps those last two axes and drops a leading
        single-step time axis.
        """
        v = np.asarray(self._da.values)
        if self.has_freq and self.has_dir:
            v = swap_last_two(v)  # (..., dir, freq) → (..., freq, dir)
        if len(self._da.time) == 1:
            v = v[0]
        return v

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def da(self) -> mikeio.DataArray:
        """The underlying mikeio DataArray (zero-copy)."""
        return self._da

    @property
    def geometry(self) -> object:
        """The underlying mikeio spectral geometry."""
        return self._da.geometry

    @property
    def name(self) -> str:
        """Item name from the underlying DataArray."""
        return self._da.name

    @property
    def data(self) -> np.ndarray:
        """Copy of the spectral data in pywswat axis order ``(time?, location?, freq?, dir?)``."""
        return self._raw_data.copy()

    @property
    def freq(self) -> np.ndarray | None:
        f = self._raw_freq
        return f.copy() if f is not None else None

    @property
    def direction(self) -> np.ndarray | None:
        d = self._raw_direction
        return d.copy() if d is not None else None

    @property
    def time(self) -> np.ndarray | None:
        t = self._raw_time
        return t.copy() if t is not None else None

    @property
    def has_freq(self) -> bool:
        return self._raw_freq is not None

    @property
    def has_dir(self) -> bool:
        return self._raw_direction is not None

    @property
    def has_time(self) -> bool:
        return self._raw_time is not None

    @property
    def has_location(self) -> bool:
        """True for line and area spectra (node / element dimension)."""
        dims = tuple(getattr(self._da, "dims", ()))
        return bool(_LOCATION_DIMS & set(dims))

    @property
    def nf(self) -> int | None:
        f = self._raw_freq
        return len(f) if f is not None else None

    @property
    def nd(self) -> int | None:
        d = self._raw_direction
        return len(d) if d is not None else None

    @property
    def nt(self) -> int | None:
        t = self._raw_time
        return len(t) if t is not None else None

    @property
    def nloc(self) -> int | None:
        """Number of spatial locations (elements or nodes), or *None*."""
        if not self.has_location:
            return None
        dims = list(getattr(self._da, "dims", ()))
        loc_dim = next(d for d in dims if d in _LOCATION_DIMS)
        return int(self._da.values.shape[dims.index(loc_dim)])

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape in mikeio axis order: ``(time?, element/node?, direction?, frequency?)``."""
        return self._da.values.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # ------------------------------------------------------------------
    # Internal computation helpers (work in pywswat axis order)
    # ------------------------------------------------------------------

    def _axis_index(self, name: str) -> int:
        if name == "time":
            return 0
        if name == "location":
            return int(self.has_time)
        if name == "freq":
            return int(self.has_time) + int(self.has_location)
        if name == "dir":
            return int(self.has_time) + int(self.has_location) + int(self.has_freq)
        raise ValueError(f"Unknown axis name: {name!r}")

    def _delta_freq(self) -> np.ndarray:
        f = self._raw_freq
        assert f is not None
        nf = len(f)
        if nf == 1:
            return np.array([f[0]])
        df = np.empty(nf)
        df[1:-1] = (f[2:] - f[:-2]) / 2.0
        df[0] = f[1] - f[0]
        df[-1] = f[-1] - f[-2]
        return df

    def _delta_dir(self) -> float:
        d = self._raw_direction
        assert d is not None
        return 2.0 * np.pi / len(d)

    def _freq_spectrum(self) -> np.ndarray:
        """S(f) — integrate direction if present."""
        data = self._raw_data
        if self.has_dir:
            return np.sum(data, axis=self._axis_index("dir")) * self._delta_dir()
        return data

    def _moments(self, orders: list[int]) -> dict[int, np.ndarray]:
        """Spectral moments mn = Σ f^n · S(f) · Δf."""
        f = self._raw_freq
        assert f is not None
        Sf = self._freq_spectrum()
        df = self._delta_freq()
        freq_axis = self._axis_index("freq")
        return {n: np.sum((f**n) * Sf * df, axis=freq_axis) for n in orders}

    def _peak_freq_index(self) -> np.ndarray | int:
        return np.argmax(self._freq_spectrum(), axis=self._axis_index("freq"))

    def _directional_fourier_coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        dirs = self._raw_direction
        assert dirs is not None
        theta = np.deg2rad(dirs)
        df = self._delta_freq()
        ddir = self._delta_dir()
        data = self._raw_data
        dir_axis = self._axis_index("dir")
        freq_axis = self._axis_index("freq")

        Sf_cos = np.sum(data * np.cos(theta) * ddir, axis=dir_axis)
        Sf_sin = np.sum(data * np.sin(theta) * ddir, axis=dir_axis)
        Sf_all = np.sum(data * ddir, axis=dir_axis)

        a1_num = np.sum(Sf_cos * df, axis=freq_axis)
        b1_num = np.sum(Sf_sin * df, axis=freq_axis)
        m0_2d = np.sum(Sf_all * df, axis=freq_axis)

        safe_m0 = np.where(m0_2d > 0, m0_2d, np.nan)
        return a1_num / safe_m0, b1_num / safe_m0

    def _mwd(self) -> np.ndarray | float:
        a1, b1 = self._directional_fourier_coeffs()
        mwd = np.rad2deg(np.arctan2(b1, a1)) % 360.0
        return float(mwd) if np.ndim(mwd) == 0 else mwd

    def _pwd(self) -> np.ndarray | float:
        dirs = self._raw_direction
        assert dirs is not None
        data = self._raw_data
        freq_axis = self._axis_index("freq")
        peak_fi = np.asarray(self._peak_freq_index())
        nd = len(dirs)

        if peak_fi.ndim == 0:
            return float(dirs[np.argmax(data[int(peak_fi), :])])

        idx = peak_fi[..., np.newaxis, np.newaxis]
        idx = np.broadcast_to(idx, idx.shape[:-2] + (1, nd))
        S_at_peak = np.take_along_axis(data, idx, axis=freq_axis).squeeze(
            axis=freq_axis
        )
        return dirs[np.argmax(S_at_peak, axis=-1)]

    def _dsd(self) -> np.ndarray | float:
        a1, b1 = self._directional_fourier_coeffs()
        r = np.clip(np.sqrt(a1**2 + b1**2), 0.0, 1.0)
        dsd = np.sqrt(2.0 * (1.0 - r))
        return float(dsd) if np.ndim(dsd) == 0 else dsd

    def _compute_params(self) -> dict[str, float | np.ndarray]:
        result: dict[str, float | np.ndarray] = {}
        if self.has_freq:
            f = self._raw_freq
            assert f is not None
            moments = self._moments([-1, 0, 1, 2])
            m_neg1, m0, m1, m2 = moments[-1], moments[0], moments[1], moments[2]

            safe_m0 = np.where(m0 > 0, m0, np.nan)
            safe_m1 = np.where(m1 > 0, m1, np.nan)
            safe_m2 = np.where(m2 > 0, m2, np.nan)

            peak_fi = self._peak_freq_index()
            result["Hm0"] = 4.0 * np.sqrt(safe_m0)
            result["Tp"] = 1.0 / f[peak_fi]
            result["T01"] = safe_m0 / safe_m1
            result["T02"] = np.sqrt(safe_m0 / safe_m2)
            result["Tm10"] = m_neg1 / safe_m0

            if self.has_dir:
                result["MWD"] = self._mwd()
                result["PWD"] = self._pwd()
                result["DSD"] = self._dsd()

        if not self.has_time and not self.has_location:
            result = {
                k: float(v) if isinstance(v, np.ndarray) else v
                for k, v in result.items()
            }
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_params(self) -> mikeio.Dataset:
        """Compute integrated wave parameters as a :class:`mikeio.Dataset`.

        Returns
        -------
        mikeio.Dataset
            Dataset with items ``Hm0``, ``Tp``, ``T01``, ``T02``, ``Tm10``
            and, when a direction axis is present, also ``MWD``, ``PWD``,
            ``DSD``.  The time axis matches the underlying DataArray.

        Examples
        --------
        >>> import mikeio
        >>> da = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")[0]
        >>> spec = SpectraArray(da)
        >>> ds = spec.to_params()
        >>> isinstance(ds, mikeio.Dataset)
        True
        >>> "Hm0" in ds.names
        True
        """
        return _params_to_dataset(
            self._compute_params(),
            self._da.time,
            self.has_time,
            geometry=_make_params_geometry(self._da.geometry),
        )

    def isel(self, location: int) -> "SpectraArray":
        """Select a single location by index.

        Uses mikeio's native ``isel``, preserving time and spectral axes.

        Parameters
        ----------
        location : int
            Index into the location (element or node) axis.

        Returns
        -------
        SpectraArray
            New instance without the location axis and with
            ``GeometryFMPointSpectrum`` geometry.

        Raises
        ------
        ValueError
            If no location axis is present.

        Examples
        --------
        >>> import mikeio
        >>> da = mikeio.read("tests/testdata/spectra/area_spectra.dfsu")[0]
        >>> spec = SpectraArray(da)
        >>> pt = spec.isel(location=5)
        >>> pt.has_location
        False
        """
        if not self.has_location:
            raise ValueError(
                "No location axis. isel() requires a multi-location spectrum "
                "(GeometryFMAreaSpectrum or GeometryFMLineSpectrum)."
            )
        dims = tuple(getattr(self._da, "dims", ()))
        loc_dim = next(d for d in dims if d in _LOCATION_DIMS)
        return SpectraArray(self._da.isel(**{loc_dim: location}))

    def integrate_dir(self) -> "SpectraArray":
        """Integrate over the direction axis.

        Operates directly on the underlying DataArray values in mikeio's
        native axis order and constructs a new DataArray with a direction-free
        spectral geometry of the same type (point, area, or line).

        Returns
        -------
        SpectraArray
            New instance without the direction axis.

        Raises
        ------
        ValueError
            If no direction axis is present.

        Examples
        --------
        >>> import mikeio
        >>> da = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")[0]
        >>> spec = SpectraArray(da)
        >>> spec_f = spec.integrate_dir()
        >>> spec_f.has_dir
        False
        >>> spec_f.has_freq
        True
        """
        if not self.has_dir:
            raise ValueError("No direction axis to integrate over.")
        nd = self.nd
        assert nd is not None
        ddir = 2.0 * np.pi / nd
        # mikeio order: (..., direction, frequency) — direction is axis -2
        new_vals = np.sum(self._da.values, axis=-2) * ddir  # (..., frequency)
        new_geom = _make_spectral_geom(
            self._da.geometry, frequencies=self._raw_freq, directions=None
        )
        return SpectraArray(
            mikeio.DataArray(data=new_vals, time=self._da.time, geometry=new_geom)
        )

    def sel_freq(self, fmin: float = 0.0, fmax: float = float("inf")) -> "SpectraArray":
        """Slice the frequency axis to ``[fmin, fmax]``.

        Constructs a new DataArray with updated spectral geometry containing
        only the selected frequencies, preserving all other axes.

        Parameters
        ----------
        fmin : float
            Lower frequency bound (inclusive), in Hz.
        fmax : float
            Upper frequency bound (inclusive), in Hz.

        Returns
        -------
        SpectraArray
            New instance with the sliced frequency axis.

        Raises
        ------
        ValueError
            If no frequency axis is present.

        Examples
        --------
        >>> import mikeio
        >>> da = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")[0]
        >>> spec = SpectraArray(da)
        >>> spec2 = spec.sel_freq(fmin=0.1, fmax=0.3)
        >>> bool(spec2.freq[0] >= 0.1)
        True
        >>> bool(spec2.freq[-1] <= 0.3)
        True
        """
        if not self.has_freq:
            raise ValueError("No frequency axis to slice.")
        freq = self._raw_freq
        assert freq is not None
        mask = (freq >= fmin) & (freq <= fmax)
        idx = np.where(mask)[0]
        # mikeio order: (..., direction, frequency) — frequency is axis -1
        new_vals = self._da.values[..., idx]
        new_geom = _make_spectral_geom(
            self._da.geometry, frequencies=freq[idx], directions=self._raw_direction
        )
        return SpectraArray(
            mikeio.DataArray(data=new_vals, time=self._da.time, geometry=new_geom)
        )

    # ------------------------------------------------------------------
    # Plot accessor
    # ------------------------------------------------------------------

    @property
    def plot(self) -> "SpectraPlotter":
        """Plotting namespace — call ``spec.plot()`` or use sub-methods."""
        from pywswat._plot import SpectraPlotter

        return SpectraPlotter(self)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dims = self._da.dims
        shape = self._da.values.shape
        dims_str = ", ".join(f"{d}:{s}" for d, s in zip(dims, shape))
        lines = [f"<pywswat.SpectraArray>  name: {self.name}", f"dims: ({dims_str})"]

        f = self._raw_freq
        if f is not None:
            lines.append(f"frequency: {f[0]:.3f} - {f[-1]:.3f} Hz  ({self.nf} bins)")
        d = self._raw_direction
        if d is not None:
            lines.append(f"direction: {d[0]:.0f} - {d[-1]:.0f} deg  ({self.nd} bins)")
        t = self._raw_time
        if t is not None:
            lines.append(f"time: {t[0]} - {t[-1]}  ({self.nt} steps)")

        return "\n".join(lines)


# Backward-compatible alias
Spectra = SpectraArray
