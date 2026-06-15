"""Core spectral data structures — pure-numpy, no mikeio geometry.

The :class:`SpectraArray` stores wave spectral energy density as a plain
numpy array in pywswat axis order ``(time?, loc?, freq?, dir?)``.  All
spectral computation (moments, wave parameters, axis reductions) runs here.
mikeio geometry is accessed only through :mod:`pywswat.mikeio._adapters` at I/O
time and never used for computation.

:class:`SpectraArray` also accepts a ``mikeio.DataArray`` as its first
argument for backward compatibility; in that case it parses the mikeio
object into numpy arrays internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pywswat._utils import arr_or_none

if TYPE_CHECKING:
    from pywswat.plot._plot import SpectraPlotter


_SENTINEL = object()


def _fm_loc_dim_name(io_geom: Any) -> str:
    """Return the mikeio location dim name for an FM spectral geometry."""
    try:
        from mikeio.spatial._FM_geometry_spectral import (
            GeometryFMAreaSpectrum,
            GeometryFMLineSpectrum,
        )

        if isinstance(io_geom, GeometryFMAreaSpectrum):
            return "element"
        if isinstance(io_geom, GeometryFMLineSpectrum):
            return "node"
    except Exception:
        pass
    return "location"


class SpectraArray:
    """Wave spectral energy density backed by a pure-numpy array.

    Internal axis order: ``(time?, loc?, freq?, dir?)``
    The time dimension is **absent** from the stored array when there is only
    a single time step (``has_time`` is ``False``).

    Parameters
    ----------
    data : np.ndarray or mikeio.DataArray
        Spectral data in pywswat axis order ``(time?, loc?, freq?, dir?)``.
        Pass a ``mikeio.DataArray`` for backward compatibility — it will be
        parsed automatically.
    freq : array-like of float, optional
        Frequency axis [Hz].
    direction : array-like of float, optional
        Direction axis [degrees].
    time : pd.DatetimeIndex, optional
        Time axis.  A single-element index means one time step with no
        leading time dimension in ``data`` (``has_time`` will be ``False``).
        Defaults to a single epoch (``1970-01-01``).
    x : array-like of float, optional
        x-coordinates for multi-location spectra, shape ``(n_locs,)``.
    y : array-like of float, optional
        y-coordinates for multi-location spectra, shape ``(n_locs,)``.
    name : str, optional
        Item name (default ``"Energy density"``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pywswat.core import SpectraArray
    >>> freqs = np.linspace(0.05, 0.5, 32)
    >>> dirs = np.arange(0, 360, 10, dtype=float)
    >>> data = np.ones((len(freqs), len(dirs)))   # (nf, nd) pywswat order
    >>> t = pd.DatetimeIndex(["2000-01-01"])
    >>> spec = SpectraArray(data, freq=freqs, direction=dirs, time=t)
    >>> spec.has_freq and spec.has_dir
    True
    >>> spec.has_location
    False
    """

    def __init__(
        self,
        data: "np.ndarray | Any",
        *,
        freq: np.ndarray | None = None,
        direction: np.ndarray | None = None,
        time: pd.DatetimeIndex | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        name: str = "Energy density",
        _io_geom: Any = None,
        _grid_shape: "tuple[int, ...] | None" = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            self._init_from_da(data)
            return

        self._data = np.asarray(data, dtype=float)
        self._freq = arr_or_none(freq)
        self._direction = arr_or_none(direction)
        self._time: pd.DatetimeIndex = (
            time if time is not None else pd.DatetimeIndex(["1970-01-01"])
        )
        self._x = np.asarray(x, dtype=float) if x is not None else None
        self._y = np.asarray(y, dtype=float) if y is not None else None
        self._name = str(name)
        self._io_geom: Any = _io_geom
        self._grid_shape: tuple[int, ...] | None = _grid_shape
        self._validate()

    def _init_from_da(self, da: Any) -> None:
        """Parse a mikeio.DataArray (backward-compat path)."""
        from pywswat.mikeio._adapters import _parse_da as _p

        parsed = _p(da)
        self.__dict__.update(parsed.__dict__)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self._freq is None and self._direction is None:
            raise ValueError(
                "SpectraArray requires at least one spectral axis "
                "(freq or direction). "
                "Expected a DataArray with GeometryFMPointSpectrum, "
                "GeometryFMAreaSpectrum, or GeometryFMLineSpectrum geometry."
            )

    # ------------------------------------------------------------------
    # Internal copy helper
    # ------------------------------------------------------------------

    def _copy_with(
        self,
        data: np.ndarray,
        *,
        freq: Any = _SENTINEL,
        direction: Any = _SENTINEL,
        time: Any = _SENTINEL,
        x: Any = _SENTINEL,
        y: Any = _SENTINEL,
        name: Any = _SENTINEL,
        _io_geom: Any = _SENTINEL,
        _grid_shape: Any = _SENTINEL,
    ) -> "SpectraArray":
        """Return a new SpectraArray with *data* and selectively updated attrs.

        Unspecified attributes are inherited from *self*.  Pass ``None``
        explicitly to clear an axis (e.g. ``direction=None`` after
        :meth:`integrate_dir`).
        """
        sa: "SpectraArray" = SpectraArray.__new__(SpectraArray)
        sa._data = np.asarray(data, dtype=float)
        sa._freq = self._freq if freq is _SENTINEL else arr_or_none(freq)
        sa._direction = (
            self._direction if direction is _SENTINEL else arr_or_none(direction)
        )
        sa._time = self._time if time is _SENTINEL else time
        sa._x = (
            self._x
            if x is _SENTINEL
            else (np.asarray(x, dtype=float) if x is not None else None)
        )
        sa._y = (
            self._y
            if y is _SENTINEL
            else (np.asarray(y, dtype=float) if y is not None else None)
        )
        sa._name = self._name if name is _SENTINEL else str(name)
        sa._io_geom = self._io_geom if _io_geom is _SENTINEL else _io_geom
        sa._grid_shape = self._grid_shape if _grid_shape is _SENTINEL else _grid_shape
        return sa

    # ------------------------------------------------------------------
    # Properties — axes
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Item name."""
        return self._name

    @property
    def freq(self) -> np.ndarray | None:
        """Frequency axis [Hz], or ``None`` if absent."""
        return self._freq.copy() if self._freq is not None else None

    @property
    def direction(self) -> np.ndarray | None:
        """Direction axis [degrees], or ``None`` if absent."""
        return self._direction.copy() if self._direction is not None else None

    @property
    def time(self) -> np.ndarray | None:
        """Time axis as numpy datetime64 array, or ``None`` for a single step."""
        return np.asarray(self._time) if len(self._time) > 1 else None

    @property
    def x(self) -> np.ndarray | None:
        """x-coordinates ``(n_locs,)``, or ``None`` for point spectra."""
        return self._x.copy() if self._x is not None else None

    @property
    def y(self) -> np.ndarray | None:
        """y-coordinates ``(n_locs,)``, or ``None`` for point spectra."""
        return self._y.copy() if self._y is not None else None

    @property
    def data(self) -> np.ndarray:
        """Copy of spectral data.

        For flat (FM-mesh) spectra: ``(time?, loc, freq?, dir?)``.
        For grid spectra: ``(time?, *grid_shape, freq?, dir?)`` — the location
        axis is expanded back to the original grid dimensions (e.g. ``(ny, nx)``
        for Grid2D).
        """
        d = self._data.copy()
        if self._grid_shape is not None and self.has_location:
            loc_ax = self._axis_index("location")
            s = list(d.shape)
            s[loc_ax : loc_ax + 1] = list(self._grid_shape)
            d = d.reshape(s)
        return d

    # ------------------------------------------------------------------
    # Properties — boolean flags
    # ------------------------------------------------------------------

    @property
    def has_freq(self) -> bool:
        """``True`` when a frequency axis is present."""
        return self._freq is not None

    @property
    def has_dir(self) -> bool:
        """``True`` when a direction axis is present."""
        return self._direction is not None

    @property
    def has_time(self) -> bool:
        """``True`` when more than one time step is present."""
        return len(self._time) > 1

    @property
    def has_location(self) -> bool:
        """``True`` for multi-location (line / area) spectra."""
        return self._x is not None

    # ------------------------------------------------------------------
    # Properties — axis sizes
    # ------------------------------------------------------------------

    @property
    def nf(self) -> int | None:
        """Number of frequency bins, or ``None``."""
        return len(self._freq) if self._freq is not None else None

    @property
    def nd(self) -> int | None:
        """Number of direction bins, or ``None``."""
        return len(self._direction) if self._direction is not None else None

    @property
    def nt(self) -> int | None:
        """Number of time steps, or ``None`` for a single step."""
        t = self.time
        return len(t) if t is not None else None

    @property
    def nloc(self) -> int | None:
        """Number of spatial locations, or ``None`` for point spectra."""
        if not self.has_location:
            return None
        return int(self._data.shape[self._axis_index("location")])

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape in mikeio axis order: ``(time, loc?, direction?, frequency?)``."""
        s = list(self._data.shape)
        if not self.has_time:
            s = [1] + s
        if self.has_freq and self.has_dir:
            s[-2], s[-1] = s[-1], s[-2]
        return tuple(s)

    @property
    def ndim(self) -> int:
        """Number of dimensions (mikeio order)."""
        return len(self.shape)

    # ------------------------------------------------------------------
    # Backward-compat properties
    # ------------------------------------------------------------------

    @property
    def _raw_data(self) -> np.ndarray:
        """Alias for ``_data`` — kept for backward compatibility with tests."""
        return self._data

    @property
    def da(self) -> "Any":
        """Reconstruct a mikeio DataArray (calls :meth:`to_mikeio`)."""
        return self.to_mikeio()

    @property
    def geometry(self) -> "Any":
        """Spectral geometry of the reconstructed mikeio DataArray."""
        return self.to_mikeio().geometry

    def to_mikeio(self) -> "Any":
        """Reconstruct a :class:`mikeio.DataArray` from this spectrum.

        The spectral geometry is inferred from the stored ``_io_geom``
        reference (set when reading from a mikeio file) or built from
        :attr:`x` / :attr:`y` coordinates when no reference is available.

        Returns
        -------
        mikeio.DataArray
        """
        from pywswat.mikeio._adapters import to_mikeio as _to_mikeio

        return _to_mikeio(self)

    # ------------------------------------------------------------------
    # Internal axis helpers
    # ------------------------------------------------------------------

    def _axis_index(self, name: str) -> int:
        """Return the axis index for a named dimension in pywswat order."""
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
        f = self._freq
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
        d = self._direction
        assert d is not None
        return 2.0 * np.pi / len(d)

    def _freq_spectrum(self) -> np.ndarray:
        """S(f) — integrate direction if present."""
        data = self._data
        if self.has_dir:
            return np.sum(data, axis=self._axis_index("dir")) * self._delta_dir()
        return data

    def _moments(self, orders: list[int]) -> dict[int, np.ndarray]:
        """Spectral moments mn = Σ f^n · S(f) · Δf."""
        f = self._freq
        assert f is not None
        Sf = self._freq_spectrum()
        df = self._delta_freq()
        freq_axis = self._axis_index("freq")
        return {n: np.sum((f**n) * Sf * df, axis=freq_axis) for n in orders}

    def _peak_freq_index(self) -> "np.ndarray | int":
        return np.argmax(self._freq_spectrum(), axis=self._axis_index("freq"))

    def _directional_fourier_coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        dirs = self._direction
        assert dirs is not None
        theta = np.deg2rad(dirs)
        df = self._delta_freq()
        ddir = self._delta_dir()
        data = self._data
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

    def _mwd(self) -> "np.ndarray | float":
        a1, b1 = self._directional_fourier_coeffs()
        mwd = np.rad2deg(np.arctan2(b1, a1)) % 360.0
        return float(mwd) if np.ndim(mwd) == 0 else mwd

    def _pwd(self) -> "np.ndarray | float":
        dirs = self._direction
        assert dirs is not None
        data = self._data
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

    def _dsd(self) -> "np.ndarray | float":
        a1, b1 = self._directional_fourier_coeffs()
        r = np.clip(np.sqrt(a1**2 + b1**2), 0.0, 1.0)
        dsd = np.sqrt(2.0 * (1.0 - r))
        return float(dsd) if np.ndim(dsd) == 0 else dsd

    # ------------------------------------------------------------------
    # Public API — wave parameters
    # ------------------------------------------------------------------

    def to_params(self) -> "dict[str, np.ndarray | float]":
        """Compute integrated wave parameters.

        Returns
        -------
        dict[str, np.ndarray | float]
            Wave parameters keyed by name: ``Hm0``, ``Tp``, ``T01``,
            ``T02``, ``Tm10`` and, when a direction axis is present,
            ``MWD``, ``PWD``, ``DSD``.

            Values are numpy arrays shaped ``(time?, loc?)`` or Python
            ``float`` for single-step point spectra.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from pywswat.core import SpectraArray
        >>> freq = np.linspace(0.05, 0.5, 32)
        >>> data = np.exp(-0.5 * ((freq - 0.1) / 0.02) ** 2)
        >>> spec = SpectraArray(data, freq=freq)
        >>> params = spec.to_params()
        >>> "Hm0" in params
        True
        >>> "MWD" in params
        False
        """
        result: dict[str, np.ndarray | float] = {}
        if self.has_freq:
            f = self._freq
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
        elif self._grid_shape is not None and self.has_location:
            # Reshape flat location axis back to original grid dimensions.
            # Parameter arrays have shape (time?, nloc) — reshape to (time?, *grid_shape).
            result = {
                k: (
                    v.reshape(v.shape[:-1] + self._grid_shape)
                    if isinstance(v, np.ndarray)
                    else v
                )
                for k, v in result.items()
            }
        return result

    # ------------------------------------------------------------------
    # Public API — axis operations
    # ------------------------------------------------------------------

    def integrate_dir(self) -> "SpectraArray":
        """Integrate over the direction axis.

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
        >>> import numpy as np, mikeio
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
        new_data = np.sum(self._data, axis=self._axis_index("dir")) * ddir
        return self._copy_with(new_data, direction=None)

    def sel_freq(
        self, fmin: None | float = 0.0, fmax: None | float = float("inf")
    ) -> "SpectraArray":
        """Slice the frequency axis to ``[fmin, fmax]``.

        Parameters
        ----------
        fmin : float, optional
            Lower frequency bound (inclusive) [Hz].
        fmax : float
            Upper frequency bound (inclusive) [Hz].

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
        >>> import numpy as np, mikeio
        >>> da = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")[0]
        >>> spec = SpectraArray(da)
        >>> spec2 = spec.sel_freq(fmin=0.1, fmax=0.3)
        >>> bool(spec2.freq[0] >= 0.1)
        True
        >>> bool(spec2.freq[-1] <= 0.3)
        True
        """
        if fmin is None:
            fmin = 0.0
        if fmax is None:
            fmax = float("inf")
        if not self.has_freq:
            raise ValueError("No frequency axis to slice.")
        freq = self._freq
        assert freq is not None
        idx = np.where((freq >= fmin) & (freq <= fmax))[0]
        new_data = np.take(self._data, idx, axis=self._axis_index("freq"))
        return self._copy_with(new_data, freq=freq[idx])

    def isel(self, location: int) -> "SpectraArray":
        """Select a single location by integer index, returning a point spectrum.

        Parameters
        ----------
        location : int
            Index into the location axis.

        Returns
        -------
        SpectraArray
            New instance without the location axis.

        Raises
        ------
        ValueError
            If no location axis is present.

        Examples
        --------
        >>> import numpy as np, mikeio
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
        new_data = np.take(self._data, location, axis=self._axis_index("location"))
        return self._copy_with(
            new_data, x=None, y=None, _io_geom=None, _grid_shape=None
        )

    # ------------------------------------------------------------------
    # Plot accessor
    # ------------------------------------------------------------------

    @property
    def plot(self) -> "SpectraPlotter":
        """Plotting namespace — call ``spec.plot()`` or use sub-methods."""
        from pywswat.plot._plot import SpectraPlotter

        return SpectraPlotter(self)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        # Build dims string from own attributes — no to_mikeio() call needed.
        dim_parts: list[str] = []

        # Time
        if self.has_time:
            dim_parts.append(f"time:{self.nt}")
        else:
            dim_parts.append("time:1")

        # Spatial location
        if self.has_location:
            if self._grid_shape is not None:
                # Grid geometry: show individual axes (y,x or x)
                ax_names = ["y", "x"] if len(self._grid_shape) == 2 else ["x"]
                for ax, sz in zip(ax_names, self._grid_shape):
                    dim_parts.append(f"{ax}:{sz}")
            else:
                # FM mesh: "element" for area, "node" for line, fallback "location"
                loc_name = _fm_loc_dim_name(self._io_geom)
                dim_parts.append(f"{loc_name}:{self.nloc}")

        # Spectral axes (direction before frequency — mikeio convention)
        if self.has_dir:
            dim_parts.append(f"direction:{self.nd}")
        if self.has_freq:
            dim_parts.append(f"frequency:{self.nf}")

        lines = [
            f"<pywswat.SpectraArray>  name: {self._name}",
            f"dims: ({', '.join(dim_parts)})",
        ]

        f = self._freq
        if f is not None:
            lines.append(f"frequency: {f[0]:.4f} - {f[-1]:.4f} Hz  ({self.nf} bins)")
        d = self._direction
        if d is not None:
            lines.append(f"direction: {d[0]:.1f} - {d[-1]:.1f} deg  ({self.nd} bins)")
        t = self.time
        if t is not None:
            lines.append(f"time: {t[0]} - {t[-1]}  ({self.nt} steps)")
        if self.has_location:
            if self._grid_shape is not None and self._x is not None:
                lines.append(
                    f"x: [{self._x.min():.2f}, {self._x.max():.2f}]  "
                    f"y: [{self._y.min():.2f}, {self._y.max():.2f}]"
                    if self._y is not None
                    else f"x: [{self._x.min():.2f}, {self._x.max():.2f}]"
                )
            elif self._x is not None:
                lines.append(
                    f"locations: {self.nloc}  "
                    f"x: [{self._x.min():.2f}, {self._x.max():.2f}]  "
                    f"y: [{self._y.min():.2f}, {self._y.max():.2f}]"
                    if self._y is not None
                    else f"locations: {self.nloc}  x: [{self._x.min():.2f}, {self._x.max():.2f}]"
                )

        return "\n".join(lines)


# Backward-compatible alias
Spectra = SpectraArray
