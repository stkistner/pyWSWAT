"""Adapters between :class:`~pywswat.core.SpectraArray` and mikeio types.

Also owns all mikeio geometry construction helpers (FM-mesh and Grid),
keeping geometry concerns out of the spectral-analysis layer.

Functions
---------
from_mikeio(da)
    Parse a ``mikeio.DataArray`` into a :class:`~pywswat.core.SpectraArray`.
to_mikeio(spec, geometry=None)
    Reconstruct a ``mikeio.DataArray`` from a :class:`~pywswat.core.SpectraArray`.
to_params_dataset(spec, geometry=None)
    Compute wave parameters and return them as a ``mikeio.Dataset``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import mikeio
from mikeio.spatial._FM_geometry_spectral import (
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
    GeometryFMPointSpectrum,
)

from pywswat._utils import arr_or_none, swap_last_two

_SPEC_EUM_TYPE = mikeio.EUMType.Wave_energy_density
_UNIT_1D = mikeio.EUMUnit.meter_pow_2_sec
_UNIT_2D = mikeio.EUMUnit.meter_pow_2_sec_per_deg

_PARAM_EUM = {
    "Hm0": (mikeio.EUMType.Significant_wave_height, mikeio.EUMUnit.meter),
    "Tp": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "T01": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "T02": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "Tm10": (mikeio.EUMType.Wave_period, mikeio.EUMUnit.second),
    "MWD": (mikeio.EUMType.Mean_Wave_Direction, mikeio.EUMUnit.degree),
    "PWD": (mikeio.EUMType.Wave_direction, mikeio.EUMUnit.degree),
    "DSD": (mikeio.EUMType.Directional_Standard_Deviation, mikeio.EUMUnit.degree),
}


# ──────────────────────────────────────────────────────────────────────────────
# Public adapters
# ──────────────────────────────────────────────────────────────────────────────


def from_mikeio(da: mikeio.DataArray) -> "SpectraArray":
    """Parse a ``mikeio.DataArray`` into a core :class:`~pywswat.core.SpectraArray`."""
    return _parse_da(da)


def to_mikeio(
    spec: "SpectraArray",
    geometry: Any = None,
) -> mikeio.DataArray:
    """Reconstruct a ``mikeio.DataArray`` from a core :class:`~pywswat.core.SpectraArray`."""
    data = spec._data.copy()

    if spec.has_freq and spec.has_dir:
        data = swap_last_two(data)
    if not spec.has_time:
        data = data[np.newaxis, ...]

    geom_obj = geometry if geometry is not None else spec._io_geom
    out_geom = _build_spectral_geom(spec, geom_obj)

    unit = _UNIT_2D if spec.has_dir else _UNIT_1D
    return mikeio.DataArray(
        data=data,
        time=spec._time,
        geometry=out_geom,
        type=_SPEC_EUM_TYPE,
        unit=unit,
        name=spec.name,
    )


def to_params_dataset(
    spec: "SpectraArray",
    geometry: Any = None,
) -> mikeio.Dataset:
    """Compute wave parameters and return them as a ``mikeio.Dataset``."""
    params = spec.to_params()
    geom_obj = geometry if geometry is not None else spec._io_geom
    out_geom = _params_output_geom(spec, geom_obj)

    da_list = []
    for key, val in params.items():
        if key not in _PARAM_EUM:
            continue
        eum_type, eum_unit = _PARAM_EUM[key]
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif not spec.has_time:
            arr = arr[np.newaxis]
        kwargs: dict = dict(
            data=arr, time=spec._time, type=eum_type, unit=eum_unit, name=key
        )
        if out_geom is not None:
            kwargs["geometry"] = out_geom
        da_list.append(mikeio.DataArray(**kwargs))
    return mikeio.Dataset(da_list)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers — FM mesh (absorbed from spectral layer)
# ──────────────────────────────────────────────────────────────────────────────


def _is_grid_geom(da: mikeio.DataArray) -> bool:
    """Return True when *da* has a Grid1D or Grid2D geometry."""
    from mikeio.spatial._grid_geometry import Grid1D, Grid2D

    return isinstance(da.geometry, (Grid1D, Grid2D))


def _grid_spatial_shape(geom: object) -> tuple[int, ...]:
    """Return ``(nx,)`` for Grid1D or ``(ny, nx)`` for Grid2D."""
    from mikeio.spatial._grid_geometry import Grid1D, Grid2D

    if isinstance(geom, Grid1D):
        return (geom.nx,)
    if isinstance(geom, Grid2D):
        return (geom.ny, geom.nx)
    raise TypeError(f"Expected Grid1D or Grid2D, got {type(geom).__name__!r}")


def _grid_to_spectral_geom(
    input_geom: object,
    freqs: np.ndarray | None,
    directions: np.ndarray | None,
) -> GeometryFMLineSpectrum:
    """Map a Grid1D or Grid2D geometry to a :class:`GeometryFMLineSpectrum`.

    Every grid node becomes one spectral node.  Grid2D is flattened row-major
    (C order: y varies slowest, x varies fastest).
    """
    from mikecore.DfsuFile import DfsuFileType
    from mikeio.spatial._grid_geometry import Grid1D, Grid2D

    if isinstance(input_geom, Grid1D):
        x = input_geom.x
        y = np.zeros_like(x)
    elif isinstance(input_geom, Grid2D):
        xv, yv = np.meshgrid(input_geom.x, input_geom.y)
        x = xv.ravel()
        y = yv.ravel()
    else:
        raise TypeError(f"Expected Grid1D or Grid2D, got {type(input_geom).__name__!r}")

    n = len(x)
    node_coords = np.column_stack([x, y, np.zeros(n)])
    element_table: list[list[int]] = (
        [[0]] if n < 2 else [[i, i + 1] for i in range(n - 1)]
    )
    codes = np.zeros(n, dtype=int)

    return GeometryFMLineSpectrum(
        node_coordinates=node_coords,
        element_table=element_table,
        codes=codes,
        projection=input_geom.projection_string,
        frequencies=freqs,
        directions=directions,
        dfsu_type=DfsuFileType.DfsuSpectral1D,
    )


def _to_spectral_geom(
    input_geom: object,
    freqs: np.ndarray,
    directions: np.ndarray,
) -> GeometryFMAreaSpectrum | GeometryFMLineSpectrum:
    """Map a GeometryFM2D to its spectral counterpart."""
    from mikecore.DfsuFile import DfsuFileType
    from mikeio.spatial._FM_geometry import GeometryFM2D

    if not isinstance(input_geom, GeometryFM2D):
        raise TypeError(
            f"_to_spectral_geom requires a GeometryFM2D input geometry, "
            f"got {type(input_geom).__name__!r}."
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
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_da(da: Any) -> "SpectraArray":
    """Create a SpectraArray by parsing a mikeio DataArray."""
    from pywswat.core import SpectraArray

    if not isinstance(da, mikeio.DataArray):
        raise TypeError(
            f"Expected a mikeio.DataArray, got {type(da).__name__!r}. "
            "To build a SpectraArray from numpy arrays use the keyword "
            "arguments: SpectraArray(data, freq=..., direction=..., time=...)."
        )

    geom = da.geometry
    freq = arr_or_none(getattr(geom, "frequencies", None))
    direction = arr_or_none(getattr(geom, "directions", None))

    if freq is None and direction is None:
        raise ValueError(
            "The DataArray geometry has neither a frequency nor a direction axis. "
            "Expected a spectral DataArray with GeometryFMPointSpectrum, "
            "GeometryFMAreaSpectrum, or GeometryFMLineSpectrum geometry."
        )

    v = np.asarray(da.values, dtype=float)
    if freq is not None and direction is not None:
        v = swap_last_two(v)
    if len(da.time) == 1:
        v = v[0]

    x, y = _extract_coords(geom)

    spec: SpectraArray = SpectraArray.__new__(SpectraArray)
    spec._data = v
    spec._freq = freq
    spec._direction = direction
    spec._time = da.time
    spec._x = x
    spec._y = y
    spec._name = da.name
    spec._io_geom = geom
    spec._grid_shape = None  # FM-file reads are never structured grids
    return spec


def _extract_coords(
    geom: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract (x, y) location coordinates from a mikeio spectral geometry.

    Returns ``(None, None)`` for point spectra — a single point has no
    location *axis*, so ``has_location`` must remain ``False``.
    """
    if isinstance(geom, GeometryFMLineSpectrum):
        nc = geom.node_coordinates
        return nc[:, 0], nc[:, 1]
    if isinstance(geom, GeometryFMAreaSpectrum):
        try:
            ec = geom.element_coordinates
            return ec[:, 0], ec[:, 1]
        except Exception:
            nc = geom.node_coordinates
            return nc[:, 0], nc[:, 1]
    return None, None


def _build_spectral_geom(
    spec: "SpectraArray",
    io_geom: Any,
) -> Any:
    """Build the output spectral geometry for :func:`to_mikeio`."""
    from mikeio.spatial._grid_geometry import Grid1D, Grid2D

    freq = spec._freq
    direction = spec._direction

    if io_geom is None:
        if spec.has_location and spec._x is not None:
            return _coords_to_line_geom(
                spec._x, spec._y or np.zeros_like(spec._x), freq, direction
            )
        return GeometryFMPointSpectrum(frequencies=freq, directions=direction)

    if isinstance(io_geom, (GeometryFMAreaSpectrum, GeometryFMLineSpectrum)):
        return _make_spectral_geom(io_geom, freq, direction)

    if isinstance(io_geom, (Grid1D, Grid2D)):
        return _grid_to_spectral_geom(io_geom, freq, direction)

    if isinstance(io_geom, GeometryFMPointSpectrum):
        return GeometryFMPointSpectrum(
            frequencies=freq,
            directions=direction,
            x=getattr(io_geom, "x", None),
            y=getattr(io_geom, "y", None),
        )

    return _make_spectral_geom(io_geom, freq, direction)


def _params_output_geom(spec: "SpectraArray", io_geom: Any) -> Any:
    """Build the non-spectral output geometry for :func:`to_params_dataset`."""
    from mikeio.spatial._grid_geometry import Grid1D, Grid2D

    if io_geom is None:
        return None
    if isinstance(io_geom, (GeometryFMAreaSpectrum, GeometryFMLineSpectrum)):
        return _make_params_geometry(io_geom)
    if isinstance(io_geom, (Grid1D, Grid2D)):
        return io_geom
    return None


def _make_spectral_geom(
    geom: Any,
    frequencies: np.ndarray | None,
    directions: np.ndarray | None,
) -> Any:
    """Rebuild a spectral FM geometry with new freq/dir, preserving topology."""
    if isinstance(geom, GeometryFMPointSpectrum):
        return GeometryFMPointSpectrum(frequencies=frequencies, directions=directions)
    if isinstance(geom, (GeometryFMAreaSpectrum, GeometryFMLineSpectrum)):
        cls = type(geom)
        return cls(
            node_coordinates=geom.node_coordinates,
            element_table=geom.element_table,
            codes=geom.codes,
            projection=geom.projection_string,
            dfsu_type=geom._type,
            frequencies=frequencies,
            directions=directions,
        )
    raise TypeError(
        f"Unsupported spectral geometry: {type(geom).__name__!r}. "
        "Expected GeometryFMPointSpectrum, GeometryFMAreaSpectrum, or "
        "GeometryFMLineSpectrum."
    )


def _make_params_geometry(geom: Any) -> Any:
    """Return the non-spectral FM2D geometry for wave-parameter DataArrays."""
    from mikecore.DfsuFile import DfsuFileType

    if isinstance(geom, GeometryFMAreaSpectrum):
        return mikeio.spatial.GeometryFM2D(
            node_coordinates=geom.node_coordinates,
            element_table=geom.element_table,
            codes=geom.codes,
            projection=geom.projection_string,
            dfsu_type=DfsuFileType.Dfsu2D,
        )
    if isinstance(geom, GeometryFMLineSpectrum):
        return mikeio.spatial.GeometryFM2D(
            node_coordinates=geom.node_coordinates,
            element_table=geom.element_table,
            codes=geom.codes,
            projection=geom.projection_string,
            dfsu_type=DfsuFileType.Dfsu1D,
        )
    return None


def _coords_to_line_geom(
    x: np.ndarray,
    y: np.ndarray,
    freq: np.ndarray | None,
    direction: np.ndarray | None,
) -> GeometryFMLineSpectrum:
    """Build a minimal GeometryFMLineSpectrum from coordinate arrays."""
    from mikecore.DfsuFile import DfsuFileType

    n = len(x)
    node_coords = np.column_stack([x, y, np.zeros(n)])
    element_table: list[list[int]] = (
        [[0]] if n < 2 else [[i, i + 1] for i in range(n - 1)]
    )
    codes = np.zeros(n, dtype=int)
    return GeometryFMLineSpectrum(
        node_coordinates=node_coords,
        element_table=element_table,
        codes=codes,
        projection="LONG/LAT",
        frequencies=freq,
        directions=direction,
        dfsu_type=DfsuFileType.DfsuSpectral1D,
    )


# Forward-reference for type annotations
from pywswat.core import SpectraArray  # noqa: E402
