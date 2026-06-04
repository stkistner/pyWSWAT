from __future__ import annotations

from pathlib import Path

import mikeio

from pywswat.wave_spectrum import SpectraArray
from pywswat.collection import SpectraSet


def read(
    path: str | Path,
    item: int | str | list[int | str] | None = None,
) -> SpectraArray | SpectraSet:
    """Read a MIKEIO spectral file into a :class:`SpectraArray` or :class:`SpectraSet`.

    Supports all mikeio spectral geometry types:

    * ``.dfsu`` with ``GeometryFMPointSpectrum``  → single-location spectra
    * ``.dfsu`` with ``GeometryFMLineSpectrum``   → line spectra (node axis)
    * ``.dfsu`` with ``GeometryFMAreaSpectrum``   → area spectra (element axis)

    When *item* is a single index or name a :class:`SpectraArray` is returned.
    When *item* is omitted or a list a :class:`SpectraSet` is returned.

    Parameters
    ----------
    path : str or Path
        Path to the spectral file (e.g. ``.dfsu``).
    item : int, str, list of int/str, or None
        Item(s) to read.  A single value returns a :class:`SpectraArray`;
        ``None`` (default) or a list returns a :class:`SpectraSet`.

    Returns
    -------
    SpectraArray or SpectraSet

    Examples
    --------
    >>> spec = read("tests/testdata/spectra/pt_spectra.dfsu")
    >>> isinstance(spec, SpectraSet)
    True
    >>> spec_da = read("tests/testdata/spectra/pt_spectra.dfsu", item=0)
    >>> isinstance(spec_da, SpectraArray)
    True
    """
    path = Path(path)
    ds = mikeio.read(str(path))

    if item is None or isinstance(item, list):
        if item is not None:
            ds = mikeio.Dataset([ds[i] for i in item])
        return SpectraSet(ds)

    return SpectraArray(ds[item])
