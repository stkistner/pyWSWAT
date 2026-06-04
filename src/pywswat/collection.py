from __future__ import annotations

import numpy as np
import mikeio

from pywswat.wave_spectrum import SpectraArray


class SpectraSet:
    """A collection of spectral DataArrays, wrapping a :class:`mikeio.Dataset`.

    All items must share the same spectral geometry (frequencies and
    directions).  Provides batch spectral analysis and a Dataset-level
    :meth:`to_params`.

    Parameters
    ----------
    ds : mikeio.Dataset
        A Dataset whose items all carry spectral geometry
        (``GeometryFMPointSpectrum``, ``GeometryFMAreaSpectrum``, or
        ``GeometryFMLineSpectrum``).

    Raises
    ------
    ValueError
        If *ds* is empty, or if items differ in spectral axis lengths.

    Examples
    --------
    >>> import mikeio
    >>> ds = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")
    >>> spec_set = SpectraSet(ds)
    >>> spec_set.n_items
    1
    >>> isinstance(spec_set[0], SpectraArray)
    True
    """

    def __init__(self, ds: mikeio.Dataset) -> None:
        self._ds = ds
        self._arrays: dict[str, SpectraArray] = {
            da.name: SpectraArray(da) for da in ds
        }
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if not self._arrays:
            raise ValueError("'ds' must contain at least one spectral item.")
        first_name, first = next(iter(self._arrays.items()))
        for name, spec in self._arrays.items():
            if name == first_name:
                continue
            if spec.nf != first.nf or spec.nd != first.nd:
                raise ValueError(
                    f"Item {name!r} has different spectral axis lengths than "
                    f"{first_name!r}."
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ds(self) -> mikeio.Dataset:
        """The underlying mikeio Dataset."""
        return self._ds

    @property
    def names(self) -> list[str]:
        """Names of all spectral items."""
        return list(self._arrays.keys())

    @property
    def n_items(self) -> int:
        """Number of spectral items."""
        return len(self._arrays)

    @property
    def freq(self) -> np.ndarray | None:
        """Shared frequency axis (Hz)."""
        return next(iter(self._arrays.values())).freq

    @property
    def direction(self) -> np.ndarray | None:
        """Shared direction axis (degrees)."""
        return next(iter(self._arrays.values())).direction

    @property
    def time(self) -> np.ndarray | None:
        """Shared time axis."""
        return next(iter(self._arrays.values())).time

    @property
    def has_freq(self) -> bool:
        return next(iter(self._arrays.values())).has_freq

    @property
    def has_dir(self) -> bool:
        return next(iter(self._arrays.values())).has_dir

    @property
    def has_time(self) -> bool:
        return next(iter(self._arrays.values())).has_time

    @property
    def has_location(self) -> bool:
        return next(iter(self._arrays.values())).has_location

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a single item's logical data (time?, location?, freq?, dir?)."""
        return next(iter(self._arrays.values())).shape

    # ------------------------------------------------------------------
    # Indexing and iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_items

    def __iter__(self):  # type: ignore[return]
        return iter(self._arrays.values())

    def __getitem__(
        self, key: str | int | list | slice
    ) -> SpectraArray | "SpectraSet":
        """Index into the collection.

        Parameters
        ----------
        key : str, int, list of str/int, or slice
            A ``str`` or ``int`` returns the matching :class:`SpectraArray`.
            A ``list`` or ``slice`` returns a new :class:`SpectraSet`.

        Examples
        --------
        >>> import mikeio
        >>> ds = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")
        >>> ss = SpectraSet(ds)
        >>> isinstance(ss[0], SpectraArray)
        True
        """
        names = self.names
        if isinstance(key, str):
            return self._arrays[key]
        if isinstance(key, (int, np.integer)):
            return self._arrays[names[int(key)]]
        if isinstance(key, slice):
            selected = names[key]
            return SpectraSet(mikeio.Dataset([self._ds[n] for n in selected]))
        if isinstance(key, list):
            resolved: list[str] = []
            for k in key:
                if isinstance(k, str):
                    resolved.append(k)
                elif isinstance(k, (int, np.integer)):
                    resolved.append(names[int(k)])
                else:
                    raise TypeError(
                        f"List index must be str or int, got {type(k).__name__!r}."
                    )
            return SpectraSet(mikeio.Dataset([self._ds[n] for n in resolved]))
        raise TypeError(f"Invalid index type: {type(key).__name__!r}.")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def to_params(self) -> mikeio.Dataset:
        """Compute integrated wave parameters for all items.

        Returns a single :class:`mikeio.Dataset`.  For a single-item
        ``SpectraSet`` parameter names are unchanged (e.g. ``"Hm0"``).
        For multi-item sets they are prefixed with the item name
        (e.g. ``"wind_sea_Hm0"``, ``"swell_Hm0"``).

        Returns
        -------
        mikeio.Dataset

        Examples
        --------
        >>> import mikeio
        >>> ds = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")
        >>> ss = SpectraSet(ds)
        >>> params = ss.to_params()
        >>> isinstance(params, mikeio.Dataset)
        True
        >>> "Hm0" in params.names
        True
        """
        if self.n_items == 1:
            return next(iter(self._arrays.values())).to_params()

        all_das: list[mikeio.DataArray] = []
        for item_name, spec in self._arrays.items():
            item_ds = spec.to_params()
            for param_name in item_ds.names:
                src = item_ds[param_name]
                all_das.append(
                    mikeio.DataArray(
                        data=src.values,
                        time=src.time,
                        type=src.type,
                        unit=src.unit,
                        name=f"{item_name}_{param_name}",
                    )
                )
        return mikeio.Dataset(all_das)

    def integrate_dir(self) -> "SpectraSet":
        """Integrate over the direction axis for all items.

        Returns
        -------
        SpectraSet
            New set without the direction axis.

        Raises
        ------
        ValueError
            If no direction axis is present.

        Examples
        --------
        >>> import mikeio
        >>> ds = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")
        >>> ss = SpectraSet(ds)
        >>> ss_f = ss.integrate_dir()
        >>> ss_f.has_dir
        False
        """
        new_das = [spec.integrate_dir().da for spec in self._arrays.values()]
        return SpectraSet(mikeio.Dataset(new_das))

    def sel_freq(
        self, fmin: float = 0.0, fmax: float = float("inf")
    ) -> "SpectraSet":
        """Slice the frequency axis for all items.

        Parameters
        ----------
        fmin : float
            Lower frequency bound (inclusive), in Hz.
        fmax : float
            Upper frequency bound (inclusive), in Hz.

        Returns
        -------
        SpectraSet
            New set with the sliced frequency axis.

        Examples
        --------
        >>> import mikeio
        >>> ds = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")
        >>> ss = SpectraSet(ds)
        >>> ss2 = ss.sel_freq(fmin=0.1, fmax=0.3)
        >>> bool(ss2.freq[0] >= 0.1)
        True
        """
        new_das = [spec.sel_freq(fmin, fmax).da for spec in self._arrays.values()]
        return SpectraSet(mikeio.Dataset(new_das))

    def isel(self, location: int) -> "SpectraSet":
        """Select a single location by index for all items.

        Parameters
        ----------
        location : int
            Index into the location (element or node) axis.

        Returns
        -------
        SpectraSet
            New set without the location axis.

        Examples
        --------
        >>> import mikeio
        >>> ds = mikeio.read("tests/testdata/spectra/area_spectra.dfsu")
        >>> ss = SpectraSet(ds)
        >>> ss_pt = ss.isel(location=5)
        >>> ss_pt.has_location
        False
        """
        new_das = [spec.isel(location).da for spec in self._arrays.values()]
        return SpectraSet(mikeio.Dataset(new_das))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = ["<pywswat.SpectraSet>"]

        first = next(iter(self._arrays.values()))
        da = first.da
        dims_str = ", ".join(f"{d}:{s}" for d, s in zip(da.dims, da.values.shape))
        lines.append(f"dims: ({dims_str})")

        if self.has_freq and self.freq is not None:
            lines.append(f"frequency: {self.freq[0]:.3f} - {self.freq[-1]:.3f} Hz")
        if self.has_dir and self.direction is not None:
            lines.append(
                f"direction: {self.direction[0]:.0f} - {self.direction[-1]:.0f} deg"
            )
        if self.has_time and self.time is not None:
            t = self.time
            lines.append(f"time: {t[0]} - {t[-1]}  ({len(t)} steps)")

        if self.n_items > 10:
            lines.append(f"items: {self.n_items} items")
        else:
            lines.append("items:")
            for i, name in enumerate(self.names):
                lines.append(f"  {i}: {name}")

        return "\n".join(lines)

