"""Utils for mikeio specific code that could be useful for wave analysis"""

import mikeio
import numpy as np

__all__ = ["subset_area"]


def subset_area(
    dataset,
    x: slice | None = None,
    y: slice | None = None,
    # area: tuple[float, float, float, float] | None = None,
):
    """Subset a dataset to a given area.

    Parameters
    ----------
    dataset : mikeio.Dataset or DataArray
        The dataset to subset.
    x : slice, optional
        The x range to subset to, by default None (no subsetting).
    y : slice, optional
        The y range to subset to, by default None (no subsetting).

    Returns
    -------
    xarray.Dataset
        The subsetted dataset.
    """
    if x is None and y is None:
        return dataset
    if x is None:
        x = slice(None)
    if y is None:
        y = slice(None)

    if not hasattr(dataset, "geometry"):
        raise ValueError(
            "Dataset must have a geometry attribute for spatial subsetting."
        )

    if isinstance(dataset.geometry, mikeio.spatial._FM_geometry.GeometryFM2D):
        x0, y0, x1, y1 = x.start, y.start, x.stop, y.stop
        xyz = dataset.geometry.element_coordinates
        if x0 is None:
            x0 = np.min(xyz[:, 0])
        if x1 is None:
            x1 = np.max(xyz[:, 0])
        if y0 is None:
            y0 = np.min(xyz[:, 1])
        if y1 is None:
            y1 = np.max(xyz[:, 1])

        msk = np.where(
            (xyz[:, 0] >= x0)
            & (xyz[:, 0] <= x1)
            & (xyz[:, 1] >= y0)
            & (xyz[:, 1] <= y1)
        )[0]
        if len(msk) == 0:
            raise ValueError("No elements found in the specified area.")
        return dataset.isel(element=msk)

    elif isinstance(dataset.geometry, mikeio.spatial._grid_geometry.Grid2D):

        def squeeze(x):
            if x is None:
                return None
            else:
                return int(np.squeeze(x))

        x0, y0, x1, y1, dx, dy = x.start, y.start, x.stop, y.stop, x.step, y.step
        i0, i1, j0, j1, di, dj = None, None, None, None, None, None
        if x0 is not None:
            i0 = squeeze(dataset.geometry.find_index(x=x0)[0])
        if x1 is not None:
            i1 = squeeze(dataset.geometry.find_index(x=x1)[0])
        if y0 is not None:
            j0 = squeeze(dataset.geometry.find_index(y=y0)[1])
        if y1 is not None:
            j1 = squeeze(dataset.geometry.find_index(y=y1)[1])

        if dx is not None:
            di = int(dx // (np.median(np.diff(dataset.geometry.x))))
            if di == 0:
                di = None
        if dy is not None:
            dj = int(dy // (np.median(np.diff(dataset.geometry.y))))
            if dj == 0:
                dj = None

        return dataset.isel(x=slice(i0, i1, di), y=slice(j0, j1, dj))
