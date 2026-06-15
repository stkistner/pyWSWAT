from ._adapters import to_params_dataset, to_mikeio
from pywswat.spectral._spectral import spectrum_mem


def spectral_params(
    ds_2d,
    elevation=None,
    u=None,
    v=None,
    window="hann",
    overlap=0.5,
    window_size=256,
    fmin=None,
    fmax=None,
):
    """Calculate spectral parameters from 2D time series.

    Parameters
    ----------
    ds_2d : xarray.Dataset
        Dataset containing the 2D time series data.
    elevation : str, optional
        Name of the variable in `ds_2d` representing elevation data.
    u : str, optional
        Name of the variable in `ds_2d` representing u-velocity data.
    v : str, optional
        Name of the variable in `ds_2d` representing v-velocity data.
    window : str or tuple or array_like, optional
        Desired window to use. See `scipy.signal.get_window` for details.
    overlap : float, optional
        Fraction of overlap between windows (0 to 1).
    window_size : int, optional
        Length of each window.
    fmin : float, optional
        Minimum frequency to include in the output dataset.
    fmax : float, optional
        Maximum frequency to include in the output dataset.

    Returns
    -------
    xarray.Dataset
        Dataset containing the calculated spectral parameters.
    """
    data_mem = spectrum_mem(
        ds_2d,
        elevation=elevation,
        u=u,
        v=v,
        window=window,
        overlap=overlap,
        window_size=window_size,
    )

    return to_params_dataset(
        data_mem.sel_freq(fmin=fmin, fmax=fmax),
    )


def spectrum(
    ds_2d,
    elevation=None,
    u=None,
    v=None,
    window="hann",
    overlap=0.5,
    window_size=256,
    fmin=None,
    fmax=None,
):
    """Calculate spectral parameters from 2D time series.

    Parameters
    ----------
    ds_2d : xarray.Dataset
        Dataset containing the 2D time series data.
    elevation : str, optional
        Name of the variable in `ds_2d` representing elevation data.
    u : str, optional
        Name of the variable in `ds_2d` representing u-velocity data.
    v : str, optional
        Name of the variable in `ds_2d` representing v-velocity data.
    window : str or tuple or array_like, optional
        Desired window to use. See `scipy.signal.get_window` for details.
    overlap : float, optional
        Fraction of overlap between windows (0 to 1).
    window_size : int, optional
        Length of each window.
    fmin : float, optional
        Minimum frequency to include in the output dataset.
    fmax : float, optional
        Maximum frequency to include in the output dataset.

    Returns
    -------
    xarray.Dataset
        Dataset containing the calculated spectral parameters.
    """
    data_mem = spectrum_mem(
        ds_2d,
        elevation=elevation,
        u=u,
        v=v,
        window=window,
        overlap=overlap,
        window_size=window_size,
    )

    return to_mikeio(
        data_mem.sel_freq(fmin=fmin, fmax=fmax),
    )
