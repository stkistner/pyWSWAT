"""Core FIR algorithms."""

from __future__ import annotations

from warnings import warn

import mikeio
import numpy as np
from scipy.signal import firwin, lfilter


def fir_filter(
    data: np.ndarray,
    time_ax: np.ndarray,
    window_size: int = 1025,
    fmax: float | None = None,
    fmin: float | None = None,
    window: str = "hamming",
):
    """
    Generates and applies Finite Imuplse Response filter on a timeseries

    Use for low pass (fmax), high pass (fmin), band pass (fmin < fmax) or band stop band pass (fmax < fmin)
    Uses scipy.signal lfilter & firwin

    Parameters
    ----------
    data : pd.Series, xr.DataArray or mikeio.DataArray
        N-dimension datastructure with time axis.
        Series assumes Index is time.
        Datarray needs to have a time dimension (see 'index_dim')
    window_size : int
        Width of FIR. Recommended to use a odd number
    fmin : float
        Minimum frequency (highpass filter).
        If None (default), no lowpass will be applied
    fmax : float
        frequencies below (lowpass filter)
        If None (default), no highpass will be applied
    window : str of scipy.signal.windows
        Default: 'hann'
        See scipy.signal.get_window for a list of windows and required parameters.

    Returns
    -------
    result : filtered dataseries
    """

    if window_size % 2 == 0:
        warn("N-width is even. Timeaxis will be offset by half a timestep")

    # Get time axis
    if isinstance(ds, pd.Series):
        time_ax = ds.index
    elif isinstance(ds, xr.DataArray):
        if index_dim not in ds.dims:
            raise ValueError(f"Invalid index_dim: {index_dim} not in xr.DataArray dims")
        time_ax = ds[index_dim].values
    elif isinstance(ds, mikeio.dataset._dataset.DataArray):
        time_ax = ds.time
    else:
        raise ValueError("Invalid input. Must be pd.Series or xr.DataArray")
    time_ax = pd.to_datetime(time_ax)

    if N >= len(time_ax):
        raise ValueError(
            f"Invalid N: N must be less than the length of the time axis. N={N}, len(time_ax)={len(time_ax)}"
        )

    # FIR
    sample_rate = 1 / _get_T(time_ax)
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0
    # The cutoff frequency of the filter.

    # QC
    if fmax is not None:
        if (fmax > nyq_rate) | (fmax <= 0):
            raise ValueError(
                f"Invalid cutoff frequency: frequencies must be greater than 0 and less than Nyquist freq={nyq_rate}"
            )
    if fmin is not None:
        if (fmin > nyq_rate) | (fmin <= 0):
            raise ValueError(
                f"Invalid cutoff frequency: frequencies must be greater than 0 and less than Nyquist freq={nyq_rate}"
            )

    if fmax is None and fmin is None:
        warn("No frequencies passed")
        return ds

    # Low pass
    if fmax is not None and fmin is None:
        f_ = fmax
        pass_zero = True
    # High pass
    if fmax is None and fmin is not None:
        f_ = fmin
        pass_zero = False
    # Bandpass, Bandstop
    if fmax is not None and fmin is not None:
        # Bandpass
        if fmax > fmin:
            f_ = np.array([fmin, fmax])
            pass_zero = False
        # Bandstop
        else:
            f_ = np.array([fmax, fmin])
            pass_zero = True

    # Use firwin to create a lowpass FIR filter.
    taps = firwin(N, f_ / nyq_rate, window=window, pass_zero=pass_zero)

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate

    # Apply the filter to the signal.
    if isinstance(ds, pd.Series):
        nd_fir = lfilter(taps, 1, ds.values)
        # Create a new time axis for the filtered signal.
        ds_fir = pd.Series(nd_fir, index=time_ax - pd.Timedelta(delay, unit="s"))
        # Truncate rubish
        ds_fir = ds_fir[N - 1 :]
    elif isinstance(ds, xr.DataArray):
        # Ax no.
        ax_no = ds.dims.index(index_dim)
        # Apply
        nd_fir = np.apply_along_axis(
            lambda x: lfilter(taps, 1, x), axis=ax_no, arr=ds.values
        )
        # Create time ax
        ds_fir = xr.DataArray(nd_fir, coords=ds.coords, name=ds.name).assign_coords(
            {"time": time_ax - pd.Timedelta(delay, unit="s")}
        )
        # Truncate rubish
        ds_fir = ds_fir.isel({index_dim: slice(N - 1, None)})

    elif isinstance(ds, mikeio.dataset._dataset.DataArray):
        # Ax no.
        ax_no = ds.dims.index(index_dim)
        # Apply
        nd_fir = np.apply_along_axis(
            lambda x: lfilter(taps, 1, x), axis=ax_no, arr=ds.values
        )
        # Create time ax
        ds_fir = mikeio.DataArray(
            nd_fir, item=ds.item, geometry=ds.geometry, time=time_ax
        )
        ds_fir.time = time_ax - pd.Timedelta(delay, unit="s")
        # Truncate rubish
        ds_fir = ds_fir.isel(time=slice(N - 1, None))

    return ds_fir
