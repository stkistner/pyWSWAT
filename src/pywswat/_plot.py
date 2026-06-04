from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pywswat.wave_spectrum import SpectraArray


class SpectraPlotter:
    """Plotting namespace attached to a :class:`~pywswat.SpectraArray` as ``spec.plot``.

    Parameters
    ----------
    spectra : SpectraArray
        The parent SpectraArray object.
    """

    def __init__(self, spectra: "SpectraArray") -> None:
        self._s = spectra

    def __call__(self, **kwargs: Any) -> Any:
        """Default plot — dispatches to timeseries() or spectrum() based on axes.

        Dispatches to ``timeseries()`` when the spectrum has both a time and a
        frequency axis, otherwise dispatches to ``spectrum()``.

        Parameters
        ----------
        **kwargs
            Forwarded to the dispatched method.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
        """
        if self._s.has_time and self._s.has_freq:
            return self.timeseries(**kwargs)
        return self.spectrum(**kwargs)

    def spectrum(
        self,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        time_idx: int = 0,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        log_colors: bool = True,
        period_ticks: list[float] | None = None,
        colorbar: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot the wave spectrum.

        For a **2D** spectrum (frequency + direction) a polar plot is drawn
        with direction as the angular axis and period (= 1/frequency) as the
        radial axis — short periods at the centre, long periods at the outer
        ring.  North is at the top and angles increase clockwise (nautical
        convention).

        For a **1D** frequency spectrum a simple line plot is drawn.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  For a 2D spectrum the axes **must** use
            ``projection="polar"``; a new polar figure is created when *None*.
            For a 1D spectrum any regular axes is accepted.
        figsize : tuple of float, optional
            Figure size used when a new figure is created.
        title : str, optional
            Axes title.  Defaults to ``"Wave Spectrum"``.
        time_idx : int, optional
            Time-step index to plot when the spectrum has a time axis
            (default ``0``).
        cmap : str, optional
            Colormap for the 2D polar plot (default ``"RdBu_r"``).
        vmin : float, optional
            Lower colour-scale limit.  Derived automatically when *None*.
        vmax : float, optional
            Upper colour-scale limit.  Defaults to the data maximum.
        log_colors : bool, optional
            Use logarithmic colour normalisation (default ``True``).
        period_ticks : list of float, optional
            Explicit period-axis tick values in seconds for the 2D polar plot.
            Derived automatically when *None*.
        colorbar : bool, optional
            Add a colorbar to the 2D polar plot (default ``True``).
        **kwargs
            Extra keyword arguments forwarded to ``ax.contourf`` (2D) or
            ``ax.plot`` (1D).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if not self._s.has_freq:
            raise ValueError("spectrum() requires a frequency axis.")

        if self._s.has_location:
            raise ValueError(
                "spectrum() does not support spectra with a location axis. "
                "Use .isel(location=i) to select a single point first."
            )

        # Select time step from the underlying DA (always has time as axis 0)
        da_vals = self._s.da.values  # mikeio order: (time, [dir,] freq)
        t_idx = time_idx if self._s.has_time else 0
        slice_2d = da_vals[t_idx]    # (nd, nf) for 2D or (nf,) for 1D

        time_note = ""
        if self._s.has_time:
            t = self._s.time
            time_note = f" ({t[t_idx]})" if t is not None else f" (t={t_idx})"

        # ── 1D frequency spectrum ─────────────────────────────────────────────
        if not self._s.has_dir:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.plot(self._s.freq, slice_2d, **kwargs)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Energy density (m² Hz⁻¹)")
            ax.set_title((title or "Wave Spectrum") + time_note)
            return ax

        # ── 2D polar spectrum ─────────────────────────────────────────────────
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import LogFormatterSciNotation, LogLocator

        freqs = self._s.freq       # (nf,) Hz
        assert freqs is not None
        dirs = self._s.direction   # (nd,) degrees
        assert dirs is not None

        # slice_2d: mikeio order (nd, nf)
        Z = slice_2d               # (nd, nf)

        # Close the directional loop so the polar plot is continuous at 360°
        dirs_ext = np.append(dirs, dirs[0] + 360.0)     # (nd+1,)
        theta = np.deg2rad(dirs_ext)                      # (nd+1,) radians
        Z_ext = np.vstack([Z, Z[0:1]])                   # (nd+1, nf)

        # Period axis: sort ascending so short periods are at centre
        periods = 1.0 / freqs                             # (nf,)
        sort_idx = np.argsort(periods)
        periods_asc = periods[sort_idx]                   # (nf,) ascending
        Z_sorted = Z_ext[:, sort_idx]                     # (nd+1, nf) reordered

        # Colour limits
        vmax_use = float(np.nanmax(Z_sorted)) if vmax is None else vmax
        if vmax_use <= 0:
            vmax_use = 1.0
        vmin_use = (
            min(1e-4, float(np.exp(np.log(vmax_use) - 4)))
            if vmin is None
            else vmin
        )
        if vmax_use <= vmin_use:
            vmin_use = vmax_use / 8.0

        # Clip upper tail so it still renders with extend="max"
        Z_plot = np.where(Z_sorted > vmax_use, vmax_use, Z_sorted)

        if log_colors:
            norm = LogNorm(vmin=vmin_use, vmax=vmax_use)
            levels = np.geomspace(vmin_use, vmax_use, 64)
        else:
            norm = None
            levels = np.linspace(vmin_use, vmax_use, 64)

        # Create polar axes if not supplied
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection="polar")

        # contourf(theta, r, Z) where Z must be (n_r, n_theta)
        cs = ax.contourf(
            theta,
            periods_asc,
            Z_plot.T,          # (nf, nd+1)
            levels=levels,
            norm=norm,
            cmap=cmap,
            extend="max",
            **kwargs,
        )

        # ── Direction axis: nautical convention (N up, clockwise) ────────────
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        # ── Period (radial) axis: log scale, period labels ───────────────────
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0f}s")
        )
        if period_ticks is not None:
            ax.set_yticks(period_ticks)
        ax.set_ylim(periods_asc[0], periods_asc[-1])

        ax.set_title((title or "Wave Spectrum") + time_note)

        if colorbar:
            cb_kwargs: dict = dict(ax=ax, pad=0.12,
                                   label="Energy density (m² Hz⁻¹ rad⁻¹)")
            if log_colors:
                cb_kwargs.update(ticks=LogLocator(),
                                 format=LogFormatterSciNotation())
            plt.colorbar(cs, **cb_kwargs)

        return ax

    def timeseries(
        self,
        params: list[str] | None = None,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot spectral parameters as time series.

        One subplot per requested parameter, stacked vertically, sharing the
        x-axis.

        Parameters
        ----------
        params : list of str, optional
            Parameter names to plot (must be keys of ``to_params()``).
            Defaults to ``["Hm0"]``.
        ax : matplotlib.axes.Axes or array-like of Axes, optional
            Pre-existing axes.  When *None* a new figure is created.
        figsize : tuple of float, optional
            Figure size passed to ``plt.subplots`` when creating a new figure.
        **kwargs
            Extra keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
            Single Axes when only one parameter is requested, otherwise a
            1-D array of Axes.
        """
        import matplotlib.pyplot as plt

        if not self._s.has_time:
            raise ValueError(
                "timeseries() requires a time axis (has_time is False)."
            )
        if not self._s.has_freq:
            raise ValueError(
                "timeseries() requires a frequency axis (has_freq is False)."
            )

        if params is None:
            params = ["Hm0"]

        param_data = self._s.to_params()
        for p in params:
            if p not in param_data.names:
                raise ValueError(
                    f"Parameter '{p}' not found in to_params() output. "
                    f"Available: {param_data.names}"
                )

        if self._s.time is not None:
            time_values = self._s.time
        else:
            assert self._s.nt is not None
            time_values = np.arange(self._s.nt)

        n = len(params)
        if ax is None:
            fig, axes = plt.subplots(
                n, 1, figsize=figsize, sharex=True, squeeze=False
            )
            axes = axes[:, 0]
        else:
            axes = np.atleast_1d(ax)

        for i, p in enumerate(params):
            axes[i].plot(time_values, param_data[p].values, **kwargs)
            axes[i].set_ylabel(p)

        axes[-1].set_xlabel("Time")

        if n == 1:
            return axes[0]
        return axes

    def hist(
        self,
        param: str = "Hm0",
        bins: int = 30,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot a histogram of a spectral parameter over time.

        Parameters
        ----------
        param : str, optional
            Parameter name (key of ``to_params()``).  Defaults to ``"Hm0"``.
        bins : int, optional
            Number of histogram bins.  Defaults to 30.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created when *None*.
        figsize : tuple of float, optional
            Figure size passed to ``plt.figure`` when creating a new figure.
        **kwargs
            Extra keyword arguments forwarded to ``ax.hist``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if not self._s.has_freq:
            raise ValueError(
                "hist() requires a frequency axis (has_freq is False)."
            )
        if not self._s.has_time:
            raise ValueError(
                "hist() requires a time axis (has_time is False)."
            )

        param_data = self._s.to_params()
        if param not in param_data.names:
            raise ValueError(
                f"Parameter '{param}' not found in to_params() output. "
                f"Available: {param_data.names}"
            )

        values = param_data[param].values

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.hist(values, bins=bins, **kwargs)
        ax.set_xlabel(param)
        ax.set_ylabel("Count")
        return ax
