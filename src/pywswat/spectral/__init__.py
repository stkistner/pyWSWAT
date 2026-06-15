"""Spectral analysis algorithms — FFT, Welch, and MEM estimators."""

from pywswat.spectral._spectral import spectrum, spectrum_welch, spectrum_mem

__all__ = ["spectrum", "spectrum_welch", "spectrum_mem"]
