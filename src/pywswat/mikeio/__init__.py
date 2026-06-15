"""I/O infrastructure — mikeio adapters and file reader."""

from pywswat.mikeio._reader import read
from pywswat.mikeio._adapters import from_mikeio, to_mikeio, to_params_dataset
from pywswat.mikeio._workflows import spectral_params, spectrum
from pywswat.mikeio._utils import subset_area

__all__ = [
    "read",
    "from_mikeio",
    "to_mikeio",
    "to_params_dataset",
    "spectral_params",
    "subset_area",
    "spectrum",
]
