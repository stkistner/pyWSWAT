![logo](images/pyWAT.svg)
![Python Versions](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-blue)
[![Full test](https://github.com/DHI/pywswat/actions/workflows/full_test.yml/badge.svg)](https://github.com/DHI/pywswat/actions/workflows/full_test.yml)
![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux-blue)

# pyWAT — Wave Spectral Analysis for MIKE

**pyWAT** is a Python library for working with wave energy spectra produced by DHI's MIKE suite. It wraps [mikeio](https://github.com/DHI/mikeio) spectral DataArrays in a clean, analysis-friendly API and adds spectral estimation from raw time series.


## Installation

```bash
🚧 pip install pywswat # doesn't work yet
```

## Features

- **Read** directional wave spectra from MIKE `.dfsu` spectral files (point, line, and area geometry)
- **Estimate** spectra from time series: FFT, Welch, and Maximum Entropy Method (MEM) directional
- **Compute** integrated wave parameters: Hm0, Tp, T01, T02, Tm10, MWD, PWD, DSD
- **Manipulate** spectra: frequency slicing, direction integration, location selection
- **Plot** 1-D frequency and 2-D polar spectra