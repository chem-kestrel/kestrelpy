"""MIT License

KestrelPy
Copyright (c) 2026
George Nunn, Benjamin S. Grace, Martin A. Bates and Paul H. Walton.
Department of Chemistry, University of York, United Kingdom.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS  OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN  AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION  WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from pathlib import Path
from typing import Literal

import numpy as np

from kestrelpy.calculations.intensities.spectroscopy import Spectrum
from kestrelpy.workflows.simulate_spectra import (
    convolute_spectra_from_files as _convolute_spectra_from_files,
)
from kestrelpy.workflows.simulate_spectra import (
    convolute_spectrum as _convolute_spectrum,
)
from kestrelpy.workflows.simulate_spectra import (
    convolute_spectrum_from_file as _convolute_spectrum_from_file,
)


def convolute_spectrum(
    spectrum: Spectrum,
    fwhm: float = 1000.0,
    start: float = 5000.0,
    final: float = 30000.0,
    steps: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the convoluted spectra from a Spectrum object.

    Args:
        spectrum: The Spectrum object.
        fwhm: The full-width half-maximum (FWHM) in wavenumbers (cm-1).
        start: The starting value of the x-axis, in wavenumbers (cm-1).
        final: The final value of the x-axis, in wavenumbers (cm-1).
        steps: The number of interpolating steps between 'start' and 'final'.

    Example:
        For a typical kestrelpy calculation:
        ```python
        import kestrelpy as kp


        results = kp.run_calculation(
            ...,
            intensities=kp.IntensityBlock(
                ...,
                absorption=kp.AbsorptionBlock(...),
            )
        )
        x, ys = kp.convolute_spectrum(
            results[0].intensities[0].absorption,
            ...,
        )

        # Plot the first spectrum
        plt.plot(x, ys[0, :])
        ```
    """
    return _convolute_spectrum(
        spectrum=spectrum,
        fwhm=fwhm,
        start=start,
        final=final,
        steps=steps,
    )


def convolute_spectrum_from_file(
    file_path: str | Path,
    calculation_type: Literal["absorption", "cd", "mcd"],
    fwhm: float = 1000.0,
    start: float = 5000.0,
    final: float = 30000.0,
    steps: int = 250,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Generate the convoluted spectra from a spectra json file.

    Args:
        file_path: The file to load the json from.
        calculation_type: The calculation-type to convolute. One of:
            'absorption', 'cd', or 'mcd'.
        fwhm: The full-width half-maximum (FWHM) in wavenumbers (cm-1).
        start: The starting value of the x-axis, in wavenumbers (cm-1).
        final: The final value of the x-axis, in wavenumbers (cm-1).
        steps: The number of interpolating steps between 'start' and 'final'.

    Returns:
        label: "T=250.0K, B=0.0T".
        np.ndarray: The x values.
        np.ndarray: The y values.

    Example:
        For a typical kestrelpy calculation:
        ```python
        import kestrelpy as kp


        results = kp.run_calculation(
            ...,
            intensities=kp.IntensityBlock(
                ...,
                absorption=kp.AbsorptionBlock(...),
            ),
            ...,
            printing=kp.PrintingBlock(file_path="molecule1.out", write_spectra=True),
        )
        for (label, x, y) in kp.convolute_spectrum_from_file(
            "molecule1_run001_intensity001_spectra.json"
            "absorption",
            ...,
        ):
            plt.plot(x, y)
        ```
    """
    return _convolute_spectrum_from_file(
        file_path=file_path,
        calculation_type=calculation_type,
        fwhm=fwhm,
        start=start,
        final=final,
        steps=steps,
    )


def convolute_spectra_from_files(
    input: str | Path,
    output_directory: str | Path,
    fwhm: float = 1000.0,
    start: float = 5000.0,
    final: float = 30000.0,
    steps: int = 250,
) -> None:
    """Generate the convoluted spectra from intensity spectra json.

    Args:
        input: The input file name or pattern to match.
        output_directory: Where to deposit the generated spectra CSVs.
        fwhm: The full-width half-maximum (FWHM) in wavenumbers (cm-1).
        start: The starting value of the x-axis, in wavenumbers (cm-1).
        final: The final value of the x-axis, in wavenumbers (cm-1).
        steps: The number of interpolating steps between 'start' and 'final'.

    Example:
        For a directory with loads of "<name>_run<num>_intensity<num>_spectra.json"
        you can automatically generate the spectra as CSV using:
        ```python
        import kestrelpy as kp
        kp.convolute_spectra_from_files(
            input="CoCl4_run*_intensity*_spectra.json",
            output_directory=".",
        )
        ```
        This will find all files on the system's path which has the structure given
        by inpute, where the '*' after 'run' and 'intensity' will get every parameter
        run and intensity run. `output_directory="."` will write the files to the
        current working directory.

        Or you can target a specific file by doing:
        `input="CoCl4_run001_intensity001_spectra.json"`
    """
    _convolute_spectra_from_files(
        input=input,
        output_directory=output_directory,
        fwhm=fwhm,
        start=start,
        final=final,
        steps=steps,
    )
