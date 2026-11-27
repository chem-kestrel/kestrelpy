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

import glob
import json
from pathlib import Path
from typing import Literal

import numpy as np

from kestrelpy.calculations.intensities.spectroscopy import (
    SpectroscopicTransition,
    Spectrum,
)
from kestrelpy.calculations.intensities.spectroscopy import (
    convolute_spectrum as _convolute_spectrum,
)
from kestrelpy.formatting.csv_formatting.spectra import format_csv_for_spectra
from kestrelpy.output import write_csv

CALCULATION_TYPES = ("absorption", "cd", "mcd")


def convolute_spectrum(
    spectrum: Spectrum,
    fwhm: float = 1000.0,
    start: float = 10000.0,
    final: float = 30000.0,
    steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.linspace(start, final, steps, endpoint=True)
    return (
        x_values,
        _convolute_spectrum(
            spectrum=spectrum,
            energy_range=x_values,
            fwhm=fwhm,
            summed=True,
        ),
    )


def convolute_spectrum_from_file(
    file_path: str | Path,
    calculation_type: Literal["absorption", "cd", "mcd"],
    fwhm: float = 1000.0,
    start: float = 10000.0,
    final: float = 30000.0,
    steps: int = 200,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if calculation_type not in CALCULATION_TYPES:
        raise ValueError(
            f"Invalid calculation_type encountered: {calculation_type}. "
            f"Please choose one of: {CALCULATION_TYPES}"
        )

    json_file = Path(file_path)
    with open(json_file, "r") as f:
        data = json.load(f)

    output: list[tuple[str, np.ndarray, np.ndarray]] = []
    for block in data[calculation_type]:
        spectrum = Spectrum(
            temperature=block["temperature"],
            magnetic_field=block["magnetic_field"],
            transitions=[
                SpectroscopicTransition(
                    energy=row["energy"],
                    intensity=row["intensity"],
                )
                for row in block["transitions"]
            ],
        )
        output.append(
            (
                f"T={spectrum.temperature}K, B={spectrum.magnetic_field}T",
                *convolute_spectrum(
                    spectrum=spectrum,
                    fwhm=fwhm,
                    start=start,
                    final=final,
                    steps=steps,
                ),
            )
        )
    return output


def convolute_spectra_from_files(
    input: str | Path,
    output_directory: str | Path,
    fwhm: float = 1000.0,
    start: float = 10000.0,
    final: float = 30000.0,
    steps: int = 200,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(input)))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {input}")

    energy_range = np.linspace(start, final, steps, endpoint=True)

    for json_file in files:
        json_file_path = Path(json_file)

        with open(json_file_path, "r") as f:
            data = json.load(f)

        calculation_id = data["calculation_id"]
        intensity_id = data["intensity_id"]

        for observable in ("absorption", "cd", "mcd"):
            headers = ["Energy (cm-1)"]
            spectra = []
            for block in data[observable]:
                T = block["temperature"]
                B = block["magnetic_field"]
                headers.append(f"T={T}K, B={B}T")
                spectrum = Spectrum(
                    temperature=T,
                    magnetic_field=B,
                    transitions=[
                        SpectroscopicTransition(
                            energy=row["energy"],
                            intensity=row["intensity"],
                        )
                        for row in block["transitions"]
                    ],
                )
                conv = _convolute_spectrum(
                    spectrum=spectrum,
                    energy_range=energy_range,
                    fwhm=fwhm,
                    summed=True,
                )
                spectra.append(conv)

            if not spectra:
                continue

            fname = (
                f"prun{calculation_id:03d}_irun{intensity_id:03d}_"
                f"{observable}_fwhm{str(fwhm)}.csv"
            )
            write_csv(
                file_path=output_directory / fname,
                rows=format_csv_for_spectra(
                    energy_range=energy_range,
                    convoluted_spectrum=np.array(spectra).T,
                    headers=headers,
                    fwhm=fwhm,
                ),
            )
