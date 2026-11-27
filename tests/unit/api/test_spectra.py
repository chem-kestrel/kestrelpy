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

import json
import os
from pathlib import Path

import kestrelpy as kp
from kestrelpy.calculations.intensities.spectroscopy import (
    SpectroscopicTransition,
    Spectrum,
)


def test_convolute_spectrum() -> None:
    x, y = kp.convolute_spectrum(
        spectrum=Spectrum(
            temperature=300.0,
            magnetic_field=0.0,
            transitions=[SpectroscopicTransition(energy=20000.0, intensity=1.0)],
        ),
        fwhm=1000.0,
        start=5000.0,
        final=30000.0,
        steps=250,
    )
    assert len(x) == 250
    assert x[0] == 5000.0
    assert x[-1] == 30000.0
    assert len(y) == 250


def test_convolute_spectrum_from_file(tmp_path: Path) -> None:
    file_path = tmp_path / "spectra.json"
    mock_data = {
        "calculation_id": 1,
        "intensity_id": 1,
        "absorption": [
            {
                "temperature": 300.0,
                "magnetic_field": 0.0,
                "transitions": [
                    {
                        "energy": 20000.0,
                        "intensity": 1.0,
                    },
                ],
            },
            {
                "temperature": 200.0,
                "magnetic_field": 0.0,
                "transitions": [
                    {
                        "energy": 20000.0,
                        "intensity": 1.0,
                    },
                ],
            },
        ],
        "cd": [],
        "mcd": [],
    }
    with open(file_path, "w") as f:
        json.dump(mock_data, f)

    output = kp.convolute_spectrum_from_file(
        file_path=file_path,
        calculation_type="absorption",
        fwhm=1000.0,
        start=5000.0,
        final=30000.0,
        steps=250,
    )
    assert len(output) == 2

    for (label, x, y), expected_label in zip(
        output, ("T=300.0K, B=0.0T", "T=200.0K, B=0.0T")
    ):
        assert label == expected_label
        assert len(x) == 250
        assert x[0] == 5000.0
        assert x[-1] == 30000.0
        assert len(y) == 250


def test_convolute_spectra_from_files(tmp_path: Path) -> None:
    ids = ((0, 0), (0, 1), (1, 0), (1, 1))
    for run_id, intensity_id in ids:
        file_path = (
            tmp_path / f"test_run00{run_id}_intensity00{intensity_id}_spectra.json"
        )
        mock_data = {
            "calculation_id": run_id,
            "intensity_id": intensity_id,
            "absorption": [
                {
                    "temperature": 300.0,
                    "magnetic_field": 0.0,
                    "transitions": [
                        {
                            "energy": 20000.0,
                            "intensity": 1.0,
                        },
                    ],
                },
                {
                    "temperature": 200.0,
                    "magnetic_field": 0.0,
                    "transitions": [
                        {
                            "energy": 20000.0,
                            "intensity": 1.0,
                        },
                    ],
                },
            ],
            "cd": [
                {
                    "temperature": 300.0,
                    "magnetic_field": 0.0,
                    "transitions": [
                        {
                            "energy": 20000.0,
                            "intensity": 1.0,
                        },
                    ],
                },
            ],
            "mcd": [
                {
                    "temperature": 300.0,
                    "magnetic_field": 0.0,
                    "transitions": [
                        {
                            "energy": 20000.0,
                            "intensity": 1.0,
                        },
                    ],
                },
            ],
        }
        with open(file_path, "w") as f:
            json.dump(mock_data, f)

    output_dir = tmp_path / "convoluted_spectra/"
    kp.convolute_spectra_from_files(
        input=tmp_path / "test_run00*_intensity00*_spectra.json",
        output_directory=output_dir,
        fwhm=1000.0,
        start=5000.0,
        final=30000.0,
        steps=250,
    )

    files = os.listdir(output_dir)
    assert len(files) == 12  # 4 * 3 File for absorption, CD, and MCD
    assert sorted(files) == sorted(
        [
            f"prun00{run_id}_irun00{intensity_id}_{mode}_fwhm1000.0.csv"
            for (run_id, intensity_id) in ids
            for mode in ("absorption", "cd", "mcd")
        ]
    )
