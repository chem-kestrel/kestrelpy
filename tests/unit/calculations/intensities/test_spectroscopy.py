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

import numpy as np
import pytest

from kestrelpy.calculations.constants import DEBYE_TO_ESU_CM
from kestrelpy.calculations.intensities.spectroscopy import (
    SpectroscopicTransition,
    Spectrum,
    calculate_gaussian,
    calculate_molar_absorption_coefficient,
    calculate_oscillator_strength,
    convolute_spectrum,
    max_thermally_occupied_energy,
)


@pytest.mark.parametrize(
    ["tdpm2_au", "energy", "expected_fosc"],
    [
        # The below test cases were taken from:
        #   https://www.faccts.de/docs/orca/5.0/tutorials/spec/UVVis.html
        # Copying and pasting the text:
        # ````
        # -----------------------------------------------------------------------------
        #     ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
        # -----------------------------------------------------------------------------
        # State   Energy    Wavelength  fosc         T2        TX        TY        TZ
        #         (cm-1)      (nm)                 (au**2)    (au)      (au)      (au)
        # -----------------------------------------------------------------------------
        #    1   19699.5    507.6   0.047917124   0.80078   0.87904  -0.16736   0.00724
        #    2   30354.1    329.4   0.012327447   0.13370   0.05769   0.36102  -0.00597
        #    3   31373.4    318.7   0.023725133   0.24896  -0.48626   0.08573  -0.07183
        #    4   32906.2    303.9   0.002354014   0.02355  -0.04012  -0.14813   0.00027
        #    5   34099.1    293.3   0.176508288   1.70411   1.27793  -0.24569  -0.10310
        # [...]
        # ````
        (0.80078, 19699.5, 0.047917124),
        (0.13370, 30354.1, 0.012327447),
        (0.24896, 31373.4, 0.023725133),
        (0.02355, 32906.2, 0.002354014),
        (1.70411, 34099.1, 0.176508288),
    ],
)
def test_conversion_from_transition_dipole_moment_to_oscillator_strengths(
    tdpm2_au: float, energy: float, expected_fosc: float
) -> None:
    # Validate the expected oscilattor strength (fosc)
    wavenumber_to_hartree = 1 / 219474.63
    energy_hartree = energy * wavenumber_to_hartree
    assert np.isclose(
        (2.0 / 3.0) * energy_hartree * tdpm2_au, expected_fosc, atol=1.0e-05
    )

    # Calculate the transition dipole moment**2 in units of Debye**2.
    au_to_debye = 1 / 0.39456
    tdpm2_debye = tdpm2_au * (au_to_debye**2.0) * (DEBYE_TO_ESU_CM**2.0)

    # Calculate the oscillator strength
    calculated_fosc = calculate_oscillator_strength(
        transition_dipole_moment_squared=tdpm2_debye,
        energy=energy,
    )
    assert np.isclose(calculated_fosc, expected_fosc, atol=1.0e-2)


def test_calculate_gaussian() -> None:
    energy_range = np.linspace(0.0, 10000.0, 500)
    output = calculate_gaussian(
        energy_range=energy_range,
        transition_energies=np.ascontiguousarray(5000.0),
        intensities=np.ascontiguousarray(calculate_molar_absorption_coefficient(0.001)),
        fwhm=2000.0,
    )
    expected_output = np.genfromtxt(
        Path(__file__).parent / "data" / "expected_gaussian.csv", delimiter=","
    )

    assert np.allclose(output, expected_output)


def test_max_thermally_occupied_energy() -> None:
    energies = np.array(
        [
            0.0,
            0.0,
            100.0,
            150.0,
            500.0,
            1000.0,
            1500.0,
            2500.0,
        ]
    )
    max_index = max_thermally_occupied_energy(
        energies=energies, temperature=300.0, limit=0.99
    )
    assert max_index == 4


def test_calculate_gaussian_full_width_half_maximum() -> None:
    energy_range = np.array([-10.0, 0.0, 10.0])
    output = calculate_gaussian(
        energy_range=energy_range,
        transition_energies=np.array([0.0]),
        intensities=np.array([1.0]),
        fwhm=20.0,
    )[0]
    assert abs(output[0] - (output[1] * 0.5)) < 1.0e-6
    assert abs(output[2] - (output[1] * 0.5)) < 1.0e-6


def test_convolute_spectrum() -> None:
    spectrum = Spectrum(
        temperature=300.0,
        magnetic_field=0.0,
        transitions=[
            SpectroscopicTransition(
                energy=5000.0,
                intensity=1.0,
            ),
            SpectroscopicTransition(
                energy=10000.0,
                intensity=2.0,
            ),
        ],
    )
    convoluted_spectrum = convolute_spectrum(
        spectrum=spectrum,
        energy_range=np.linspace(3000.0, 12000.0, 300, endpoint=True),
        fwhm=500.0,
    )
    assert convoluted_spectrum.shape == (2, 300)
    expected_convolution = np.genfromtxt(
        Path(__file__).parent / "data" / "expected_convolution.csv", delimiter=","
    )
    assert np.allclose(convoluted_spectrum, expected_convolution)
