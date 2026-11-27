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

from unittest.mock import Mock

import numpy as np

from kestrelpy.calculations.constants import BOLTZMANN
from kestrelpy.calculations.intensities.absorption import Absorption, Transition
from kestrelpy.calculations.intensities.spectroscopy import (
    calculate_molar_absorption_coefficient,
)
from tests.unit.calculations.intensities.conftest import get_electronic_level


def test_absorption_calculates_boltzmann_weighting() -> None:
    absorption = Absorption(
        transitions=[
            Transition(
                energy=1500.0,
                fosc=1.0,
                transition_dipole_moments_squared=Mock(),
                from_level=get_electronic_level(index=0, degeneracy=2),
                to_level=Mock(),
            ),
            Transition(
                energy=3000.0,
                fosc=2.0,
                transition_dipole_moments_squared=Mock(),
                from_level=get_electronic_level(index=0, degeneracy=2),
                to_level=Mock(),
            ),
            Transition(
                energy=1495.0,
                fosc=3.0,
                transition_dipole_moments_squared=Mock(),
                from_level=get_electronic_level(index=1, degeneracy=3),
                to_level=Mock(),
            ),
            Transition(
                energy=2995.0,
                fosc=4.0,
                transition_dipole_moments_squared=Mock(),
                from_level=get_electronic_level(index=1, degeneracy=3),
                to_level=Mock(),
            ),
        ],
        energies=[0.0, 5.0],
    )
    spectrum_300K = absorption.get_spectrum(temperature=300.0)

    denominator = BOLTZMANN * 300.0
    partition_function = np.exp(-np.array([0.0, 5.0]) / denominator)

    spectrum_300K.transitions[0].energy == 1500.0
    spectrum_300K.transitions[0].intensity == (
        calculate_molar_absorption_coefficient(fosc=1.0)
        * np.exp(-0.0 / denominator)
        / (partition_function * 2)
    )
    spectrum_300K.transitions[1].energy == 3000.0
    spectrum_300K.transitions[1].intensity == (
        calculate_molar_absorption_coefficient(fosc=2.0)
        * np.exp(-0.0 / denominator)
        / (partition_function * 2)
    )
    spectrum_300K.transitions[2].energy == 1495.0
    spectrum_300K.transitions[2].intensity == (
        calculate_molar_absorption_coefficient(fosc=3.0)
        * np.exp(-5.0 / denominator)
        / (partition_function * 3)
    )
    spectrum_300K.transitions[3].energy == 2995.0
    spectrum_300K.transitions[3].intensity == (
        calculate_molar_absorption_coefficient(fosc=4.0)
        * np.exp(-5.0 / denominator)
        / (partition_function * 3)
    )
