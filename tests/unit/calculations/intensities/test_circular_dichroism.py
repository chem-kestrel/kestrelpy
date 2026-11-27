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
from kestrelpy.calculations.intensities.circular_dichroism import (
    CDTransition,
    CircularDichroism,
)
from kestrelpy.calculations.intensities.spectroscopy import (
    CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION,
)
from tests.unit.calculations.intensities.conftest import get_electronic_level


def test_circular_dichroism_calculates_boltzmann_weighting() -> None:
    circular_dichroism = CircularDichroism(
        transitions=[
            CDTransition(
                energy=1500.0,
                rotary_strength=1.0,
                from_level=get_electronic_level(index=0, degeneracy=2),
                to_level=Mock(),
            ),
            CDTransition(
                energy=3000.0,
                rotary_strength=2.0,
                from_level=get_electronic_level(index=0, degeneracy=2),
                to_level=Mock(),
            ),
            CDTransition(
                energy=1495.0,
                rotary_strength=3.0,
                from_level=get_electronic_level(index=1, degeneracy=3),
                to_level=Mock(),
            ),
            CDTransition(
                energy=2995.0,
                rotary_strength=4.0,
                from_level=get_electronic_level(index=1, degeneracy=3),
                to_level=Mock(),
            ),
        ],
        energies=[0.0, 5.0],
    )
    spectrum_300K = circular_dichroism.get_spectrum(temperature=300.0)

    denominator = BOLTZMANN * 300.0
    partition_function = np.exp(-np.array([0.0, 5.0]) / denominator)

    spectrum_300K.transitions[0].energy == 1500.0
    spectrum_300K.transitions[0].intensity == (
        1.0
        * CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION
        * np.exp(-1500.0 / denominator)
        / (partition_function * 2)
    )
    spectrum_300K.transitions[1].energy == 3000.0
    spectrum_300K.transitions[1].intensity == (
        2.0
        * CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION
        * np.exp(-3000.0 / denominator)
        / (partition_function * 2)
    )
    spectrum_300K.transitions[2].energy == 1495.0
    spectrum_300K.transitions[2].intensity == (
        3.0
        * CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION
        * np.exp(-1495.0 / denominator)
        / (partition_function * 3)
    )
    spectrum_300K.transitions[3].energy == 2995.0
    spectrum_300K.transitions[3].intensity == (
        4.0
        * CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION
        * np.exp(-2995.0 / denominator)
        / (partition_function * 3)
    )
