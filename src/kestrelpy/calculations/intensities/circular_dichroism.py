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

from dataclasses import dataclass

import numpy as np

from kestrelpy.calculations.constants import BOHR_MAGNETON_CGS, DEBYE_TO_ESU_CM
from kestrelpy.calculations.intensities.spectroscopy import (
    CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION,
    IntensityResults,
    SpectroscopicTransition,
    Spectrum,
    get_boltzmann_weightings,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ElectronicLevel,
)


@dataclass
class CDTransition:
    energy: float
    # Requires etdm in 10**-18 esu cm; 1 D = 1e-18 esu cm
    # Requires mag moment in erg G**-1; 1 bohr = 9.274e-21 erg G-1
    rotary_strength: float
    from_level: ElectronicLevel
    to_level: ElectronicLevel


@dataclass
class CircularDichroism(IntensityResults):
    transitions: list[CDTransition]
    energies: list[float]

    def get_spectrum(self, temperature: float) -> Spectrum:
        boltzmann_weightings = get_boltzmann_weightings(
            energies=np.array(self.energies),
            temperatures=np.ascontiguousarray(temperature),
        )

        spectroscopic_transitions: list[SpectroscopicTransition] = []
        for transition in self.transitions:
            spectroscopic_transitions.append(
                SpectroscopicTransition(
                    energy=transition.energy,
                    intensity=(
                        transition.rotary_strength
                        * transition.energy
                        * CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION
                        * boltzmann_weightings[transition.from_level.index, 0]
                        / transition.from_level.degeneracy
                    ),
                )
            )

        return Spectrum(
            temperature=temperature,
            magnetic_field=0.0,
            transitions=spectroscopic_transitions,
        )


def calculate_cd(
    levels: list[ElectronicLevel],
    electronic_transition_dipole_moment: np.ndarray,
    muxyz: np.ndarray,
    n_ground_levels: int = 1,
) -> CircularDichroism:
    """Calculate circular dichroism intensities.

    Args:
        levels: The electronic Eigenfunctions levels.
        electronic_transition_dipole_moments: The electronic transition dipole moments.
        muxyz: The magnetic moment matrix elements.
        n_ground_levels: The thermally occupied ground levels to compute transitions from.

    Returns:
        CircularDichroism object.
    """
    transitions: list[CDTransition] = []
    for ground_level in levels[:n_ground_levels]:
        for excited_level in levels[n_ground_levels:]:
            rotary_strength = 0.0
            for i0 in ground_level.eigenfunctions:
                for j0 in excited_level.eigenfunctions:
                    rotary_strength += np.imag(
                        np.dot(
                            electronic_transition_dipole_moment[:, i0.index, j0.index]
                            * DEBYE_TO_ESU_CM,
                            muxyz[:, j0.index, i0.index] * BOHR_MAGNETON_CGS,
                        )
                    )
            energy = excited_level.energy - ground_level.energy
            transitions.append(
                CDTransition(
                    energy=energy,
                    rotary_strength=rotary_strength,
                    from_level=ground_level,
                    to_level=excited_level,
                )
            )
    return CircularDichroism(
        transitions=transitions, energies=[lvl.energy for lvl in levels]
    )
