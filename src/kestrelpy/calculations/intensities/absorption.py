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

from kestrelpy.calculations.constants import DEBYE_TO_ESU_CM
from kestrelpy.calculations.intensities.spectroscopy import (
    IntensityResults,
    SpectroscopicTransition,
    Spectrum,
    calculate_molar_absorption_coefficient,
    calculate_oscillator_strength,
    get_boltzmann_weightings,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ElectronicLevel,
)
from kestrelpy.calculations.solve_matrix import solve_matrix


@dataclass
class Transition:
    energy: float
    fosc: float
    transition_dipole_moments_squared: np.ndarray
    from_level: ElectronicLevel
    to_level: ElectronicLevel


@dataclass
class Absorption(IntensityResults):
    transitions: list[Transition]
    energies: list[float]  # Energies of occupied ground levels

    def get_spectrum(self, temperature: float) -> Spectrum:
        boltzmann_weightings = get_boltzmann_weightings(
            energies=np.array(self.energies),
            temperatures=np.ascontiguousarray(temperature),
        )

        spectroscopic_transitions: list[SpectroscopicTransition] = []
        for transition in self.transitions:
            level_index = transition.from_level.index
            degeneracy = transition.from_level.degeneracy
            spectroscopic_transitions.append(
                SpectroscopicTransition(
                    energy=transition.energy,
                    intensity=calculate_molar_absorption_coefficient(
                        fosc=transition.fosc
                    )
                    * boltzmann_weightings[level_index, 0]
                    / degeneracy,
                )
            )

        return Spectrum(
            temperature=temperature,
            magnetic_field=0.0,
            transitions=spectroscopic_transitions,
        )


def calculate_absorption(
    levels: list[ElectronicLevel],
    electronic_transition_dipole_moments: np.ndarray,
    n_ground_levels: int = 1,
) -> Absorption:
    """Calculate absorption oscillator strength.

    Args:
        levels: The electronic Eigenfunctions levels.
        electronic_transition_dipole_moments: The electronic transition dipole moments.
        n_ground_levels: The thermally occupied ground levels to compute transitions from.

    Returns:
        Absorption object.
    """
    transitions: list[Transition] = []
    for ground_level in levels[:n_ground_levels]:
        for excited_level in levels[n_ground_levels:]:
            tensor = np.zeros((3, 3), dtype=float)
            # Tensor for a single transition
            for a, b in [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 1),
                (1, 2),
                (2, 2),
            ]:  # alpha, beta: {x, y, z}
                for i0 in ground_level.eigenfunctions:
                    for j0 in excited_level.eigenfunctions:
                        tensor[a, b] += np.real(
                            electronic_transition_dipole_moments[a, i0.index, j0.index]
                            * electronic_transition_dipole_moments[
                                b, j0.index, i0.index
                            ]
                            * (DEBYE_TO_ESU_CM**2.0)
                        )
                tensor[b, a] = tensor[a, b]
            solution = solve_matrix(tensor)
            energy = excited_level.energy - ground_level.energy
            transitions.append(
                Transition(
                    energy=float(energy),
                    fosc=calculate_oscillator_strength(
                        transition_dipole_moment_squared=np.sum(solution.values),
                        energy=energy,
                    ),
                    transition_dipole_moments_squared=solution.values,
                    from_level=ground_level,
                    to_level=excited_level,
                )
            )
    return Absorption(
        transitions=transitions,
        energies=[lvl.energy for lvl in levels],
    )
