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

import numpy as np
from scipy.optimize import linear_sum_assignment

from kestrelpy.calculations.many_electron_eigenfunctions.basis_set import (
    POSSIBLE_SPIN_MULTIPLICITIES,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    Eigenfunction,
    ManyElectronEigenfunctions,
)
from kestrelpy.calculations.utils import get_minimal_d_config
from kestrelpy.comparison.data import SpinMultiplicityEnergies


class SpinMultiplicityEnergyAssignmentError(Exception):
    """Raised when an issue was encountered with the spin multiplicity"""

    pass


def validate_spin_multiplicity_assignments(
    d_electrons: int, assignments: list[SpinMultiplicityEnergies]
) -> None:
    """Validate the spin multiplicity assignments are valid"""
    minimal_d_electrons = get_minimal_d_config(d_electrons=d_electrons)
    possible_spin_multiplicities = POSSIBLE_SPIN_MULTIPLICITIES[minimal_d_electrons]

    seen_spin_multiplicities: set[int] = set()
    for assignment in assignments:
        # Check that there is a single spin multiplicity assignment.
        if assignment.spin_multiplicity in seen_spin_multiplicities:
            raise SpinMultiplicityEnergyAssignmentError
        seen_spin_multiplicities.add(assignment.spin_multiplicity)

        # Check that the spin multiplicity is valid.
        if assignment.spin_multiplicity not in possible_spin_multiplicities:
            raise SpinMultiplicityEnergyAssignmentError


def match_energies_with_spin_multiplicity_assignments(
    eigenfunctions: ManyElectronEigenfunctions,
    assignments: list[SpinMultiplicityEnergies],
) -> list[list[float]]:
    """Match eigenfunction energies to the list of spin multiplicity assignments."""

    def get_maximal_spin_multiplicity(eigenfunction: Eigenfunction) -> int:
        return max(
            eigenfunction.spin_multiplicity_projection,
            key=eigenfunction.spin_multiplicity_projection.get,  # type: ignore[arg-type]
        )

    comparison: list[list[float]] = []
    for assignment in assignments:
        spin_multiplicity = assignment.spin_multiplicity
        possible_eigenfunctions: list[Eigenfunction] = [
            f
            for f in eigenfunctions.eigenfunctions
            if get_maximal_spin_multiplicity(f) == spin_multiplicity
        ]

        if assignment.is_ordered:
            comparison.append(
                [
                    round(possible_eigenfunctions[i].energy, 1)
                    for i in range(len(assignment.energies))
                ]
            )
        else:
            diff = np.abs(
                np.array([f.energy for f in possible_eigenfunctions])[:, np.newaxis]
                - np.array(assignment.energies)[np.newaxis, :]
            )
            row_idx, _ = linear_sum_assignment(diff)

            comparison.append(
                [round(possible_eigenfunctions[index].energy, 1) for index in row_idx]
            )
    return comparison


def fit_transition_energies(
    spin_multiplicity_assignments: list[SpinMultiplicityEnergies],
    eigenfunctions: ManyElectronEigenfunctions,
) -> float:
    matched_energies = match_energies_with_spin_multiplicity_assignments(
        eigenfunctions=eigenfunctions,
        assignments=spin_multiplicity_assignments,
    )

    n = 0
    diff = 0.0
    for calculated_energies, assignment in zip(
        matched_energies, spin_multiplicity_assignments, strict=True
    ):
        n += len(calculated_energies)
        diff += float(
            np.sum((np.array(calculated_energies) - np.array(assignment.energies)) ** 2)
        )
    rms = float(np.sqrt(diff / n))
    return rms
