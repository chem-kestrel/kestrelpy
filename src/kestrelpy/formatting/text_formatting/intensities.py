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

from typing import Any

from tabulate import tabulate

from kestrelpy.calculations.intensities.absorption import Transition
from kestrelpy.calculations.intensities.circular_dichroism import CDTransition
from kestrelpy.calculations.intensities.magnetic_circular_dichroism import (
    MagneticCircularDichroism,
    MCDTransition,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    Eigenfunction,
    ElectronicLevel,
)


def format_transition_between_eigenfunctions(
    from_eigenfunctions: list[Eigenfunction], to_eigenfunctions: list[Eigenfunction]
) -> str:
    ground_eigenfunctions = [f.index for f in from_eigenfunctions]
    excited_eigenfunctions = [f.index for f in to_eigenfunctions]

    from_eigenfunction_indices = (
        ground_eigenfunctions[0]
        if len(ground_eigenfunctions) == 1
        else f"{min(ground_eigenfunctions)},{max(ground_eigenfunctions)}"
    )
    to_eigenfunction_indices = (
        excited_eigenfunctions[0]
        if len(excited_eigenfunctions) == 1
        else f"{min(excited_eigenfunctions)},{max(excited_eigenfunctions)}"
    )

    return f"{from_eigenfunction_indices} -> {to_eigenfunction_indices}"


def format_transition_between_levels(
    from_level: ElectronicLevel, to_level: ElectronicLevel
) -> str:
    return format_transition_between_eigenfunctions(
        from_eigenfunctions=from_level.eigenfunctions,
        to_eigenfunctions=to_level.eigenfunctions,
    )


def format_transition_section(transitions: list[Transition]) -> str:
    table_rows: list[list[Any]] = []
    for index, transition in enumerate(transitions):
        table_rows.append(
            [
                index,
                f"{transition.from_level.index} -> {transition.to_level.index}",
                format_transition_between_levels(
                    from_level=transition.from_level,
                    to_level=transition.to_level,
                ),
                transition.energy,
                transition.fosc,
                sum(transition.transition_dipole_moments_squared * 1e36),
            ]
        )

    return tabulate(
        tabular_data=table_rows,
        headers=[
            "Index",
            "Levels",
            "Eigenfunctions",
            "Energy (cm-1)",
            "Oscillator Strength",
            "D2 (Debye**2)",
        ],
        floatfmt=("", "", "", ".1f", "5.5e", "5.5e"),
    )


def format_rotary_strength_table(transitions: list[CDTransition]) -> str:
    table_rows: list[list[Any]] = []
    for index, transition in enumerate(transitions):
        table_rows.append(
            [
                index + 0,
                f"{transition.from_level.index} -> {transition.to_level.index}",
                format_transition_between_levels(
                    from_level=transition.from_level,
                    to_level=transition.to_level,
                ),
                transition.energy,
                transition.rotary_strength,
            ]
        )

    return tabulate(
        tabular_data=table_rows,
        headers=[
            "Index",
            "Levels",
            "Eigenfunctions",
            "Energy (cm-1)",
            "Rotary strength (cgs)",
        ],
        floatfmt=(
            "",
            "",
            "",
            ".1f",
            ".5g",
        ),
    )


def get_average_mcd_intensities(
    magnetic_circular_dichroism: MagneticCircularDichroism,
) -> list[MCDTransition]:
    n_transitions = len(magnetic_circular_dichroism.orientations[0].transitions)
    energies = [0.0] * n_transitions
    d_fosc = [0.0] * n_transitions

    for orientation in magnetic_circular_dichroism.orientations:
        for index, transition in enumerate(orientation.transitions):
            energies[index] += transition.energy * orientation.weight
            d_fosc[index] += transition.d_fosc * orientation.weight

    averaged_transitions: list[MCDTransition] = []
    for i, transition in enumerate(
        magnetic_circular_dichroism.orientations[0].transitions
    ):
        averaged_transitions.append(
            MCDTransition(
                energy=energies[i],
                d_fosc=d_fosc[i],
                from_eigenfunction=transition.from_eigenfunction,
                to_eigenfunction=transition.to_eigenfunction,
            )
        )
    return averaged_transitions


def format_magnetic_circular_dichroism_table(transitions: list[MCDTransition]) -> str:
    table_rows: list[list[Any]] = []
    for index, transition in enumerate(transitions):
        table_rows.append(
            [
                index + 0,
                format_transition_between_eigenfunctions(
                    from_eigenfunctions=[transition.from_eigenfunction],
                    to_eigenfunctions=[transition.to_eigenfunction],
                ),
                transition.energy,
                transition.d_fosc,
            ]
        )

    return tabulate(
        tabular_data=table_rows,
        headers=[
            "Index",
            "Eigenfunctions",
            "Energy (cm-1)",
            "d_fosc",
        ],
        floatfmt=(
            "",
            "",
            ".1f",
            ".5g",
        ),
    )
