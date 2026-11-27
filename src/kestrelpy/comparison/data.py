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


@dataclass
class SpinMultiplicityEnergies:
    spin_multiplicity: int
    energies: list[float]
    is_ordered: bool


def define_spin_multiplicity_assignments(
    spin_multiplicity: int,
    energies: list[float],
    expand_by_spin_mult: bool = False,
    is_ordered: bool = False,
) -> SpinMultiplicityEnergies:
    """Define the assigned energies of a given spin multiplicty.

    Args:
        spin_multiplicity: The 2S+1 spin multiplicity for these energies.
        energies: List of energies in wavenumbers (cm-1).
        expand_by_spin_mult: Flag for whether each energy is duplicated spin
            multiplicity-times. Default=False.
        is_ordered: Flag to determine whether the energies are in order.

    Returns:
        SpinMultiplicityEnergies

    Examples:
        >>> define_spin_multiplicity_assignments(
            spin_multiplicity=4,
            energies=[0.0, 2000.0],
            expand_by_spin_mult=True,
            is_ordered=False,
        )
        -> SpinMultiplicityEnergies(
            spin_multiplicity=4,
            energies=[0.0, 0.0, 0.0, 0.0, 2000.0, 2000.0, 2000.0, 2000.0],
            is_ordered=False,
        )
        >>> define_spin_multiplicity_assignments(
            spin_multiplicity=4,
            energies=[0.0, 2000.0],
            expand_by_spin_mult=False,
            is_ordered=False,
        )
        -> SpinMultiplicityEnergies(
            spin_multiplicity=4,
            energies=[0.0, 2000.0],
            is_ordered=False,
        )
    """

    if expand_by_spin_mult:
        _energies = []
        for energy in energies:
            _energies.extend([float(energy)] * int(spin_multiplicity))
    else:
        _energies = list(map(float, energies))

    return SpinMultiplicityEnergies(
        spin_multiplicity=int(spin_multiplicity),
        energies=_energies,
        is_ordered=bool(is_ordered),
    )


@dataclass
class EprData:
    g_factors: list[float]
    level: int = 0

    def __post_init__(self) -> None:
        if self.g_factors != sorted(self.g_factors):
            raise ValueError("'g_factors' must be ordered g1 < g2 < g2")
        if self.level < 0:
            raise ValueError("'level' cannot be < 0")
