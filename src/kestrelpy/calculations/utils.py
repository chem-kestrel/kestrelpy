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

from enum import Enum
from math import factorial


class FirstRowTransitionMetals(str, Enum):
    titanium = "Ti"
    vanadium = "V"
    chromium = "Cr"
    manganese = "Mn"
    iron = "Fe"
    cobalt = "Co"
    nickel = "Ni"
    copper = "Cu"


TRANSITION_METAL_GROUP: dict[FirstRowTransitionMetals, int] = {
    FirstRowTransitionMetals.titanium: 4,
    FirstRowTransitionMetals.vanadium: 5,
    FirstRowTransitionMetals.chromium: 6,
    FirstRowTransitionMetals.manganese: 7,
    FirstRowTransitionMetals.iron: 8,
    FirstRowTransitionMetals.cobalt: 9,
    FirstRowTransitionMetals.nickel: 10,
    FirstRowTransitionMetals.copper: 11,
}


def get_minimal_d_config(d_electrons: int) -> int:
    """Get the minimal d electron for a given configuration.

    For example, d1 and d9 are related by the electron-hole equivalence,
    in that d1 can be thought of as a single electron and d9 can be
    thought of as a single hole.

    This method would return 1 if d_electrons is 1 or 9.

    Args:
        d_electrons: The number of d electrons.

    Returns:
        The minimal equivalent of d electrons.
    """
    if d_electrons > 5:
        return 10 - d_electrons
    else:
        return d_electrons


def get_many_d_electron_basis_size(d_electrons: int) -> int:
    """Return the maximum size of the d_electron basis.

    Args:
        d_electrons (int): The number of d electrons.

    Returns:
        int: The full basis set size.
    """
    denominator = factorial(d_electrons) * factorial(10 - d_electrons)
    return int(factorial(10) / denominator)


def calculate_oxidation_state(
    element: FirstRowTransitionMetals, d_electrons: int
) -> int:
    """Return the oxidation state for a given metal a d-electron count"""
    return int(TRANSITION_METAL_GROUP[element] - d_electrons)
