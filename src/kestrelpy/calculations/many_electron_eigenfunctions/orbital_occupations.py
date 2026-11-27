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

from itertools import combinations, product

import numpy as np

from kestrelpy.calculations.many_electron_eigenfunctions.basis_set import (
    POSSIBLE_SPIN_MULTIPLICITIES,
)
from kestrelpy.calculations.utils import get_minimal_d_config


def _calculate_combinations_from_n_electrons(
    number_of_electrons: int,
) -> list[tuple[int, ...]]:
    return list(combinations([0, 1, 2, 3, 4], r=number_of_electrons))


def _get_total_orbital_occupations(
    spin_up_occupations: list[tuple[int, ...]],
    spin_down_occupations: list[tuple[int, ...]],
) -> np.ndarray:
    total_occupations = [
        [*row[0], *row[1]]
        for row in list(product(spin_up_occupations, spin_down_occupations))
    ]

    # Filter out equivalent rows which are permutations of others.
    dedup_total_occupations = set(tuple(sorted(row)) for row in total_occupations)

    return np.array(list(dedup_total_occupations))


def _order_orbital_occupations(total_occupations: np.ndarray) -> np.ndarray:
    ncols = total_occupations.shape[1]
    for index in range(ncols - 1, -1, -1):
        total_occupations = total_occupations[total_occupations[:, index].argsort()]
    return total_occupations


def construct_orbital_occupations(d_electrons: int) -> dict[int, np.ndarray]:
    """Return all possible unique orbital combinations for the given d-electron count.

    Args:
        d_electrons: the number of d electrons.

    Returns:
        dict where the keys (int) are the spin multiplicities, and the values are
            a 2D numpy array with the orbital occupations.
    """
    minimal_d_electrons = get_minimal_d_config(d_electrons=d_electrons)
    spin_multiplicities = POSSIBLE_SPIN_MULTIPLICITIES[minimal_d_electrons]

    orbital_occupations: dict[int, np.ndarray] = {}
    for spin_mult in spin_multiplicities:
        num_unpaired = spin_mult - 1
        num_paired = d_electrons - num_unpaired

        number_of_spin_up = int(num_unpaired + (num_paired / 2))
        number_of_spin_down = int(num_paired / 2)

        spin_up_combinations = _calculate_combinations_from_n_electrons(
            number_of_electrons=number_of_spin_up
        )
        spin_down_combinations = _calculate_combinations_from_n_electrons(
            number_of_electrons=number_of_spin_down
        )
        total_occupations = _get_total_orbital_occupations(
            spin_up_occupations=spin_up_combinations,
            spin_down_occupations=spin_down_combinations,
        )
        ordered_total_occupations = _order_orbital_occupations(
            total_occupations=total_occupations
        )
        orbital_occupations[spin_mult] = ordered_total_occupations

    return orbital_occupations
