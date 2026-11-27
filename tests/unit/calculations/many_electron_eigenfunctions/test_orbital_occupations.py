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
import pytest

from kestrelpy.calculations.many_electron_eigenfunctions.orbital_occupations import (
    _order_orbital_occupations,
    construct_orbital_occupations,
)


@pytest.mark.parametrize(
    "d_electrons",
    list(range(1, 10)),
)
def test_orbital_occupations_are_unique(d_electrons: int) -> None:
    orbital_occupations = construct_orbital_occupations(d_electrons=d_electrons)

    for spin_multiplicity, occupations in orbital_occupations.items():
        # Numpy rows are not hashable, so we need to convert to hashable form
        hashable_row: list[tuple[int, ...]] = [
            tuple(row) for row in occupations.tolist()
        ]
        assert len(occupations) == len(set(hashable_row))


def test_order_orbital_occupations() -> None:
    occupations = np.vstack(
        (
            np.array([2, 1, 0]),
            np.array([1, 2, 0]),
            np.array([0, 1, 2]),
        )
    )
    expected_ordered_occupations = np.vstack(
        (
            np.array([0, 1, 2]),
            np.array([1, 2, 0]),
            np.array([2, 1, 0]),
        )
    )
    ordered_occupations = _order_orbital_occupations(total_occupations=occupations)
    assert np.array_equal(ordered_occupations, expected_ordered_occupations)
