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

import pytest

from kestrelpy.comparison.data import define_spin_multiplicity_assignments


@pytest.mark.parametrize(
    ("spin_multiplicity", "energies", "expand_flag", "expected_energies"),
    (
        (4, [0.0, 2000.0, 4000.0], False, [0.0, 2000.0, 4000.0]),
        (4, [0.0, 2000.0, 4000.0, 4000.0], False, [0.0, 2000.0, 4000.0, 4000.0]),
        (4, [0.0, 2000.0], True, [0.0, 0.0, 0.0, 0.0, 2000.0, 2000.0, 2000.0, 2000.0]),
    ),
)
def test_defined_spin_multiplicity_assignments(
    spin_multiplicity: int,
    energies: list[float],
    expand_flag: bool,
    expected_energies: list[float],
) -> None:
    assignments = define_spin_multiplicity_assignments(
        spin_multiplicity=spin_multiplicity,
        energies=energies,
        expand_by_spin_mult=expand_flag,
    )
    assert assignments.energies == expected_energies
    assert assignments.spin_multiplicity == spin_multiplicity
