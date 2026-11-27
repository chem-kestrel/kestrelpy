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

from kestrelpy.calculations.many_electron_eigenfunctions.basis_set import (
    get_many_electron_basis,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ManyElectronEigenfunctions,
    get_configuration_projection,
    get_many_electron_eigenfunctions,
)
from kestrelpy.calculations.many_electron_eigenfunctions.orbital_occupations import (
    construct_orbital_occupations,
)


@pytest.fixture
def dummy_eigenfunctions() -> ManyElectronEigenfunctions:
    return get_many_electron_eigenfunctions(
        many_electron_basis=get_many_electron_basis(d_electrons=1),
        effective_hamiltonian=np.diag(
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        ),
    )


def test_get_configuration_projection_adds_configurations(
    dummy_eigenfunctions: ManyElectronEigenfunctions,
) -> None:
    # Assert no configurations prior to calling `get_configuration_projection`
    for eigenfunction in dummy_eigenfunctions.eigenfunctions:
        assert len(eigenfunction.configurations) == 0

    for level in dummy_eigenfunctions.levels:
        assert len(level.configurations) == 0

    # Add configuration projections
    eigenfunctions = get_configuration_projection(
        many_electron_eigenfunctions=dummy_eigenfunctions,
        ligand_field_eigenfunctions=dummy_eigenfunctions,
        ligand_field_orbital_energies=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        orbital_occupations=construct_orbital_occupations(d_electrons=1),
    )

    # Assert configuration projections now exist.
    for eigenfunction in eigenfunctions.eigenfunctions:
        assert len(eigenfunction.configurations) == 1

    for level in eigenfunctions.levels:
        assert len(level.configurations) == 1
