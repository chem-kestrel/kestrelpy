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

from typing import TypeAlias

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.linalg import ishermitian

from kestrelpy.calculations.effective_hamiltonian import ConstructEffectiveHamiltonian
from kestrelpy.calculations.interelectronic_repulsion import (
    InterelectronRepulsion,
    get_interelectronic_repulsion_parameters,
)
from kestrelpy.calculations.ligand_field_matrix import (
    MultipoleExpansion,
    get_multipole_expansion,
)
from kestrelpy.calculations.many_electron_matrices import get_non_zero_matrix_elements
from kestrelpy.calculations.parameters import (
    MagneticField,
    OrbitalReductionFactors,
    SpinOrbitCoupling,
)

Parameters: TypeAlias = tuple[
    MultipoleExpansion,
    InterelectronRepulsion,
    SpinOrbitCoupling,
    MagneticField,
    OrbitalReductionFactors,
]


@pytest.fixture()
def effective_hamiltonian_parameters() -> Parameters:
    return (
        get_multipole_expansion(
            np.array(
                [
                    [1130.4, 173.0, 85.1, -118.3, 1047.0],
                    [173.0, 339.4, 568.2, -56.9, 1130.9],
                    [85.1, 568.2, 3676.1, -537.7, -488.0],
                    [-118.3, -56.9, -537.7, 1789.5, 1009.6],
                    [1047.0, 1130.9, -488.0, 1009.6, 16864.6],
                ]
            )
        ),
        get_interelectronic_repulsion_parameters(racah_b=1000.0, racah_c=4000.0),
        SpinOrbitCoupling(SOC=500.0),
        MagneticField(BX=1.0, BY=1.0, BZ=1.0),
        OrbitalReductionFactors(KX=1.0, KY=1.0, KZ=1.0),
    )


class TestEffectiveHamiltonian:
    @pytest.mark.parametrize(
        "d_electrons",
        [
            pytest.param(1, id="d1"),
            pytest.param(2, id="d2"),
            pytest.param(3, id="d3"),
            pytest.param(4, id="d4"),
            pytest.param(5, id="d5"),
            pytest.param(6, id="d6"),
            pytest.param(7, id="d7"),
            pytest.param(8, id="d8"),
            pytest.param(9, id="d9"),
        ],
    )
    def test_matrix_is_hermitian(
        self,
        d_electrons: int,
        effective_hamiltonian_parameters: Parameters,
    ) -> None:
        matrices = get_non_zero_matrix_elements(d_electrons=d_electrons)
        effective_hamiltonian = ConstructEffectiveHamiltonian(
            d_electrons=d_electrons, matrices=matrices
        )
        matrix, ligand_field_matrix, matrix_nosoc = effective_hamiltonian.construct(
            *effective_hamiltonian_parameters
        )
        assert ishermitian(matrix, atol=1.0e-05, rtol=1.0e-05)
        assert ishermitian(ligand_field_matrix, atol=1.0e-05, rtol=1.0e-05)
        assert ishermitian(matrix_nosoc, atol=1.0e-05, rtol=1.0e-05)


@pytest.mark.parametrize(
    "d_config",
    [
        pytest.param(1, id="d1"),
        pytest.param(2, id="d2"),
        pytest.param(3, id="d3"),
        pytest.param(4, id="d4"),
        pytest.param(5, id="d5"),
        pytest.param(6, id="d6"),
        pytest.param(7, id="d7"),
        pytest.param(8, id="d8"),
        pytest.param(9, id="d9"),
    ],
)
def test_benchmark_constructing_effective_hamiltonian(
    d_config: int,
    effective_hamiltonian_parameters: Parameters,
    benchmark: BenchmarkFixture,
) -> None:
    matrix_elements = get_non_zero_matrix_elements(d_electrons=d_config)
    effective_hamiltonian = ConstructEffectiveHamiltonian(
        d_electrons=d_config, matrices=matrix_elements
    )
    benchmark(effective_hamiltonian.construct, *effective_hamiltonian_parameters)
