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

from kestrelpy.calculations.constants import BOLTZMANN
from kestrelpy.calculations.interelectronic_repulsion import (
    get_interelectronic_repulsion_parameters,
)
from kestrelpy.calculations.ligand_field_matrix import get_ligand_field_parameters
from kestrelpy.calculations.magnetism.susceptibilities import (
    CONVERSION,
    _get_non_degenerate_energies,
    calculate_paramagnetic_susceptibilities,
)
from kestrelpy.calculations.parameters import (
    MagneticField,
    OrbitalReductionFactors,
    SpinOrbitCoupling,
)
from kestrelpy.models.input_models.input_models import Parameters
from kestrelpy.workflows.calculate_electronic_structure import run_parameters


def test_get_non_degenerate_energies() -> None:
    energies = np.array([0.0, 0.0, 1000.0, 1000.0, 2500.0, 2500.0, 2500.0, 2500.0])
    output = _get_non_degenerate_energies(energies=energies)

    expected_energy_diff = np.array(
        [
            [0.0, 0.0, -1000.0, -1000.0, -2500.0, -2500.0, -2500.0, -2500.0],
            [0.0, 0.0, -1000.0, -1000.0, -2500.0, -2500.0, -2500.0, -2500.0],
            [1000.0, 1000.0, 0.0, 0.0, -1500.0, -1500.0, -1500.0, -1500.0],
            [1000.0, 1000.0, 0.0, 0.0, -1500.0, -1500.0, -1500.0, -1500.0],
            [2500.0, 2500.0, 1500.0, 1500.0, 0.0, 0.0, 0.0, 0.0],
            [2500.0, 2500.0, 1500.0, 1500.0, 0.0, 0.0, 0.0, 0.0],
            [2500.0, 2500.0, 1500.0, 1500.0, 0.0, 0.0, 0.0, 0.0],
            [2500.0, 2500.0, 1500.0, 1500.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    expected_non_degenerate = np.array(
        [
            [False, False, True, True, True, True, True, True],
            [False, False, True, True, True, True, True, True],
            [True, True, False, False, True, True, True, True],
            [True, True, False, False, True, True, True, True],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
        ],
    )
    assert np.array_equal(output[0], expected_energy_diff)
    assert np.array_equal(output[1], expected_non_degenerate)


def calculate_susceptibility_from_russell_saunders(
    spin: float, orbital: float, j_total: float
) -> float:
    """Compute the paramagnetic susceptibility (cgs 1e-6) from Russell-Saunders coupling.

    References:
        1. Equation 10.9: B. Figgis, Introduction to ligand fields, Interscience Publishers, New York, 1966
    """
    g = 1 + (
        (spin * (spin + 1) - orbital * (orbital + 1) + j_total * (j_total + 1))
        / (2 * j_total * (j_total + 1))
    )
    return float(
        1e6 * g * g * j_total * (j_total + 1) * (CONVERSION / (3.0 * BOLTZMANN * 300.0))
    )


def ligand_field_susceptibility_Ag_first_order_only(
    spin: float, soc: float, delta10Dq: float, kiso: float
) -> float:
    """Compute the paramagnetic susceptibility (cgs 1e-6) for Ag from the First-order Zeeman.

    References:
        1. Equation 10.26: B. Figgis, Introduction to ligand fields, Interscience Publishers, New York, 1966
    """
    x_s_o = calculate_susceptibility_from_russell_saunders(spin, 0.0, spin) / 1e6
    return (x_s_o * (1.0 - 8.0 * kiso * kiso * soc / abs(delta10Dq))) * 1e6


def ligand_field_susceptibility_Ag_second_order(
    spin: float, soc: float, delta10Dq: float, kiso: float
) -> float:
    """Compute the paramagnetic susceptibility (cgs 1e-6) for Ag from the First+Second order Zeeman.

    References:
        1. Equation 10.32: B. Figgis, Introduction to ligand fields, Interscience Publishers, New York, 1966
    """
    x_s_o = calculate_susceptibility_from_russell_saunders(spin, 0.0, spin) / 1e6
    return (
        float(
            x_s_o * (1.0 - 8.0 * kiso * kiso * soc / abs(delta10Dq))
            + (8.0 * kiso * kiso * CONVERSION / abs(delta10Dq))
        )
        * 1e6
    )


def ligand_field_susceptibility_Eg_first_order_only(
    spin: float, soc: float, delta10Dq: float, kiso: float
) -> float:
    """Compute the paramagnetic susceptibility (cgs 1e-6) for Eg from the First-order Zeeman.

    References:
        1. Equation 10.28: B. Figgis, Introduction to ligand fields, Interscience Publishers, New York, 1966
    """
    x_s_o = calculate_susceptibility_from_russell_saunders(spin, 0.0, spin) / 1e6
    return (x_s_o * (1.0 - 4.0 * kiso * kiso * soc / abs(delta10Dq))) * 1e6


def get_parameters(
    d_electrons: int,
    delta10Dq: float,
    racah_b: float,
    racah_c: float,
    soc: float,
    kiso: float,
) -> Parameters:
    deltaDq = delta10Dq / 10.0
    return Parameters(
        d_electrons=d_electrons,
        ligand_field=get_ligand_field_parameters(
            np.array(
                [
                    [deltaDq, 0.0, 0.0, 0.0, 5.0 * deltaDq],
                    [0.0, -4.0 * deltaDq, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0 * deltaDq, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -4.0 * deltaDq, 0.0],
                    [5.0 * deltaDq, 0.0, 0.0, 0.0, deltaDq],
                ]
            )
        ),
        interelectronic_repulsion=get_interelectronic_repulsion_parameters(
            racah_b=racah_b, racah_c=racah_c
        ),
        spin_orbit_coupling=SpinOrbitCoupling(SOC=soc),
        magnetic_field=MagneticField(0.0, 0.0, 0.0),
        orbital_reduction_factors=OrbitalReductionFactors(KX=kiso, KY=kiso, KZ=kiso),
    )


@pytest.mark.parametrize(
    ["parameters", "foz_soz_limit", "expected_susceptibility"],
    (
        pytest.param(
            get_parameters(9, 0.0, 0.0, 0.0, 1000.0, 1.0),
            (6, 6),
            calculate_susceptibility_from_russell_saunders(0.5, 2.0, 2.5),
            id="D9, Free-ion, S=0.5, L=2.0, J=2.50",
        ),
        pytest.param(
            get_parameters(1, 0, 0.0, 0.0, 1000.0, 1.0),
            None,
            calculate_susceptibility_from_russell_saunders(0.50, 2.0, 1.50),
            id="D1 S=0.5, L=2.0, J=1.50",
        ),
        pytest.param(
            get_parameters(9, 4000, 0.0, 0.0, 0.0, 0.0),
            None,
            ligand_field_susceptibility_Eg_first_order_only(0.5, 0.0, 4000.0, 1.0),
            id="D9 2Eg ground state kiso=0 should be == kiso=1",
        ),
        pytest.param(
            get_parameters(9, 4000, 0.0, 0.0, 0.0, 1.0),
            None,
            ligand_field_susceptibility_Eg_first_order_only(0.5, 0.0, 4000.0, 1.0),
            id="D9 2Eg ground state kiso=1 should be == kiso=0",
        ),
        pytest.param(
            get_parameters(8, 4000, 1000.0, 4000.0, 0.0, 0.0),
            None,
            ligand_field_susceptibility_Ag_first_order_only(1.0, 0.0, 4000.0, 1.0),
            id="D8 3A2g ground state kiso=0 should be == kiso=1",
        ),
        pytest.param(
            get_parameters(8, 4000, 1000.0, 4000.0, 0.0, 1.0),
            None,
            ligand_field_susceptibility_Ag_first_order_only(1.0, 0.0, 4000.0, 1.0),
            id="D8 3A2g ground state kiso=1 should be == kiso=0",
        ),
        pytest.param(
            get_parameters(3, 4000, 0.0, 4000.0, 0.0, 0.0),
            None,
            ligand_field_susceptibility_Ag_first_order_only(1.5, 0.0, 4000.0, 1.0),
            id="D3 4A2g ground state kiso=0 should be == kiso=1",
        ),
        pytest.param(
            get_parameters(3, 4000, 0.0, 4000.0, 0.0, 1.0),
            None,
            ligand_field_susceptibility_Ag_first_order_only(1.5, 0.0, 4000.0, 1.0),
            id="D3 4A2g ground state kiso=1 should be == kiso=0",
        ),
        pytest.param(
            get_parameters(3, 4000, 0.0, 4000.0, 0.0, 0.0),
            (4, 120),
            ligand_field_susceptibility_Ag_second_order(1.5, 0.0, 4000.0, 0.0),
            id="D3 4A2g ground state kiso=0 should be == kiso=1",
        ),
        pytest.param(
            get_parameters(3, 4000, 0.0, 4000.0, 0.0, 1.0),
            (4, 120),
            ligand_field_susceptibility_Ag_second_order(1.5, 0.0, 4000.0, 1.0),
            id="D3 4A2g ground state kiso=1 should be == kiso=0",
        ),
        pytest.param(
            get_parameters(7, 0, 1000.0, 4000.0, 515.0, 1.0),
            (24, 28),
            18412.0,
            id="CAMMAG Co++ free-ion value for kiso=1.0 (co_d7_free_ion_output.txt)",
        ),
        pytest.param(
            get_parameters(7, 0, 1000.0, 4000.0, 515.0, 0.5),
            (24, 28),
            10704.0,
            id="CAMMAG Co++ free-ion value for kiso=0.5 (co_d7_free_ion_output_kiso_half.txt)",
        ),
    ),
)
def test_calculation_of_paramagnetic_susceptibility(
    parameters: Parameters,
    foz_soz_limit: tuple[int, int] | None,
    expected_susceptibility: float,
) -> None:
    results = run_parameters(
        parameters=parameters, calculate_configuration_projections=False
    )
    if foz_soz_limit is None:
        foz_limit = (
            max(e.index for e in results.eigenfunctions.levels[0].eigenfunctions) + 1
        )
        soz_limit = foz_limit
    else:
        foz_limit, soz_limit = foz_soz_limit
    susceptibilities = calculate_paramagnetic_susceptibilities(
        muxyz=results.get_magnetic_dipole_moments(),
        energies=np.array([e.energy for e in results.eigenfunctions.eigenfunctions]),
        temperatures=np.array([300.0]),
        first_order_zeeman_max_index=foz_limit,
        second_order_zeeman_max_index=soz_limit,
    )
    value = np.round(np.average(susceptibilities.tensors[0].values) * 1e6, decimals=0)
    assert (
        abs(value - expected_susceptibility) / abs(expected_susceptibility) < 0.005
    )  # 0.5 %
