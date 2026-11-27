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

from unittest.mock import Mock

import numpy as np
import pytest

from kestrelpy.calculations.ligand_field_matrix import LigandField
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_parameters import (
    calculate_4s_kappa_value,
    calculate_kappa_free_ion,
    calculate_kappa_from_fermi,
    calculate_max_4s_orbital_admixtures,
    calculate_p,
    estimate_kappa_admixture,
    estimate_kappa_polarisation,
    estimate_ligand_field_trace,
)
from kestrelpy.calculations.many_electron_eigenfunctions.configuration_projection import (
    Configuration,
)


@pytest.mark.parametrize(
    "metal, oxidation_state, p_value",
    [("63Cu", 2, 1168), ("51V", 4, 515), ("57Fe", 3, 98)],
)
def test_calculate_p(metal: str, oxidation_state: int, p_value: int) -> None:
    """Test P is correctly calculated for a range of metals and oxidation states"""
    output = np.round(calculate_p(metal, oxidation_state), 0)
    assert output == p_value


@pytest.fixture
def dummy_configurations() -> Configuration:
    """Returns parameters for a standard mock complex"""
    return Configuration(occupation=[0], contribution=1.0, spin_multiplicity=2)  # type: ignore


@pytest.fixture
def mock_ligand_field() -> LigandField:
    return LigandField(
        ligand_field_matrix=np.diag([9618.0, 1800.0, -488.0, 1800.0, 9618.0]),
        eigenvalues=np.array([0.0, 2287.0, 2287.0, 10106.0, 10106.0]),
        eigenvectors=np.array(
            [
                (0.0, 0.0, 0.0, 0.0, -1.0),
                (0.0, 0.0, -1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, -1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, -1.0, 0.0),
            ]
        ),
        zero_point_energy=0.0,
        multipole_expansion=Mock(),
    )


class TestKappaCalculations:
    @pytest.mark.parametrize(
        "metal, oxidation_state, kappa_value",
        [
            ("Cu", 2, 0.2410),
            ("V", 4, 0.3912),
            ("Co", 2, 0.2972),
        ],
    )
    def test_calculate_kappa_free_ion(
        self, metal: str, oxidation_state: int, kappa_value: float
    ) -> None:
        """
        Test that free-ion kappa values are correctly calculated for a range of
        metals and oxidation states
        """
        kappa_free_ion = calculate_kappa_free_ion(metal, oxidation_state)
        assert kappa_free_ion == kappa_value

    @pytest.mark.parametrize(
        "ligand_field, expected_kappa_value",
        [
            (23800.0, -0.0289),
            (50200.0, 0.0962),
            (17150.0, -0.0604),
        ],
    )
    def test_estimate_kappa_polarisation(
        self, ligand_field: float, expected_kappa_value: float
    ) -> None:
        """
        Test that the kappa value for a given complex is correctly calculated from the metal, oxidation
        state and the sum of e values (ligand field trace)
        """
        kappa_polarisation = estimate_kappa_polarisation(
            ligand_field_trace=ligand_field
        )
        assert kappa_polarisation == expected_kappa_value

    @pytest.mark.parametrize(
        "admixture, s_fermi, unpaired_electrons, p, expected_kappa",
        [
            pytest.param(0.8, 5036.0, 5, 560.0, -1.4389, id="Mn2+ HS complex"),
            pytest.param(0.5, 4165.0, 3, 386.0, -1.7984, id="V2+ HS complex"),
            pytest.param(0.05, 5995.0, 1, 1168.0, -0.2566, id="Cu2+ complex"),
            pytest.param(0.0, 5995.0, 1, 1168.0, 0.0, id="No admixture expected"),
        ],
    )
    def test_calculate_4s_kappa_value(
        self,
        admixture: float,
        s_fermi: float,
        unpaired_electrons: int,
        p: float,
        expected_kappa: float,
    ) -> None:
        """
        Test the kappa value for the 4s admixture is correctly calculated given the
        sum of e sigma minus e pi and dz2 contribution to singly occupied orbitals.
        """
        kappa_admixture = calculate_4s_kappa_value(
            admixture=admixture,
            unpaired_electrons=unpaired_electrons,
            p=p,
            s_fermi=s_fermi,
        )
        assert kappa_admixture == expected_kappa

    @pytest.mark.parametrize(
        "lf_eigenvalues, e_sigma_minus_pi, max_admixture",
        [
            ([0.0, 1000.0, 1000.0, 8000.0, 16000.0], 8000.0, 29.36),
            ([0.0, 0.0, 0.0, 8000.0, 8000.0], -35000.0, 100.0),
            ([0.0, 5000.0, 8000.0, 12000.0, 20000.0], 25000.0, 0.0),
        ],
    )
    def test_calculate_max_4s_orbital_admixture(
        self, lf_eigenvalues: np.ndarray, e_sigma_minus_pi: float, max_admixture: float
    ) -> None:
        """
        Test the maximum 4s admixture, ignoring any symmetry constraints,
        is correctly calculated for a range of lf energies and parameter values.
        """
        theoretical_admixture = calculate_max_4s_orbital_admixtures(
            e_sigma_minus_e_pi=e_sigma_minus_pi
        )
        assert np.round(theoretical_admixture[0], 2) == max_admixture

    @pytest.mark.parametrize(
        "sigma_minus_pi, electrons, p, s_fermi, expected_kappa, expected_admixture",
        [
            (8000.0, 1, 1068.0, 5995.0, -1.6481, 0.2936),
            (-2000.0, 1, 515.0, 4165.0, -3.9758, 0.4916),
            (25000.0, 1, 1068.0, 5995.0, -0.0, 0.0),
        ],
    )
    def test_estimate_kappa_admixture(
        self,
        sigma_minus_pi: float,
        electrons: int,
        p: float,
        s_fermi: float,
        expected_kappa: float,
        expected_admixture: float,
        dummy_configurations: Configuration,
        mock_ligand_field: LigandField,
    ) -> None:
        """
        Test that the kappa value for the 4s admixture contribution to the
        isotropic Fermi contact is correctly calculated.
        """
        kappa, s_admixture = estimate_kappa_admixture(
            e_sigma_minus_e_pi=sigma_minus_pi,
            level_configurations=[dummy_configurations],
            ligand_field=mock_ligand_field,
            unpaired_electrons=electrons,
            p=p,
            s_fermi=s_fermi,
        )
        assert kappa == expected_kappa
        assert round(np.sum(s_admixture), 4) == expected_admixture

    @pytest.mark.parametrize(
        "metal, oxidation_state, kappa, ligand_field",
        [
            ("Cu", 2, 0.2122, 23823),
            ("V", 4, 0.4888, 50494),
            ("Co", 2, 0.2366, 17113),
        ],
    )
    def test_estimate_ligand_field_trace(
        self, metal: str, oxidation_state: int, kappa: float, ligand_field: float
    ) -> None:
        """
        Test that the sum of e values (ligand field trace) is correctly calculated, given a metal,
        oxidation state and kappa value
        """
        ligand_trace = estimate_ligand_field_trace(metal, oxidation_state, kappa)
        assert ligand_trace == ligand_field

    @pytest.mark.parametrize(
        "metal, oxidation_state, fermi, kappa",
        [
            ("63Cu", 2, 270.0, 0.2312),
            ("59Co", 2, 150.0, 0.1966),
            ("51V", 4, 400.0, 0.7767),
        ],
    )
    def test_calculate_kappa_from_fermi(
        self, metal: str, oxidation_state: int, fermi: float, kappa: float
    ) -> None:
        """Test that a fermi contact value can be converted into a kappa value"""
        calculated_kappa = calculate_kappa_from_fermi(metal, oxidation_state, fermi)
        assert calculated_kappa == kappa
