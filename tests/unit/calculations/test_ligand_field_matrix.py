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

from cmath import isclose

import numpy as np
import pytest

from kestrelpy.calculations.ligand_field_matrix import (
    DOrbitalBasis,
    MultipoleExpansion,
    _get_inverse_multiple_expansion,
    get_multipole_expansion,
    rotate_lf_frame,
)

DXY = DOrbitalBasis.DXY.value
DXZ = DOrbitalBasis.DXZ.value
DZ2 = DOrbitalBasis.DZ2.value
DYZ = DOrbitalBasis.DYZ.value
DX2Y2 = DOrbitalBasis.DX2Y2.value
ZERO_COMPLEX = complex(0.0, 0.0)


def get_symmetric_matrix(elements: list[tuple[int, int, float]]) -> np.ndarray:
    """Return a symmetric matrix from a list of indices and matrix elements.

    Args:
        elements: index_i, index_j, value - will automatically populate
            matrix_ji = matrix_ij.

    Returns:
        np.ndarray
    """
    matrix = np.zeros((5, 5), dtype=float)
    for element in elements:
        index_i, index_j, value = element
        matrix[index_i, index_j] = value
        matrix[index_j, index_i] = value
    return matrix


def test_rotate_lf_frame() -> None:
    """
    Test whether a 5x5 Ligand Field matrix is correctly rotated to the
    frame given by a set of vectors, i.e. the g tensor (magnetic) frame.
    As this is a unitary transformation, the eigenvalues should be unchanged.
    """
    g_vectors = np.array(
        ([0.997, -0.034, 0.069], [0.04, 0.996, -0.078], [0.066, -0.08, -0.995])
    )
    lf_matrix = np.array(
        [
            [1130.4, 173.0, 85.1, -118.3, 1047.0],
            [173.0, 339.4, 568.2, -56.9, 1130.9],
            [85.1, 568.2, 3676.1, -537.7, -488.0],
            [-118.3, -56.9, -537.7, 1789.5, 1009.6],
            [1047.0, 1130.9, -488.0, 1009.6, 16864.6],
        ]
    )
    eigenvalues = np.array([[152.06, 1017.88, 1638.78, 3897.91, 17093.37]])
    expected_matrix = np.array(
        [
            [1103.8, 34.0, 208.8, -214.5, -145.4],
            [34.0, 161.0, 130.4, -67.1, -177.1],
            [208.8, 130.4, 3865.3, -242.7, -468.9],
            [-214.5, -67.1, -242.7, 1596.7, -48.5],
            [-145.4, -177.1, -468.9, -48.5, 17073.2],
        ]
    )

    rotated_matrix = rotate_lf_frame(ref_vectors=g_vectors, lf_matrix=lf_matrix)
    assert np.allclose(rotated_matrix.ligand_field_matrix, expected_matrix, rtol=2e-3)
    assert np.allclose(rotated_matrix.eigenvalues, eigenvalues, rtol=1e-5)


class TestMultipoleExpansionLigandField:
    """Testing the calculations of the multipole expansion in a ligand-field context."""

    def test_multipole_expansion_is_generated_correctly(self) -> None:
        esig, epix, epiy = 1000.0, 500.0, 300.0
        expected_multipole_expansion = self._generate_expected_multipole_expansion(
            esig=esig, epix=epix, epiy=epiy
        )
        multiple_expansion = get_multipole_expansion(
            self._generate_ligand_field_matrix(esig=esig, epix=epix, epiy=epiy)
        )
        assert isclose(multiple_expansion.Y_2_0, expected_multipole_expansion.Y_2_0)
        assert isclose(multiple_expansion.Y_2_2, expected_multipole_expansion.Y_2_2)
        assert isclose(multiple_expansion.Y_2_n2, expected_multipole_expansion.Y_2_n2)
        assert isclose(multiple_expansion.Y_4_0, expected_multipole_expansion.Y_4_0)
        assert isclose(multiple_expansion.Y_4_2, expected_multipole_expansion.Y_4_2)
        assert isclose(multiple_expansion.Y_4_n2, expected_multipole_expansion.Y_4_n2)

    def _generate_expected_multipole_expansion(
        self, esig: float, epix: float, epiy: float
    ) -> MultipoleExpansion:
        Y_2_0 = complex(np.sqrt(np.pi / 5.0) * (2.0 * esig + epix + epiy), 0.0)
        Y_2_2 = complex(np.sqrt(0.3 * np.pi) * (epix - epiy), 0.0)
        Y_4_0 = complex(
            0.4 * np.sqrt(np.pi) * (3.0 * esig - 2.0 * epix - 2.0 * epiy), 0.0
        )
        Y_4_2 = complex(np.sqrt(0.4 * np.pi) * (epix - epiy), 0.0)
        return MultipoleExpansion(
            Y_2_2=Y_2_2,
            Y_2_1=ZERO_COMPLEX,
            Y_2_0=Y_2_0,
            Y_2_n1=ZERO_COMPLEX,
            Y_2_n2=_get_inverse_multiple_expansion(Y_2_2, 2),
            Y_4_4=ZERO_COMPLEX,
            Y_4_3=ZERO_COMPLEX,
            Y_4_2=Y_4_2,
            Y_4_1=ZERO_COMPLEX,
            Y_4_0=Y_4_0,
            Y_4_n1=ZERO_COMPLEX,
            Y_4_n2=_get_inverse_multiple_expansion(Y_4_2, 2),
            Y_4_n3=ZERO_COMPLEX,
            Y_4_n4=ZERO_COMPLEX,
        )

    def _generate_ligand_field_matrix(
        self, esig: float, epix: float, epiy: float
    ) -> np.ndarray:
        return get_symmetric_matrix(
            [(DZ2, DZ2, esig), (DXZ, DXZ, epix), (DYZ, DYZ, epiy)]
        )


class TestMultipoleExpansionTransitionDipoleMomentX:
    """
    Test that multipole expansion elements are calculated correctly
    for the x-transition-dipole-moment contribution.
    """

    def test_multipole_expansion_is_generated_correctly(self) -> None:
        psig, ppix, ppiy = 1000.0, 500.0, 300.0
        fsig, fpix, fpiy = 1000.0, 500.0, 300.0
        expected_multipole_expansion = self._generate_expected_multipole_expansion(
            psig=psig, ppix=ppix, ppiy=ppiy, fsig=fsig, fpix=fpix, fpiy=fpiy
        )
        multiple_expansion = get_multipole_expansion(
            self._generate_ligand_field_matrix(
                psig=psig, ppix=ppix, ppiy=ppiy, fsig=fsig, fpix=fpix, fpiy=fpiy
            )
        )
        assert isclose(multiple_expansion.Y_2_1, expected_multipole_expansion.Y_2_1)
        assert isclose(multiple_expansion.Y_2_n1, expected_multipole_expansion.Y_2_n1)
        assert isclose(multiple_expansion.Y_4_3, expected_multipole_expansion.Y_4_3)
        assert isclose(multiple_expansion.Y_4_n3, expected_multipole_expansion.Y_4_n3)
        assert isclose(multiple_expansion.Y_4_1, expected_multipole_expansion.Y_4_1)
        assert isclose(multiple_expansion.Y_4_n1, expected_multipole_expansion.Y_4_n1)

    def _generate_expected_multipole_expansion(
        self,
        psig: float,
        ppix: float,
        ppiy: float,
        fsig: float,
        fpix: float,
        fpiy: float,
    ) -> MultipoleExpansion:
        """Generate the expected multipole expansion elements from p and f parameters.

        The expressions for the multipole expansion elements are taken from table 3.5:
            >>> Caroline A. Brown , Malcolm Gerloch & Robert F. McMeeking (1988) A cellular
                ligand-field model for 'l-l' spectral intensities, Molecular Physics: An
                International Journal at the Interface Between Chemistry and Physics,
                64:5, 771-792, DOI: 10.1080/00268978800100553
        """
        Y_2_1 = (
            # PTx contribution
            complex(
                -np.sqrt(np.pi / 30.0) * (4.0 * ppix + 6.0 * ppiy + 3.0 * psig), 0.0
            )
            # FTx contribution
            + complex(
                -0.5 * np.sqrt(np.pi / 30.0) * (3.0 * fpix - 3.0 * fpiy - 4.0 * fsig),
                0.0,
            )
        )
        Y_4_1 = (
            # PTx contribution
            complex(-np.sqrt(np.pi / 5.0) * (3.0 * psig - 3.0 * ppix - ppiy), 0.0)
            # FTx contribution
            + complex(
                0.25 * np.sqrt(np.pi / 5.0) * (8.0 * fsig - 13.0 * fpix - fpiy), 0.0
            )
        )
        Y_4_3 = (
            # PTx contribution
            complex(np.sqrt(np.pi * 1.4) * (ppiy - ppix), 0.0)
            # FTx contribution
            + complex(-0.25 * np.sqrt(np.pi * 1.4) * (fpiy - fpix), 0.0)
        )
        return MultipoleExpansion(
            Y_2_2=ZERO_COMPLEX,
            Y_2_1=Y_2_1,
            Y_2_0=ZERO_COMPLEX,
            Y_2_n1=_get_inverse_multiple_expansion(Y_2_1, 1),
            Y_2_n2=ZERO_COMPLEX,
            Y_4_4=ZERO_COMPLEX,
            Y_4_3=Y_4_3,
            Y_4_2=ZERO_COMPLEX,
            Y_4_1=Y_4_1,
            Y_4_0=ZERO_COMPLEX,
            Y_4_n1=_get_inverse_multiple_expansion(Y_4_1, 1),
            Y_4_n2=ZERO_COMPLEX,
            Y_4_n3=_get_inverse_multiple_expansion(Y_4_3, 3),
            Y_4_n4=ZERO_COMPLEX,
        )

    def _generate_ligand_field_matrix(
        self,
        psig: float,
        ppix: float,
        ppiy: float,
        fsig: float,
        fpix: float,
        fpiy: float,
    ) -> np.ndarray:
        """Generate a 5x5 matrix to generate a multipole expansion from.

        Equations in terms of p and f parameters are given in Table 10 of:
            >>> Bridgeman, A. J.; Gerloch, M. Spectral Intensities of
                Transition Metal Complexes.Coordination Chemistry Reviews,
                1997, 165, 315-446. https://doi.org/10.1016/s0010-8545(97)90163-6.
        """
        return get_symmetric_matrix(
            [
                (
                    DXZ,
                    DZ2,
                    np.sqrt(1.0 / 3.0) * ((1.5 * psig - ppix) + (1.5 * fpix - fsig)),
                ),
                (DX2Y2, DXZ, ppix - 0.25 * fpix),
                (DXY, DYZ, ppiy - 0.25 * fpiy),
            ]
        )


class TestMultipoleExpansionTransitionDipoleMomentY:
    """
    Test that multipole expansion elements are calculated correctly
    for the y-transition-dipole-moment contribution.
    """

    def test_multipole_expansion_is_generated_correctly(self) -> None:
        psig, ppix, ppiy = 1000.0, 500.0, 300.0
        fsig, fpix, fpiy = 1000.0, 500.0, 300.0
        expected_multipole_expansion = self._generate_expected_multipole_expansion(
            psig=psig, ppix=ppix, ppiy=ppiy, fsig=fsig, fpix=fpix, fpiy=fpiy
        )
        multiple_expansion = get_multipole_expansion(
            self._generate_ligand_field_matrix(
                psig=psig, ppix=ppix, ppiy=ppiy, fsig=fsig, fpix=fpix, fpiy=fpiy
            )
        )
        assert isclose(multiple_expansion.Y_2_1, expected_multipole_expansion.Y_2_1)
        assert isclose(multiple_expansion.Y_2_n1, expected_multipole_expansion.Y_2_n1)
        assert isclose(multiple_expansion.Y_4_3, expected_multipole_expansion.Y_4_3)
        assert isclose(multiple_expansion.Y_4_n3, expected_multipole_expansion.Y_4_n3)
        assert isclose(multiple_expansion.Y_4_1, expected_multipole_expansion.Y_4_1)
        assert isclose(multiple_expansion.Y_4_n1, expected_multipole_expansion.Y_4_n1)

    def _generate_expected_multipole_expansion(
        self,
        psig: float,
        ppix: float,
        ppiy: float,
        fsig: float,
        fpix: float,
        fpiy: float,
    ) -> MultipoleExpansion:
        """Generate the expected multipole expansion elements from p and f parameters.

        The expressions for the multipole expansion elements are taken from table 3.5:
            >>> Caroline A. Brown , Malcolm Gerloch & Robert F. McMeeking (1988) A cellular
                ligand-field model for 'l-l' spectral intensities, Molecular Physics: An
                International Journal at the Interface Between Chemistry and Physics,
                64:5, 771-792, DOI: 10.1080/00268978800100553
        """
        Y_2_1 = (
            # PTy contribution
            complex(0.0, np.sqrt(np.pi / 30.0) * (4.0 * ppiy + 6.0 * ppix + 3.0 * psig))
            # FTy contribution
            + complex(
                0.0,
                0.5 * np.sqrt(np.pi / 30.0) * (3.0 * fpiy - 3.0 * fpix - 4.0 * fsig),
            )
        )
        Y_4_1 = (
            # PTy contribution
            complex(0.0, np.sqrt(np.pi / 5.0) * (3.0 * psig - 3.0 * ppiy - ppix))
            # FTy contribution
            + complex(
                0.0,
                -0.25 * np.sqrt(np.pi / 5.0) * (8.0 * fsig - 13.0 * fpiy - fpix),
            )
        )
        Y_4_3 = (
            # PTy contribution
            complex(0.0, -np.sqrt(np.pi * 1.4) * (ppiy - ppix))
            # FTy contribution
            + complex(0.0, 0.25 * np.sqrt(np.pi * 1.4) * (fpiy - fpix))
        )
        return MultipoleExpansion(
            Y_2_2=ZERO_COMPLEX,
            Y_2_1=Y_2_1,
            Y_2_0=ZERO_COMPLEX,
            Y_2_n1=_get_inverse_multiple_expansion(Y_2_1, 1),
            Y_2_n2=ZERO_COMPLEX,
            Y_4_4=ZERO_COMPLEX,
            Y_4_3=Y_4_3,
            Y_4_2=ZERO_COMPLEX,
            Y_4_1=Y_4_1,
            Y_4_0=ZERO_COMPLEX,
            Y_4_n1=_get_inverse_multiple_expansion(Y_4_1, 1),
            Y_4_n2=ZERO_COMPLEX,
            Y_4_n3=_get_inverse_multiple_expansion(Y_4_3, 3),
            Y_4_n4=ZERO_COMPLEX,
        )

    def _generate_ligand_field_matrix(
        self,
        psig: float,
        ppix: float,
        ppiy: float,
        fsig: float,
        fpix: float,
        fpiy: float,
    ) -> np.ndarray:
        """Generate a 5x5 matrix to generate a multipole expansion from.

        Equations in terms of p and f parameters are given in Table 10 of:
            >>> Bridgeman, A. J.; Gerloch, M. Spectral Intensities of
                Transition Metal Complexes.Coordination Chemistry Reviews,
                1997, 165, 315-446. https://doi.org/10.1016/s0010-8545(97)90163-6.
        """
        return get_symmetric_matrix(
            [
                (
                    DYZ,
                    DZ2,
                    np.sqrt(1.0 / 3.0) * ((1.5 * psig - ppiy) + (1.5 * fpiy - fsig)),
                ),
                (DX2Y2, DYZ, -ppiy + 0.25 * fpiy),
                (DXY, DXZ, ppix - 0.25 * fpix),
            ]
        )


class TestMultipoleExpansionTransitionDipoleMomentZ:
    """
    Test that multipole expansion elements are calculated correctly
    for the z-transition-dipole-moment contribution.
    """

    def test_multipole_expansion_is_generated_correctly(self) -> None:
        psig, ppix, ppiy = 1000.0, 500.0, 300.0
        fsig, fpix, fpiy = 1000.0, 500.0, 300.0
        expected_multipole_expansion = self._generate_expected_multipole_expansion(
            psig=psig, ppix=ppix, ppiy=ppiy, fsig=fsig, fpix=fpix, fpiy=fpiy
        )
        multiple_expansion = get_multipole_expansion(
            self._generate_ligand_field_matrix(
                psig=psig, ppix=ppix, ppiy=ppiy, fsig=fsig, fpix=fpix, fpiy=fpiy
            )
        )
        assert isclose(multiple_expansion.Y_2_2, expected_multipole_expansion.Y_2_2)
        assert isclose(multiple_expansion.Y_2_n2, expected_multipole_expansion.Y_2_n2)
        assert isclose(multiple_expansion.Y_2_0, expected_multipole_expansion.Y_2_0)
        assert isclose(multiple_expansion.Y_4_2, expected_multipole_expansion.Y_4_2)
        assert isclose(multiple_expansion.Y_4_n2, expected_multipole_expansion.Y_4_n2)
        assert isclose(multiple_expansion.Y_4_0, expected_multipole_expansion.Y_4_0)

    def _generate_expected_multipole_expansion(
        self,
        psig: float,
        ppix: float,
        ppiy: float,
        fsig: float,
        fpix: float,
        fpiy: float,
    ) -> MultipoleExpansion:
        """Generate the expected multipole expansion elements from p and f parameters.

        The expressions for the multipole expansion elements are taken from table 3.5:
            >>> Caroline A. Brown , Malcolm Gerloch & Robert F. McMeeking (1988) A cellular
                ligand-field model for 'l-l' spectral intensities, Molecular Physics: An
                International Journal at the Interface Between Chemistry and Physics,
                64:5, 771-792, DOI: 10.1080/00268978800100553
        """
        Y_2_0 = complex(
            2.0
            * np.sqrt(np.pi * 0.2)
            * ((2.0 * psig + ppix + ppiy) + (2.0 * fsig + fpix + fpiy)),
            0.0,
        )
        Y_2_2 = complex(
            2.0 * np.sqrt(0.3 * np.pi) * ((ppix - ppiy) + (fpix - fpiy)), 0.0
        )
        Y_4_0 = complex(
            0.8
            * np.sqrt(np.pi)
            * (
                (3.0 * psig - 2.0 * ppix - 2.0 * ppiy)
                + (3.0 * fsig - 2.0 * fpix - 2.0 * fpiy)
            ),
            0.0,
        )
        Y_4_2 = complex(
            2.0 * np.sqrt(0.4 * np.pi) * ((ppix - ppiy) + (fpix - fpiy)), 0.0
        )
        return MultipoleExpansion(
            Y_2_2=Y_2_2,
            Y_2_1=ZERO_COMPLEX,
            Y_2_0=Y_2_0,
            Y_2_n1=ZERO_COMPLEX,
            Y_2_n2=_get_inverse_multiple_expansion(Y_2_2, 2),
            Y_4_4=ZERO_COMPLEX,
            Y_4_3=ZERO_COMPLEX,
            Y_4_2=Y_4_2,
            Y_4_1=ZERO_COMPLEX,
            Y_4_0=Y_4_0,
            Y_4_n1=ZERO_COMPLEX,
            Y_4_n2=_get_inverse_multiple_expansion(Y_4_2, 2),
            Y_4_n3=ZERO_COMPLEX,
            Y_4_n4=ZERO_COMPLEX,
        )

    def _generate_ligand_field_matrix(
        self,
        psig: float,
        ppix: float,
        ppiy: float,
        fsig: float,
        fpix: float,
        fpiy: float,
    ) -> np.ndarray:
        """Generate a 5x5 matrix to generate a multipole expansion from.

        Equations in terms of p and f parameters are given in Table 10 of:
            >>> Bridgeman, A. J.; Gerloch, M. Spectral Intensities of
                Transition Metal Complexes.Coordination Chemistry Reviews,
                1997, 165, 315-446. https://doi.org/10.1016/s0010-8545(97)90163-6.
        """
        return get_symmetric_matrix(
            [
                (DZ2, DZ2, 2.0 * psig + 2.0 * fsig),
                (DXZ, DXZ, 2.0 * ppix + 2.0 * fpix),
                (DYZ, DYZ, 2.0 * ppiy + 2.0 * fpiy),
            ]
        )


class TestInverseMultipoleExpansionElements:
    """Test the _get_inverse_multipole_expansion function."""

    @pytest.mark.parametrize("q", [1, 3, 5, 7, 9])
    def test_inverse_multipole_expansions_are_calculated_correctly(
        self, q: int
    ) -> None:
        test_multipole_expansion = complex(1.0, 2.0)
        expected_inverse_multipole_expansion_odd_order = complex(-1.0, 2.0)
        expected_inverse_multipole_expansion_even_order = complex(1.0, -2.0)

        assert (
            _get_inverse_multiple_expansion(test_multipole_expansion, q)
            == expected_inverse_multipole_expansion_odd_order
        )
        assert (
            _get_inverse_multiple_expansion(test_multipole_expansion, q - 1)
            == expected_inverse_multipole_expansion_even_order
        )
