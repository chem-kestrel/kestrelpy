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

from kestrelpy.calculations.interelectronic_repulsion import InterelectronRepulsion
from kestrelpy.calculations.ligand_field_matrix import MultipoleExpansion
from kestrelpy.calculations.many_electron_matrices import NonZeroMatrixElements
from kestrelpy.calculations.operators import Operators
from kestrelpy.calculations.parameters import (
    MagneticField,
    OrbitalReductionFactors,
    SpinOrbitCoupling,
)
from kestrelpy.calculations.utils import get_many_d_electron_basis_size

Coefficients: TypeAlias = (
    MultipoleExpansion | InterelectronRepulsion | SpinOrbitCoupling | MagneticField
)


class ConstructEffectiveHamiltonian:
    """Construct the Effective Hamiltonian.

    Attributes:
        _d_electrons (int): The number of d_electrons. Used for optimisations.
        _matrices (NonZeroMatrixElements): Contains the matrix indices
            and values, where the values are non-zero.
    """

    EPSILON = 1.0e-5

    def __init__(self, d_electrons: int, matrices: NonZeroMatrixElements):
        self._d_electrons: int = d_electrons
        self._size: int = get_many_d_electron_basis_size(d_electrons)
        self._matrices: NonZeroMatrixElements = matrices

    def construct(
        self,
        multipole_expansion: MultipoleExpansion,
        interelectronic_repulsion: InterelectronRepulsion,
        spin_orbit_coupling: SpinOrbitCoupling,
        magnetic_field: MagneticField,
        orbital_reduction_factors: OrbitalReductionFactors,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct the Effective Hamiltonian.

        Constructs a Hamiltonian of the form:

            >>> H_eff = H_LFT + H_IR + H_SOC + H_MAG

                where:
                    H_LFT = ligand field
                    H_IR = interelectronic repulsion
                    H_SOC = spin-orbit coupling
                    H_MAG = external magnetic field perturbation

        Args:
            multipole_expansion (MultipoleExpansion)
            interelectronic_repulsion (InterelectronRepulsion)
            spin_orbit_coupling (SpinOrbitCoupling)
            magnetic_field (MagneticField)
            orbital_reduction_factors (OrbitalReductionFactors)

        Returns:
            full_hamiltonian_matrix,
            ligand_field_only_matrix,
            full_hamiltonian_matrix_nosoc
        """
        matrix = np.zeros((self._size, self._size), dtype=complex)
        matrix = self._add_ligand_field(
            matrix=matrix, multipole_expansion=multipole_expansion
        )
        ligand_field_matrix = np.copy(matrix)
        self._add_interelectronic_repulsion(
            matrix=matrix, interelectronic_repulsion=interelectronic_repulsion
        )
        self.add_magnetic_field_contribution(
            matrix=matrix,
            magnetic_field=magnetic_field,
            orbital_reduction_factors=orbital_reduction_factors,
        )
        no_soc = np.copy(matrix)
        self._add_spin_orbit_coupling(
            matrix=matrix, spin_orbit_coupling=spin_orbit_coupling
        )

        return matrix, ligand_field_matrix, no_soc

    def _add_to_matrix(
        self,
        matrix: np.ndarray,
        coefficients: Coefficients,
        operators: type[Operators.InterelectronicRepulsion]
        | type[Operators.SphericalHarmonic]
        | type[Operators.SpinOrbitCoupling],
    ) -> np.ndarray:
        for operator in operators:
            coefficient = getattr(coefficients, operator)
            indices: np.ndarray = self._matrices.indices.get(operator)
            values: np.ndarray = self._matrices.values.get(operator)
            matrix[(indices[:, 0], indices[:, 1])] += coefficient * values
        return matrix

    def _add_ligand_field(
        self, matrix: np.ndarray, multipole_expansion: MultipoleExpansion
    ) -> np.ndarray:
        return self._add_to_matrix(
            matrix, multipole_expansion, Operators.SphericalHarmonic
        )

    def _add_interelectronic_repulsion(
        self, matrix: np.ndarray, interelectronic_repulsion: InterelectronRepulsion
    ) -> np.ndarray:
        if self._d_electrons == 1 or self._d_electrons == 9:
            return matrix
        return self._add_to_matrix(
            matrix, interelectronic_repulsion, Operators.InterelectronicRepulsion
        )

    def _add_spin_orbit_coupling(
        self, matrix: np.ndarray, spin_orbit_coupling: SpinOrbitCoupling
    ) -> np.ndarray:
        if abs(spin_orbit_coupling.SOC) < self.EPSILON:
            return matrix
        return self._add_to_matrix(
            matrix, spin_orbit_coupling, Operators.SpinOrbitCoupling
        )

    def add_magnetic_field_contribution(
        self,
        matrix: np.ndarray,
        magnetic_field: MagneticField,
        orbital_reduction_factors: OrbitalReductionFactors,
    ) -> np.ndarray:
        if self._is_magnetic_field_zero(
            magnetic_field
        ) or self._are_orbital_reduction_factors_zero(orbital_reduction_factors):
            return matrix
        for mag_field, orbital_momentum, spin_momentum, reduction_factor in zip(
            [magnetic_field.BX, magnetic_field.BY, magnetic_field.BZ],
            [
                Operators.OrbitalAngularMomentum.LX,
                Operators.OrbitalAngularMomentum.LY,
                Operators.OrbitalAngularMomentum.LZ,
            ],
            [
                Operators.SpinAngularMomentum.SX,
                Operators.SpinAngularMomentum.SY,
                Operators.SpinAngularMomentum.SZ,
            ],
            [
                orbital_reduction_factors.KX,
                orbital_reduction_factors.KY,
                orbital_reduction_factors.KZ,
            ],
        ):
            coefficient = (
                complex(0.0, mag_field)
                if orbital_momentum == Operators.OrbitalAngularMomentum.LY
                else complex(mag_field, 0.0)
            )
            # Orbital angular momentum contribution
            indices = self._matrices.indices[orbital_momentum]
            values = (
                self._matrices.values[orbital_momentum] * coefficient * reduction_factor
            )
            matrix[(indices[:, 0], indices[:, 1])] += values
            # Spin angular momentum contribution
            indices = self._matrices.indices[spin_momentum]
            values = self._matrices.values[spin_momentum] * coefficient
            matrix[(indices[:, 0], indices[:, 1])] += values
        return matrix

    def _is_magnetic_field_zero(self, magnetic_field: MagneticField) -> bool:
        return all(
            [
                abs(magnetic_field.BX) < self.EPSILON,
                abs(magnetic_field.BY) < self.EPSILON,
                abs(magnetic_field.BZ) < self.EPSILON,
            ]
        )

    def _are_orbital_reduction_factors_zero(
        self, orbital_reduction_factors: OrbitalReductionFactors
    ) -> bool:
        return all(
            [
                abs(orbital_reduction_factors.KX) < self.EPSILON,
                abs(orbital_reduction_factors.KY) < self.EPSILON,
                abs(orbital_reduction_factors.KZ) < self.EPSILON,
            ]
        )
