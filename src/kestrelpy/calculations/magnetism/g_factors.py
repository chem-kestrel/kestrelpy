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

=========================================================================

Calculate g-factors in the ligand field framework.

Calculations are implementing equation 8 from reference:
    A. Bencini, C. Benelli, D. Gatteschi,
    The angular overlap model for the description of the paramagnetic properties of transition metal complexes,
    Coordination Chemistry Reviews,
    Volume 60,
    1984,
    Pages 131-169,
    ISSN 0010-8545,
    https://doi.org/10.1016/0010-8545(84)85064-X.
    (https://www.sciencedirect.com/science/article/pii/001085458485064X)
"""

from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from kestrelpy.calculations.rotation_matrix import ensure_right_handed_axis
from kestrelpy.calculations.solve_matrix import solve_matrix
from kestrelpy.configuration import get_logger

G_E = 2.002319

_LOGGER = get_logger(__name__)


@dataclass
class GSquaredTensorMetadata:
    eigenfunction_start_index: int
    eigenfunction_final_index: int


@dataclass
class GSquaredTensor:
    tensor: np.ndarray
    values: np.ndarray
    vectors: np.ndarray
    metadata: GSquaredTensorMetadata

    def __str__(self) -> str:
        start = self.metadata.eigenfunction_start_index
        final = self.metadata.eigenfunction_final_index
        vectors = self.vectors.real + np.abs(self.vectors.imag)
        return "\n".join(
            [
                f"G-factors for Eigenfunctions: {start} -> {final}",
                tabulate(
                    tabular_data=[[self.values[i], *vectors[:, i]] for i in range(3)],
                    headers=["g-factors", "x", "y", "z"],
                    floatfmt=(".4f", ".3f", ".3f", ".3f"),
                    tablefmt="outline",
                ),
            ]
        )

    @classmethod
    def zero_g_tensor(cls) -> "GSquaredTensor":
        """Return g tensor with NaN for systems which are EPR silent"""
        return cls(
            tensor=np.full((3, 3), np.nan),
            values=np.full(3, np.nan),
            vectors=np.full((3, 3), np.nan),
            metadata=GSquaredTensorMetadata(
                eigenfunction_start_index=0, eigenfunction_final_index=0
            ),
        )

    def get_gxyz(self) -> np.ndarray:
        """Return the g-factors projected along the x, y, and z directions.

        Returns:
            gx, gy, gz.
        """
        max_projection = np.argmax(self.vectors * self.vectors.conjugate(), axis=1)
        return self.values[max_projection]

    def order_g_factors(self) -> None:
        """Return g values sorted by Δg to give g3 as g value most deviated from ge"""
        order = get_g_factor_order(
            values=self.values,
        )
        vectors = self.vectors[:, order]
        self.values = self.values[order]
        self.vectors = ensure_right_handed_axis(vectors=vectors)

    def apply_g_order(self, order: np.ndarray) -> None:
        """Apply a given g ordering to the GSquaredTensor object"""
        values = self.values[order]
        vectors = self.vectors[:, order]

        self.values = values
        self.vectors = ensure_right_handed_axis(vectors=vectors)


def get_g_factor_order(values: np.ndarray) -> np.ndarray:
    """
    Order g values so that g1=gx, g2=gy, g3=gz using the convention that::

    |g_zz - g_e| >= |g_xx - g_e| >= |g_yy - g_e|

    Args:
        values: g values in ascending order.

    Return:
        xyz_index
    """
    g_sorted = np.sort(values)
    if g_sorted[0] < G_E < g_sorted[2]:
        return np.argsort(values)

    if (values.max() - values.min()) < 1e-2:
        return np.array([0, 1, 2])  # Isotropic case

    else:
        g_iso = np.mean(values)
        order = np.argsort(np.abs(values - g_iso))
        g_y, g_x, g_z = order

        return np.array([g_x, g_y, g_z])


def calculate_g_factors(muxyz: np.ndarray, start: int, final: int) -> GSquaredTensor:
    """Calculate the principal g-factors and their eigenvectors.

    Args:
        muxyz: The magnetic moment operator matrices acting between
            relevant eigenfunctions. The muxyz array must have the dimensions
            of the relevant eigenfunctions.
        start: The lowest eigenfunction index.
        final: The highest eigenfunction index.

    Returns:
        GSquaredTensor
    """
    g_squared_tensor = calculate_g_squared_tensor(
        muxyz=muxyz[:, start : final + 1, start : final + 1]
    )

    result = solve_matrix(g_squared_tensor)

    return GSquaredTensor(
        tensor=g_squared_tensor,
        values=np.sqrt(result.values),
        vectors=result.vectors,
        metadata=GSquaredTensorMetadata(
            eigenfunction_start_index=start,
            eigenfunction_final_index=final,
        ),
    )


def calculate_g_factors_for_indices(
    muxyz: np.ndarray, indices: list[int], multiplicity: int | None = None
) -> GSquaredTensor:
    """
    Calculate the principal g-factors and their eigenvectors for a specified
    range of indices which represent a spin-manifold.
    i.e. for a quintet state where index=2 is a singlet state, the indices
    [0, 1, 3, 4, 5] can be passed to represent the S=2 manifold.

    Args:
        muxyz: The magnetic moment operator matrices acting between
            relevant eigenfunctions. The muxyz array must have the dimensions
            of the relevant eigenfunctions.
        indices: List of indices to calculate g factors for.
        multiplicity: The spin multiplicity (2S + 1). If no value is passed this is
            determined from the number of states in the muxyz matrix.

    Returns:
        GSquaredTensor
    """
    idx = np.asarray(sorted(set(indices)), dtype=int)
    g_squared_tensor = calculate_g_squared_tensor(
        muxyz=muxyz[:, idx[:, None], idx[None, :]],
        multiplicity=multiplicity,
    )
    result = solve_matrix(g_squared_tensor)

    return GSquaredTensor(
        tensor=g_squared_tensor,
        values=np.sqrt(result.values),
        vectors=result.vectors,
        metadata=GSquaredTensorMetadata(
            eigenfunction_start_index=min(indices),
            eigenfunction_final_index=max(indices),
        ),
    )


def calculate_g_squared_tensor(
    muxyz: np.ndarray, multiplicity: int | None = None
) -> np.ndarray:
    """Calculate the g**2 tensor.

    Args:
        muxyz: The magnetic moment operator matrices acting between
            relevant eigenfunctions. The muxyz array must have the dimensions
            of the relevant eigenfunctions.
        multiplicity: The spin multiplicity (2S + 1). If no value is passed this is
            determined from the number of states in the muxyz matrix.

    Returns:
        3x3 g**2 tensor, whose basis components is in the order of x, y, and z.
    """
    number_of_states = multiplicity if multiplicity is not None else muxyz.shape[1]
    effective_spin = _get_effective_spin(number_of_states)
    denominator_term = _calculate_denominator_term(effective_spin)
    numerator_term = _calculate_numerator_terms(muxyz)
    return numerator_term / denominator_term


def _calculate_numerator_terms(muxyz: np.ndarray) -> np.ndarray:
    """Calculate the numerator terms of the g squared tensor.

    Args:
        muxyz: The magnetic moment operator matrices acting between
            relevant eigenfunctions.

    Returns:
        np.ndarray: 3x3 g**2 tensor.
    """
    g_squared_numerator = np.zeros((3, 3), dtype=complex)
    for alpha, beta in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        mu_alpha = muxyz[alpha, :, :]
        mu_beta = muxyz[beta, :, :]
        value = _calculate_tensor_element_numerator(mu_alpha, mu_beta)
        g_squared_numerator[alpha, beta] = value
        g_squared_numerator[beta, alpha] = value.conjugate()
    return g_squared_numerator


def _calculate_tensor_element_numerator(
    mu_alpha: np.ndarray, mu_beta: np.ndarray
) -> complex:
    """Calculate a g**2 tensor element."""
    return complex(np.sum(np.diagonal(np.matmul(mu_alpha, mu_beta))))


def _get_effective_spin(number_of_states: int) -> float:
    """Return the effective spin projection.

    If there are N states, then the effective spin multiplicity is also N.
    The spin multiplicity is related to the spin by the formula:
        >>> N = 2S+1
        >>> S = (N-1)/2

    Args:
        number_of_states

    Returns:
        float: (N-1)/2
    """
    return (number_of_states - 1.0) / 2.0


def _calculate_denominator_term(effective_spin: float) -> float:
    return float(np.sum(np.arange(-effective_spin, effective_spin + 1.0) ** 2.0))
