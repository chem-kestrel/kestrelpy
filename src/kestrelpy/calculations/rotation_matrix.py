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


def get_rotation_matrix(
    tta: float, phi: float, psi: float, degrees: bool = True
) -> np.ndarray:
    """Returns the Schaffer rotation matrix in a real d orbital basis.

    Note the basis set order: xy (0), yz (1), z2 (2), xz (3), x2y2 (4)
    which is different from the order given in Table 2b of the reference:
    Schäffer, C. E. (n.d.). A perturbation representation of weak
    covalent bonding. Structure and Bonding, 68-95.

    Args:
        tta: the spherical polar coordinate angle theta.
        phi: the spherical polar coordinate angle phi.
        psi: the spherical polar coordinate angle psi.
        degrees: Flag which indicates whether the input angles are in degrees.

    Returns:
        np.ndarray, shape(5, 5): The rotation matrix of the potential.
    """
    if degrees:
        tta, phi, psi = np.radians([tta, phi, psi])

    ctta = np.cos(tta)
    cphi = np.cos(phi)
    cpsi = np.cos(psi)
    stta = np.sin(tta)
    sphi = np.sin(phi)
    spsi = np.sin(psi)
    ctta2 = np.cos(2.0 * tta)
    cphi2 = np.cos(2.0 * phi)
    cpsi2 = np.cos(2.0 * psi)
    stta2 = np.sin(2.0 * tta)
    sphi2 = np.sin(2.0 * phi)
    spsi2 = np.sin(2.0 * psi)
    square_root_3 = np.sqrt(3)

    return np.array(
        [
            # Row 1 = < xy |
            [
                (cphi2 * ctta * cpsi2 + sphi2 * (0.75 + 0.25 * ctta2) * -spsi2),
                (cphi2 * stta * cpsi + sphi2 * 0.5 * stta2 * -spsi),
                (sphi2 * ((0.25 * square_root_3) - (0.25 * square_root_3) * ctta2)),
                (cphi2 * stta * spsi + sphi2 * 0.5 * stta2 * cpsi),
                (cphi2 * ctta * spsi2 + sphi2 * (0.75 + 0.25 * ctta2) * cpsi2),
            ],
            # Row 2 = < yz |
            [
                (cphi * -stta * cpsi2 + sphi * (-0.5 * stta2) * -spsi2),
                (cphi * ctta * cpsi + sphi * ctta2 * -spsi),
                (sphi * square_root_3 * 0.5 * stta2),
                (cphi * ctta * spsi + sphi * ctta2 * cpsi),
                (cphi * -stta * spsi2 + sphi * -0.5 * stta2 * cpsi2),
            ],
            # Row 3 = < z2 |
            [
                (((0.25 * square_root_3) - (0.25 * square_root_3) * ctta2) * -spsi2),
                (-(0.5 * square_root_3) * stta2 * -spsi),
                (0.25 + 0.75 * ctta2),
                (-(0.5 * square_root_3) * stta2 * cpsi),
                (((0.25 * square_root_3) - (0.25 * square_root_3) * ctta2) * cpsi2),
            ],
            # Row 4 = < xz |
            [
                (-sphi * -stta * cpsi2 + cphi * (-0.5 * stta2) * -spsi2),
                (-sphi * ctta * cpsi + cphi * ctta2 * -spsi),
                (cphi * (square_root_3 * 0.5) * stta2),
                (-sphi * ctta * spsi + cphi * ctta2 * cpsi),
                (-sphi * -stta * spsi2 + cphi * (-0.5 * stta2) * cpsi2),
            ],
            # Row 5 = < x2-y2 |
            [
                (-sphi2 * ctta * cpsi2 + cphi2 * (0.75 + 0.25 * ctta2) * -spsi2),
                (-sphi2 * stta * cpsi + cphi2 * (0.5 * stta2) * -spsi),
                (cphi2 * ((0.25 * square_root_3) - (0.25 * square_root_3) * ctta2)),
                (-sphi2 * stta * spsi + cphi2 * (0.5 * stta2) * cpsi),
                (-sphi2 * ctta * spsi2 + cphi2 * (0.75 + 0.25 * ctta2) * cpsi2),
            ],
        ],
    )


def rotate_ligand_field_matrix(
    ligand_field_matrix: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate a ligand field matrix with a rotation matrix.

    Args:
        ligand_field_matrix: shape(5, 5)
        rotation_matrix: shape(5, 5)

    Returns:
        np.ndarray, shape(5, 5): the rotated ligand field matrix.
    """
    return np.matmul(rotation_matrix, np.matmul(ligand_field_matrix, rotation_matrix.T))


def ensure_right_handed_axis(vectors: np.ndarray) -> np.ndarray:
    """Ensure a set of eigenvectors show a right-handed coordinate axis

    Checks whether the determinant of a set of eigenvectors is less than 1.
    If it is, the direction of the third column is flipped to ensure the
    determinant is positive. This is possible since eigenvector signs
    computed by np.linalg.eigh are undetermined.

    Args:
        vectors: Set of 3x3 eigenvectors

    Returns:
        np.ndarray, eigenvectors in right-handed coordinate frame
    """
    vectors = vectors.copy()
    if np.linalg.det(vectors) < 0:
        vectors[:, 2] *= -1

    return vectors
