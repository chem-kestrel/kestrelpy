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

Module for computing geometric quantities."""

from typing import Annotated, Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from kestrelpy.calculations.rotation_matrix import ensure_right_handed_axis
from kestrelpy.configuration import get_logger

logger = get_logger(__name__)

Coordinates = Annotated[np.ndarray, Literal[3]]

GLOBAL_COORDINATE_AXIS = np.diag(np.ones(3))

TRUE_ORIGIN = np.zeros(3)


def calculate_direction_cosine_matrix(
    theta: float,
    phi: float,
    psi: float,
    degrees: bool = False,
) -> np.ndarray:
    """Calculate the direction cosine matrix for theta, phi, and psi.

    The direction cosine is to relate the local x, y, and z frame of the ligand
    to the molecular XYZ frame.

    To do this we use Scipy's method to compute the rotation matrix from
    Euler angles alpha, beta, and gamma.
        phi --> alpha, theta --> beta, and psi --> gamma.

    We use intrinsic rotations to reflect how the spherical polar coordinates
    transform about the local frame of the ligand:

        1. Rotation around z by -phi to get x', y', and z'.
        2. Rotation around y' by -theta to get x'', y'', and z''.
        3. Rotation around z'' by -psi to get x''', y''', and z'''.

    NOTE: the negative angle is because scipy defines a positive rotation as
    a clockwise rotation looking from the head of the vector to the origin. In
    Kestrel, we define a positive rotation as the anti-clockwise rotation from the
    same point-of-view; or put another, we define a positive rotation that is
    clockwise looking from the origin towards the head of the vector.

    Args:
        theta: Spherical polar coordinate (ISO convention).
        phi: Spherical polar coordinate (ISO convention).
        psi: Spherical polar coordinate.
        degrees: Boolean flag if the supplied angles are in degrees (True) or
            radians (False). Default=False.

    Returns:
        3x3 direction cosine matrix, whose rows are the local frame of the ligand
        and the columns reference the molecular frame.
    """
    r = R.from_euler("zyz", [-phi, -theta, -psi], degrees=degrees)
    return r.as_matrix()


def euler_angles_from_matrix(rotation_matrix: np.ndarray) -> np.ndarray[float]:
    """
    Takes a set of 3x3 vectors and uses Scipy's method to extract the Euler angles,
    alpha, beta and gamma, from a 3x3 rotation matrix, where:
        phi --> alpha, theta --> beta, psi --> gamma.

    alpha, beta and gamma are calculated according to a ZYZ rotation from the
    cartesian frame to the that described by the rotation matrix.

    Args:
          rotation_matrix: 3x3 rotation matrix

    Returns:
        alpha, beta, gamma as a numpy array
    """
    rot_matrix = np.asarray(rotation_matrix).copy()
    rot_matrix = ensure_right_handed_axis(vectors=rot_matrix)
    rot_matrix[:, 2] = np.cross(rot_matrix[:, 0], rot_matrix[:, 1])

    rot_matrix /= np.linalg.norm(rot_matrix, axis=0)

    rotation = R.from_matrix(matrix=rot_matrix.T)
    return rotation.as_euler("ZYZ", degrees=True)  # Use ZYZ rotation convention


def calculate_dihedral(
    a: Coordinates, b: Coordinates, c: Coordinates, d: Coordinates
) -> float:
    """Returns the dihedral angle between the planes abc and bcd."""
    p0 = np.copy(a).astype(float)
    p1 = np.copy(b).astype(float)
    p2 = np.copy(c).astype(float)
    p3 = np.copy(d).astype(float)

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, np.transpose(b1)) * (1.0 / np.linalg.norm(b1))
    x = np.dot(b0xb1, np.transpose(b1xb2))

    return float(np.degrees(np.arctan2(y, x)))


def calculate_theta(z_coordinate: float) -> float:
    """Calculate the angle theta, in degrees.

    Args:
        z_coordinate: The z coordinate of the coordinating atom.

    Returns:
        Theta, in degrees.
    """
    return float(np.degrees(np.arccos(z_coordinate)))


def calculate_phi(coordinates: Coordinates) -> float:
    """Calculate the angle phi, in degrees.

    Args:
        coordinates: Normalised xyz coordinates of the coordinating atom.

    Returns:
        Phi, in degrees.
    """
    return calculate_dihedral(
        GLOBAL_COORDINATE_AXIS[0, :],
        TRUE_ORIGIN,
        GLOBAL_COORDINATE_AXIS[2, :],
        coordinates,
    )


def calculate_psi(
    local_x_orientation: Coordinates,
    theta: float,
    phi: float,
) -> float:
    """Calculate the rotation around the local ligand-frame to align the x-direction.

    The local ligand-frame's x-direction is aligned to the `local_x_orientation`
    coordinate, such that the arrow-head of the x-direction in a right-handed
    coordinate frame points towards the `local_x_orientation` position.

    Args:
        local_x_orientation: The x,y,z coordinates of the position to align the local
            ligand-frame's x-direction to. The coordinates must be normalised.
        theta: The theta spherical polar coordinate (ISO convention) in degrees.
        theta: The phi spherical polar coordinate (ISO convention) in degrees.

    Returns:
        The angle psi, in degrees.
    """
    ligand_frame = calculate_direction_cosine_matrix(
        theta=theta, phi=phi, psi=0.0, degrees=True
    )
    return calculate_dihedral(
        ligand_frame[0, :],
        TRUE_ORIGIN,
        ligand_frame[2, :],
        local_x_orientation,
    )


def calculate_spherical_polar_coordinates(
    coordinating_atom: Coordinates,
    local_x_orientation: Coordinates | None = None,
    cartesian_origin: Coordinates | None = None,
) -> tuple[float, float, float]:
    """Calculate the spherical polar coordinates (ISO convention) for a ligand.

    Args:
        coordinating_atom: The xyz coordinates of the ligand coordinating atom
            position.
        local_x_orientation: The xyz coordinates defining the local x-orientation
            of the ligand frame. If `None` then psi will be returned with a value
            of 0.0 degrees.
        cartesian_origin: The origin position of the cartesian space. If `None`
            then defaults to (0, 0, 0).

    Returns:
        theta, phi, and psi, in degrees.
    """
    origin = TRUE_ORIGIN if cartesian_origin is None else cartesian_origin

    vector_to_coordinating_atom = np.array(coordinating_atom) - origin
    normalised_vector = vector_to_coordinating_atom / np.linalg.norm(
        vector_to_coordinating_atom
    )

    theta = calculate_theta(normalised_vector[2])
    phi = calculate_phi(normalised_vector)

    if local_x_orientation is None:
        return theta, phi, 0.0

    vector_to_x_orientation = np.array(local_x_orientation) - origin
    normalised_x_orientation = np.array(vector_to_x_orientation) / np.linalg.norm(
        vector_to_x_orientation
    )
    psi = calculate_psi(
        local_x_orientation=normalised_x_orientation,
        theta=theta,
        phi=phi,
    )
    return theta, phi, psi
