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


def calculate_ds_mixing_matrix(
    *ligand_parameters: tuple[float, float, float],
) -> np.ndarray:
    """Calculate the total 5x5 d-s stabilisation matrix.

    References:
        Mark J. Riley,
        The effects of d-s mixing in low symmetry transition metal complexes,
        Inorganica Chimica Acta,
        Volume 268, Issue 1,
        1998,
        Pages 55-62,
        ISSN 0020-1693,
        https://doi.org/10.1016/S0020-1693(97)05719-8.
        (https://www.sciencedirect.com/science/article/pii/S0020169397057198)
        Abstract: The extension of the angular overlap model to include the effect
            of d-s mixing on the electronic structure of low symmetry transition metal
            complexes is investigated. It is shown that even for complexes with no
            symmetry, an axis can be defined in which only one d orbital is affected
            by the d-s mixing. However in a general low symmetry complex, the use of
            a coordinate system defined by this axis can introduce additional complexity
            into the description of the low symmetry ligand field. This formalism is used
            to calculate one electron d orbital energies for a number of low symmetry
            distortions of a ML5 complex. In addition the effect of the d-s mixing on the
            electronic groundstate of octahedral Cu(II)L6 complexes is examined as a
            function of the Jahn-Teller active coordinates.
    """
    vector = np.zeros(5)
    for eds, theta, phi in ligand_parameters:
        sqrt_3_div_2 = np.sqrt(3.0) * 0.5

        sin_theta = np.sin(theta)
        sin_phi = np.sin(phi)
        sin_2_phi = np.sin(2.0 * phi)
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)
        cos_2_phi = np.cos(2.0 * phi)

        vector += np.array(
            [
                # | xy >
                sqrt_3_div_2 * sin_theta * sin_theta * sin_2_phi,
                # | yz >
                sqrt_3_div_2 * sin_theta * sin_theta * sin_phi,
                # | z2 >
                0.5 * ((3.0 * cos_theta * cos_theta) - 1.0),
                # | xz >
                sqrt_3_div_2 * sin_theta * sin_theta * cos_phi,
                # | x2y2 >
                sqrt_3_div_2 * sin_theta * sin_theta * cos_2_phi,
            ]
        ) * np.sqrt(eds)

    return -np.outer(vector, vector)
