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

from kestrelpy.calculations.magnetism.hyperfine_tensor.hyperfine_sign import (
    calculate_dipolar_sign,
    calculate_fermi_sign,
)


class TestHyperfineSignCalculations:
    """
    Test that the signs of the A values and of the individual fermi contact,
    dipolar and orbital contributions are calculated correctly.

    Values are taken from the parameterisation of Cu(NH3)4 in kestrel, with:
        e_sigma = 5560 cm-1
        e_pi_y = 400 cm-1
        k_iso = 0.57
        kappa = 0.016

    Results have been validated by comparison to DfT calculations by:
        Frank Neese (2004): Sum-over states based multi-reference ab initio calculation
        of EPR spin Hamiltonian parameters for transition metal complexes. A case study
        Magnetic Resonance in Chemistry, 42, 1, 187-198, DOI: 10.1002/mrc.1456
    """

    def test_calculate_fermi_sign(self) -> None:
        """
        Test that the fermi contact is calculated to be negative, as expected for Cu(II)
        where the nuclear g factor is positive: gn = 1.482 for 63Cu and 1.588 for 65Cu
        """
        fermi_contact = np.array([311, 311, 311])
        metal = "cu"
        expected_output = np.array([-311, -311, -311])

        output = calculate_fermi_sign(fermi_contact, metal)
        assert np.array_equal(output, expected_output)

    def test_calculate_dipolar_sign(self) -> None:
        """
        Test that the sign of the dipolar contribution is correctly calculated.
        Input data and expected output is taken from ligand field calculations of
        square planar Cu(NH3)4.
        """
        dipolar_values = np.array([270, 270, 546])
        dipolar_vectors = np.eye(3)
        expected_output = np.array([270.0, 270.0, -546.0])

        dip_sign = calculate_dipolar_sign(
            values=dipolar_values,
            vectors=dipolar_vectors,
            dipolar_z_sign=-1.0,
            metal="Cu",
        )
        assert np.allclose(dip_sign, expected_output)
