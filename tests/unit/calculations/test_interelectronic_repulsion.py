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

from kestrelpy.calculations.interelectronic_repulsion import (
    get_interelectronic_repulsion_parameters,
)


def test_slater_condon_parameters_are_correctly_calculated_from_racah_parameters() -> (
    None
):
    """Testing that the Slater-Condon parameters are calculated from Racah parameters.

    The test relies on a simple matrix transformation between Racah A, B, and C
    parameters and the Slater-Condon parameters.

    The transformation was taken from wikipedia:
        @misc{ enwiki:1198689098,
            author = "{Wikipedia contributors}",
            title = "Racah parameter --- {Wikipedia}{,} The Free Encyclopedia",
            year = "2024",
            url = "https://en.wikipedia.org/w/index.php?title=Racah_parameter&oldid=1198689098",
            note = "[Online; accessed 13-October-2024]"
        }
    """
    racah_a = 0.0
    racah_b = 1000.0
    racah_c = 4000.0
    racah_parameters = np.array([racah_a, racah_b, racah_c])

    transformation_matrix = np.array([[1, 0, -49], [0, 1, -5], [0, 0, 35]])
    expected_output = np.matmul(np.linalg.inv(transformation_matrix), racah_parameters)

    output = get_interelectronic_repulsion_parameters(racah_b=racah_b, racah_c=racah_c)

    assert output.F2 == (expected_output[1] * 49.0)
    assert output.F4 == (expected_output[2] * 441.0)
