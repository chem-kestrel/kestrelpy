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

from dataclasses import dataclass
from importlib import resources

import numpy as np

from kestrelpy.calculations import many_electron_matrix_elements
from kestrelpy.calculations.operators import Operators
from kestrelpy.calculations.utils import get_minimal_d_config


@dataclass
class NonZeroMatrixElements:
    indices: dict[str, np.ndarray]
    values: dict[str, np.ndarray]


def get_non_zero_matrix_elements(d_electrons: int) -> NonZeroMatrixElements:
    """Return non-zero matrix elements for the operators in the Effective Hamiltonian.

    Args:
        d_electrons: The number of electrons in the d electronic configuration.

    Returns:
        NonZeroMatrixElements
    """
    electron_number = get_minimal_d_config(d_electrons)
    with resources.open_binary(
        many_electron_matrix_elements, f"D{electron_number}.npz"
    ) as f:
        matrix_elements: dict[str, np.ndarray] = np.load(f)
        indices = {k: np.astype(v[:, :2], int) for k, v in matrix_elements.items()}
        elements = {k: np.astype(v[:, 2], float) for k, v in matrix_elements.items()}

    if d_electrons > 5:
        for k in elements.keys():
            if k in list(Operators.SphericalHarmonic) or k in list(
                Operators.SpinOrbitCoupling
            ):
                elements[k] *= -1.0

    return NonZeroMatrixElements(indices=indices, values=elements)
