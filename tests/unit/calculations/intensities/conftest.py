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

from pathlib import Path

import numpy as np

from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ElectronicLevel,
)

DATA_DIRECTORY = Path(__file__).parent / "data"


def get_numpy_resource(file: str) -> dict[str, np.ndarray]:
    with open(DATA_DIRECTORY / file, "rb") as f:
        return dict(np.load(f))


def get_electronic_level(index: int, degeneracy: int) -> ElectronicLevel:
    return ElectronicLevel(
        index=index,
        eigenfunctions=[],
        energy=0.0,
        zero_point_energy=0.0,
        degeneracy=degeneracy,
        spin_multiplicity_projection={},
        configurations=[],
    )
