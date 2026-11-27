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

from enum import Enum
from unittest.mock import Mock

import pytest

from kestrelpy.models.input_models.optimisation import (
    OptimisationBlock,
    OptimisationMethod,
)
from kestrelpy.workflows.optimisation.perform_optimisation import (
    UnrecognisedOptimisationMethodError,
    optimise_system,
)


class _OptimisationMethod(str, Enum):
    UNKNOWN_METHOD = "unknown_method"


def test_perform_optimisation_raises_error_for_unrecognised_system() -> None:
    instructions = OptimisationBlock(
        method=_OptimisationMethod.UNKNOWN_METHOD  # type: ignore[arg-type]
    )
    with pytest.raises(UnrecognisedOptimisationMethodError) as error:
        optimise_system(
            func=Mock(),
            args=Mock(),
            instructions=instructions,
            starting_values=Mock(),
            bounds=Mock(),
        )
        assert (
            f"Unrecognised optimisation method: {instructions.method=}, "
            f"please choose one of {[method.value for method in OptimisationMethod]}"
            in str(error)
        )
