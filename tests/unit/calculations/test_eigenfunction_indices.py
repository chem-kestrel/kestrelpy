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

import logging

import pytest
from _pytest.logging import LogCaptureFixture

from kestrelpy.calculations.eigenfunction_indices import (
    EigenfunctionIndices,
    _get_number_eigenfunction_pairs,
)


@pytest.mark.parametrize(("start", "final"), [(-1, 1), (4, 1)])
def test_validation_raises_value_error(start: int, final: int) -> None:
    with pytest.raises(ValueError):
        EigenfunctionIndices(start=start, final=final)


@pytest.mark.parametrize(
    ("a_indices", "g_indices", "expected_output", "warning_logged"),
    [
        pytest.param(
            [EigenfunctionIndices(start=0, final=1)],
            [EigenfunctionIndices(start=0, final=1)],
            1,
            False,
            id="identical_indices",
        ),
        pytest.param(
            [EigenfunctionIndices(start=0, final=1)],
            [EigenfunctionIndices(start=0, final=2)],
            1,
            True,
            id="non_identical_indices",
        ),
        pytest.param(
            [
                EigenfunctionIndices(start=0, final=1),
                EigenfunctionIndices(start=2, final=3),
            ],
            [EigenfunctionIndices(start=0, final=1)],
            1,
            True,
            id="different_number_of_pairs",
        ),
    ],
)
def test_determine_number_index_pairs(
    a_indices: list[EigenfunctionIndices],
    g_indices: list[EigenfunctionIndices],
    expected_output: int,
    warning_logged: bool,
    caplog: LogCaptureFixture,
) -> None:
    """
    Test that the sign is calculated for the correct number of Eigenfunction pairs,
    as both A values and g factors are required. Also test that the user is
    warned (via logging) when there is a mismatch in Eigenfunction pairs.
    """
    caplog.set_level(logging.WARNING)
    result = _get_number_eigenfunction_pairs(g_factors=g_indices, a_values=a_indices)

    assert result == expected_output
    assert bool(caplog.messages) == warning_logged
