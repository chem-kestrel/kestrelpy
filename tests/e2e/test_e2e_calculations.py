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

import json

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.e2e.resources.generate_fixtures import generate_test_fixture
from tests.e2e.resources.systems import (
    ELECTRONIC_STRUCTURE_TEST_CASES,
    ElectronicStructureTestCase,
)


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(test_case, id=test_case.id)
        for test_case in ELECTRONIC_STRUCTURE_TEST_CASES
    ],
)
def test_e2e(test_case: ElectronicStructureTestCase) -> None:
    actual_results = generate_test_fixture(test_case)
    with open(test_case.expected_results_path, "r") as f:
        expected_results = json.load(f)
    assert actual_results == expected_results


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(test_case, id=test_case.id)
        for test_case in ELECTRONIC_STRUCTURE_TEST_CASES
    ],
)
def test_benchmark_e2e(
    benchmark: BenchmarkFixture, test_case: ElectronicStructureTestCase
) -> None:
    benchmark(generate_test_fixture, test_case)
