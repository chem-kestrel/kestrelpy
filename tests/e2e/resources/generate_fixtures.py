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

Generate e2e test fixtures

Run the script for generating test fixtures from the project root:
```bash
$ poetry run python -m tests.e2e.resources.generate_fixtures
```
"""

import json
from typing import TypedDict, TypeVar, cast

import numpy as np

from kestrelpy.calculations.magnetism.susceptibilities import (
    ParamagneticSusceptibilityTensors,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    Eigenfunction,
)
from kestrelpy.models.results.calculations import EPRProperties
from tests.e2e.resources.systems import (
    ELECTRONIC_STRUCTURE_TEST_CASES,
    ElectronicStructureTestCase,
)

T = TypeVar("T")


class _Configuration(TypedDict):
    occupation: list[int]
    contribution: float
    spin_multiplicity: int


class _Eigenfunction(TypedDict):
    index: int
    energy: float
    spin_multiplicity_projection: dict[str, float]
    configurations: list[_Configuration]


def generate_eigenfunction_fixtures(
    eigenfunctions: list[Eigenfunction],
) -> list[_Eigenfunction]:
    output = []
    for eigenfunction in eigenfunctions:
        output.append(
            _Eigenfunction(
                index=int(eigenfunction.index),
                energy=float(round(eigenfunction.energy, 1)),
                spin_multiplicity_projection={
                    str(k): float(round(v, 3))
                    for k, v in eigenfunction.spin_multiplicity_projection.items()
                },
                configurations=[
                    _Configuration(
                        occupation=list(int(x) for x in config.occupation),
                        contribution=float(round(config.contribution, 3)),
                        spin_multiplicity=int(config.spin_multiplicity),
                    )
                    for config in eigenfunction.configurations
                ],
            )
        )
    return output


class _EPRResult(TypedDict, total=False):
    start_index: int
    final_index: int
    g_factors: list[float]
    a_values: list[float] | None
    dipolar: list[float] | None
    orbital: list[float] | None
    fermi: list[float] | None


def generate_epr_fixtures(epr_properties: list[EPRProperties]) -> list[_EPRResult]:
    output = []
    for epr_property in epr_properties:
        hyperfine = epr_property.metal_hyperfine_coupling
        if not hyperfine:
            output.append(
                _EPRResult(
                    start_index=int(epr_property.start),
                    final_index=int(epr_property.final),
                    g_factors=[
                        float(round(g, 3)) for g in epr_property.g_factors.values
                    ],
                    a_values=[],
                )
            )
        elif hyperfine.sign:
            output.append(
                _EPRResult(
                    start_index=int(epr_property.start),
                    final_index=int(epr_property.final),
                    g_factors=[
                        float(round(g, 3)) for g in epr_property.g_factors.values
                    ],
                    a_values=[float(round(a, 3)) for a in hyperfine.sign.total],
                    dipolar=[float(round(a, 3)) for a in hyperfine.sign.dipolar],
                    orbital=[float(round(a, 3)) for a in hyperfine.sign.orbital],
                    fermi=[float(round(a, 3)) for a in hyperfine.sign.fermi],
                )
            )
        else:
            assert hyperfine is not None
            assert hyperfine.fermi_contact is not None
            output.append(
                _EPRResult(
                    start_index=int(epr_property.start),
                    final_index=int(epr_property.final),
                    g_factors=[
                        float(round(g, 3)) for g in epr_property.g_factors.values
                    ],
                    fermi=[float(round(a, 3)) for a in hyperfine.fermi_contact],
                )
            )

    return output


class _SusceptibilityResults(TypedDict):
    temperature: float
    chi_T: float


def generate_susceptibility_fixtures(
    susceptibilities: ParamagneticSusceptibilityTensors,
) -> list[_SusceptibilityResults]:
    output: list[_SusceptibilityResults] = []
    for tensor in susceptibilities.tensors:
        output.append(
            _SusceptibilityResults(
                temperature=round(tensor.temperature, 1),
                chi_T=round(np.mean(tensor.values) * tensor.temperature, 3),
            )
        )
    return output


def normalise_signed_zero(obj: T) -> T:
    """A helper function to handle converting -0.0 -> +0.0."""
    if isinstance(obj, float):
        # catches both 0.0 and -0.0
        return cast(T, 0.0 if obj == 0.0 else obj)
    if isinstance(obj, list):
        return cast(T, [normalise_signed_zero(x) for x in obj])
    if isinstance(obj, dict):
        return cast(T, {k: normalise_signed_zero(v) for k, v in obj.items()})
    return obj


class Fixture(TypedDict):
    eigenfunctions: list[_Eigenfunction]
    epr: list[_EPRResult]
    susceptibilities: list[_SusceptibilityResults]


def generate_test_fixture(test_case: ElectronicStructureTestCase) -> Fixture:
    calculation = test_case.calculation()
    epr_fixtures = (
        generate_epr_fixtures(calculation.results[0].epr)
        if calculation.results[0].epr
        else []
    )

    susceptibility_fixtures = (
        generate_susceptibility_fixtures(calculation.results[0].susceptibilities)
        if calculation.results[0].susceptibilities
        else []
    )

    return normalise_signed_zero(
        Fixture(
            eigenfunctions=generate_eigenfunction_fixtures(
                calculation.results[0].calculation.eigenfunctions.eigenfunctions
            ),
            epr=epr_fixtures,
            susceptibilities=susceptibility_fixtures,
        )
    )


if __name__ == "__main__":
    for test_case in ELECTRONIC_STRUCTURE_TEST_CASES:
        test_fixtures = generate_test_fixture(test_case=test_case)
        with open(test_case.expected_results_path, "w") as f:
            json.dump(test_fixtures, f, indent=2)
        with open(test_case.expected_results_path, "a") as f:
            f.write("\n")
