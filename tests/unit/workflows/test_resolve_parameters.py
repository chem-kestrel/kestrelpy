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

import pytest

from kestrelpy.parameters import (
    DipoleAssignment,
    DipoleParameters,
    Fixed,
    LigandAssignment,
    LigandParameters,
    Link,
    MetalAssignment,
    MetalParameters,
    Varied,
)
from kestrelpy.workflows.resolve_parameters import (
    get_dipole_parameters,
    get_ligand_field_parameters,
)


def test_ligand_assignments() -> None:
    chloride = LigandAssignment(
        name="Cl",
        esigma=Fixed(5000.0),
        epix=Varied([0.0, 500.0, 1000.0]),
        epiy=Link.linear("ligand.Cl.epix"),
    )
    assert chloride == LigandAssignment.from_dict(chloride.to_dict())
    chloride.to_json()


@dataclass
class ResolvedLFParameters:
    metal: MetalParameters
    ligands: list[LigandParameters]


@pytest.mark.parametrize(
    ("input", "expected_output"),
    (
        # Metal only
        pytest.param(
            [MetalAssignment(d_electrons=9)],
            [
                ResolvedLFParameters(
                    metal=MetalParameters(d_electrons=9),
                    ligands=[],
                ),
            ],
            id="MetalAssignment",
        ),
        pytest.param(
            [
                MetalAssignment(
                    d_electrons=9,
                    racah_b=Varied([300.0, 400.0]),
                    racah_c=Varied([2000.0, 2500.0]),
                )
            ],
            [
                ResolvedLFParameters(
                    metal=MetalParameters(
                        d_electrons=9,
                        racah_b=300.0,
                        racah_c=2000.0,
                    ),
                    ligands=[],
                ),
                ResolvedLFParameters(
                    metal=MetalParameters(
                        d_electrons=9,
                        racah_b=300.0,
                        racah_c=2500.0,
                    ),
                    ligands=[],
                ),
                ResolvedLFParameters(
                    metal=MetalParameters(
                        d_electrons=9,
                        racah_b=400.0,
                        racah_c=2000.0,
                    ),
                    ligands=[],
                ),
                ResolvedLFParameters(
                    metal=MetalParameters(
                        d_electrons=9,
                        racah_b=400.0,
                        racah_c=2500.0,
                    ),
                    ligands=[],
                ),
            ],
            id="MetalAssignment with 2 varying parameters",
        ),
        pytest.param(
            [
                MetalAssignment(
                    d_electrons=9,
                    racah_b=Fixed(700.0),
                    racah_c=Link.linear("metal.racah_b", gradient=4.0),
                ),
            ],
            [
                ResolvedLFParameters(
                    metal=MetalParameters(d_electrons=9, racah_b=700.0, racah_c=2800.0),
                    ligands=[],
                ),
            ],
            id="MetalAssignment with an internal link",
        ),
        pytest.param(
            [
                MetalAssignment(
                    d_electrons=9,
                    racah_b=Varied([0.0, 500.0, 1000.0]),
                    racah_c=Link.linear("metal.racah_b", gradient=4.0),
                ),
            ],
            [
                ResolvedLFParameters(
                    metal=MetalParameters(d_electrons=9, racah_b=0.0, racah_c=0.0),
                    ligands=[],
                ),
                ResolvedLFParameters(
                    metal=MetalParameters(d_electrons=9, racah_b=500.0, racah_c=2000.0),
                    ligands=[],
                ),
                ResolvedLFParameters(
                    metal=MetalParameters(
                        d_electrons=9, racah_b=1000.0, racah_c=4000.0
                    ),
                    ligands=[],
                ),
            ],
            id="MetalAssignment with an internal link and varying a parameter",
        ),
        # 1x Metal + 1x Ligand
        pytest.param(
            [
                MetalAssignment(d_electrons=9),
                LigandAssignment(name="Cl"),
            ],
            [
                ResolvedLFParameters(
                    metal=MetalParameters(d_electrons=9),
                    ligands=[LigandParameters(name="Cl")],
                ),
            ],
            id="Basic 1x MetalAssignment + 1x LigandAssignment (default values)",
        ),
        pytest.param(
            [
                MetalAssignment(
                    d_electrons=9,
                    racah_b=Fixed(700.0),
                    racah_c=Link.linear("ligand.Cl.esigma"),
                ),
                LigandAssignment(
                    name="Cl",
                    esigma=Fixed(1000.0),
                    epix=Link.linear("metal.racah_b"),
                ),
            ],
            [
                ResolvedLFParameters(
                    metal=MetalParameters(
                        d_electrons=9,
                        racah_b=700.0,
                        racah_c=1000.0,
                    ),
                    ligands=[
                        LigandParameters(
                            name="Cl",
                            esigma=1000.0,
                            epix=700.0,
                        ),
                    ],
                ),
            ],
            id="Simple link between 1x MetalAssignment and 1x Ligand Assignment",
        ),
        pytest.param(
            [
                MetalAssignment(
                    d_electrons=9,
                    racah_b=Fixed(500.0),
                    racah_c=Link.linear("ligand.Cl.epix", gradient=2.0),
                ),
                LigandAssignment(
                    name="Cl",
                    esigma=Link.linear("metal.racah_b", gradient=2.0),
                    epix=Link.linear("ligand.Cl.esigma", gradient=2.0),
                ),
            ],
            [
                ResolvedLFParameters(
                    metal=MetalParameters(
                        d_electrons=9,
                        racah_b=500.0,
                        racah_c=4000.0,
                    ),
                    ligands=[
                        LigandParameters(
                            name="Cl",
                            esigma=1000.0,
                            epix=2000.0,
                        ),
                    ],
                ),
            ],
            id="Chain links between 1x MetalAssignment and 1x Ligand Assignment",
        ),
    ),
)
def test_get_ligand_field_parameters(
    input: list[LigandAssignment | MetalAssignment],
    expected_output: list[ResolvedLFParameters],
) -> None:
    output = [
        parameters for parameters in get_ligand_field_parameters(assignments=input)
    ]
    for output_iter, expected_output_iter in zip(output, expected_output):
        assert (
            ResolvedLFParameters(
                metal=output_iter.metal,
                ligands=output_iter.ligands,
            )
            == expected_output_iter
        )


@pytest.mark.parametrize(
    ("input", "expected_output"),
    (
        pytest.param(
            [DipoleAssignment(name="Cl")],
            [[DipoleParameters(name="Cl")]],
            id="DipoleAssignment",
        ),
        pytest.param(
            [DipoleAssignment(name="Cl", psigma=Fixed(100.0))],
            [[DipoleParameters(name="Cl", psigma=100.0)]],
            id="DipoleAssignment with basic fixed values",
        ),
        pytest.param(
            [
                DipoleAssignment(
                    name="Cl",
                    psigma=Fixed(100.0),
                    ppix=Fixed(50.0),
                    ppiy=Link.linear("ligand.Cl.ppix"),
                )
            ],
            [[DipoleParameters(name="Cl", psigma=100.0, ppix=50.0, ppiy=50.0)]],
            id="DipoleAssignment with basic link",
        ),
        pytest.param(
            [
                DipoleAssignment(
                    name="Cl",
                    psigma=Varied([0.0, 100.0]),
                    ppix=Fixed(50.0),
                    ppiy=Link.linear("ligand.Cl.ppix"),
                )
            ],
            [
                [DipoleParameters(name="Cl", psigma=0.0, ppix=50.0, ppiy=50.0)],
                [DipoleParameters(name="Cl", psigma=100.0, ppix=50.0, ppiy=50.0)],
            ],
            id="DipoleAssignment with basic variation",
        ),
    ),
)
def test_get_dipole_parameters(
    input: list[DipoleAssignment],
    expected_output: list[list[DipoleParameters]],
) -> None:
    output = [parameters for parameters in get_dipole_parameters(assignments=input)]
    assert output == expected_output
