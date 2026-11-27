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

from unittest.mock import Mock

import pytest

from kestrelpy.parameters import Fixed, LigandAssignment, Link, MetalAssignment, Varied
from kestrelpy.workflows.parameter_space import (
    _validate_input_arguments,
    slice_non_relativistic_energies,
)


@pytest.mark.parametrize(
    ("metal", "ligands", "error_msg"),
    (
        pytest.param(
            MetalAssignment(
                d_electrons=9,
                racah_b=Varied([0.0, 1.0]),
            ),
            [],
            None,
            id="Simple valid case: single parameter varied",
        ),
        pytest.param(
            MetalAssignment(
                d_electrons=9,
                racah_b=Varied([0.0, 1.0]),
                racah_c=Varied([0.0, 1.0]),
            ),
            [],
            None,
            id="Simple valid case: only metal varied",
        ),
        pytest.param(
            MetalAssignment(d_electrons=9),
            [
                LigandAssignment(
                    name="Ligand1",
                    esigma=Varied([0.0, 1.0]),
                    epix=Varied([0.0, 1.0]),
                    epiy=Link.linear("ligand.Ligand1.epix"),
                ),
            ],
            None,
            id="Simple valid case: one ligand only varied",
        ),
        pytest.param(
            MetalAssignment(d_electrons=9, racah_b=Varied([0.0, 1.0])),
            [LigandAssignment(name="Ligand1", esigma=Varied([0.0, 1.0]))],
            None,
            id="Simple valid case: one parameter varied in metal and ligand",
        ),
        pytest.param(
            MetalAssignment(d_electrons=9),
            [LigandAssignment(name="Ligand1")],
            (
                "You must vary at least one parameter, instead got: number_of_varied_parameters=0"
            ),
            id="Invalid case: no parameters varied",
        ),
    ),
)
def test_validate_input_arguments(
    metal: MetalAssignment, ligands: list[LigandAssignment], error_msg: str | None
) -> None:
    if error_msg:
        with pytest.raises(ValueError) as error:
            _validate_input_arguments(metal=metal, ligands=ligands)
        assert error_msg in str(error)
    else:
        _validate_input_arguments(metal=metal, ligands=ligands)


@pytest.mark.parametrize(
    "value",
    (
        Fixed(100.0),
        Link.linear("metal.racah_b"),
    ),
)
def test_slice_non_relativistic_energies_raises_error_if_soc_not_zero(
    value: Fixed | Link,
) -> None:
    with pytest.raises(ValueError) as error:
        metal = MetalAssignment(
            d_electrons=9,
            racah_b=Varied([0.0, 1.0]),
            racah_c=Varied([0.0, 1.0]),
            spin_orbit_coupling=value,
        )
        slice_non_relativistic_energies(
            molecule=Mock(), metal=metal, ligands=[], spin_multiplicity_assignments=[]
        )
    expected_error_message = (
        "metal.spin_orbit_coupling must be set to 0.0, instead got: "
        f"{metal.spin_orbit_coupling=}"
    )
    assert expected_error_message in str(error)
