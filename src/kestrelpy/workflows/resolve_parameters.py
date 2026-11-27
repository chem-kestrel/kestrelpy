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

from collections import defaultdict
from dataclasses import dataclass, fields
from itertools import product
from typing import Any, Iterator, Literal, NamedTuple, Sequence, Type, TypeAlias, cast

from kestrelpy.parameters import (
    Assignment,
    DipoleAssignment,
    DipoleParameters,
    Fixed,
    GeometryAssignment,
    GeometryParameters,
    HyperfineAssignment,
    HyperfineParameters,
    LigandAssignment,
    LigandParameters,
    Link,
    MetalAssignment,
    MetalParameters,
    ParameterVariation,
    Varied,
)

Assignments: TypeAlias = Type[Assignment]


class AssignmentMappings(NamedTuple):
    fixed_parameters: dict[str, float]
    varied_parameters: dict[str, list[float]]
    linked_parameters: dict[str, Link]
    reserved_parameters: dict[str, Any]


def resolve_linked_parameter(
    key: str,
    fixed_values: dict[str, float],
    linked_values: dict[str, Link],
    visited: set[str] | None = None,
) -> float:
    """Recursively resolve a linked value to a float."""
    if visited is None:
        visited = set()
    if key in visited:
        raise ValueError(f"Circular reference detected while resolving {key}")
    visited.add(key)

    # First check if linked-value is linked to another
    linked_value = linked_values.get(key)
    if linked_value:
        target_val = resolve_linked_parameter(
            key=linked_value.object_reference,
            fixed_values=fixed_values,
            linked_values=linked_values,
            visited=visited,
        )
        return linked_value.apply_link(target_val)

    # If the value is not linked to another linked parameter, then it must
    # link to a fixed value.
    value = fixed_values[key]
    if isinstance(value, float):
        return value
    elif isinstance(value, Link):
        target_val = resolve_linked_parameter(
            value.object_reference, fixed_values, visited
        )
        return value.apply_link(target_val)
    else:
        raise TypeError(f"Unexpected value type for {key}: {type(value)}")


def get_resolved_parameters(
    fixed_parameters: dict[str, float],
    varied_parameters: dict[str, list[float]],
    linked_parameters: dict[str, Link],
    reserved_parameters: dict[str, Any],
) -> Iterator[dict[str, float]]:
    final_parameters: dict[str, float] = {**fixed_parameters, **reserved_parameters}
    varied_parameter_keys = varied_parameters.keys()
    for combos in zip(product(*list(varied_parameters.values()))):
        for key, combo in zip(varied_parameter_keys, combos[0], strict=True):
            final_parameters[key] = combo

        for key, link in linked_parameters.items():
            final_parameters[key] = link.apply_link(
                resolve_linked_parameter(
                    key=link.object_reference,
                    fixed_values=final_parameters,
                    linked_values=linked_parameters,
                )
            )

        yield final_parameters


def resolve(parameter_assignments: AssignmentMappings) -> Iterator[dict[str, float]]:
    return get_resolved_parameters(
        fixed_parameters=parameter_assignments.fixed_parameters,
        varied_parameters=parameter_assignments.varied_parameters,
        linked_parameters=parameter_assignments.linked_parameters,
        reserved_parameters=parameter_assignments.reserved_parameters,
    )


def get_object_key(
    assignment: Assignments,
) -> str:
    if isinstance(assignment, DipoleAssignment | LigandAssignment):
        return f"ligand.{assignment.name}"
    elif isinstance(assignment, MetalAssignment):
        return "metal"
    elif isinstance(assignment, HyperfineAssignment):
        return "hyperfine"
    elif isinstance(assignment, GeometryAssignment):
        return f"atom.{assignment.name}"
    else:
        raise TypeError(f"Unrecognised object encountered: {type(assignment)=}")


def _validate_unique_assignment_keys(
    assignments: list[Assignments],
) -> None:
    assignment_keys = [get_object_key(assignment) for assignment in assignments]
    if len(assignments) != len(set(assignment_keys)):
        duplicates = set(
            key for key in assignment_keys if assignment_keys.count(key) > 1
        )
        raise ValueError(f"Duplicate `assignment` keys found: {duplicates}")


def get_key(
    assignment: Assignments,
    parameter: str,
) -> str:
    if not hasattr(assignment, parameter):
        raise ValueError(
            f"Object does not have the referenced parameter: {type(assignment)=}, {parameter=}"
        )
    return f"{get_object_key(assignment=assignment)}.{parameter}"


def get_parameter_variations(
    assignment: Assignments,
    fixed_parameters: dict[str, float],
    varied_parameters: dict[str, list[float]],
    linked_parameters: dict[str, Link],
    reserved_parameters: dict[str, Any],
) -> None:
    """Track the fixed, varied, and linked parameters together."""
    RESERVED_FIELD_NAMES = ["name", "d_electrons", "metal", "units"]
    for f in fields(assignment):
        fname = f.name
        key = get_key(assignment=assignment, parameter=fname)
        value = getattr(assignment, fname)
        if fname in RESERVED_FIELD_NAMES:
            reserved_parameters[key] = value
        elif isinstance(value, Fixed):
            fixed_parameters[key] = value.value
        elif isinstance(value, Varied):
            varied_parameters[key] = value.value
        elif isinstance(value, Link):
            linked_parameters[key] = value
        else:
            raise TypeError(
                f"Invalid field type encountered: {value}. Should be one of {ParameterVariation}"
            )


def map_assignments(assignments: list[Assignments]) -> AssignmentMappings:
    _validate_unique_assignment_keys(assignments)

    fixed_parameters: dict[str, float] = {}
    varied_parameters: dict[str, list[float]] = {}
    linked_parameters: dict[str, Link] = {}
    reserved_parameters: dict[str, Any] = {}  # special arguments

    for assignment in assignments:
        get_parameter_variations(
            assignment,
            fixed_parameters=fixed_parameters,
            varied_parameters=varied_parameters,
            linked_parameters=linked_parameters,
            reserved_parameters=reserved_parameters,
        )

    return AssignmentMappings(
        fixed_parameters,
        varied_parameters,
        linked_parameters,
        reserved_parameters,
    )


class ResolvedLigandFieldParameters(NamedTuple):
    variation: dict[str, float]  # varied parameters
    metal: MetalParameters
    hyperfine: HyperfineParameters
    ligands: list[LigandParameters]
    geometry: list[GeometryParameters]


@dataclass
class Key:
    obj: Literal["metal", "ligand", "hyperfine", "atom"]
    child: str | None
    parameter: str


def key_parser(key: str) -> Key:
    if key.startswith("ligand"):
        _, name, parameter = key.split(".")
        return Key(obj="ligand", child=name, parameter=parameter)
    elif key.startswith("metal"):
        _, parameter = key.split(".")
        return Key(obj="metal", child=None, parameter=parameter)
    elif key.startswith("hyperfine"):
        _, parameter = key.split(".")
        return Key(obj="hyperfine", child=None, parameter=parameter)
    elif key.startswith("atom"):
        _, name, parameter = key.split(".")
        return Key(obj="atom", child=name, parameter=parameter)
    raise ValueError(f"Unrecognised key format: {key}")


def get_ligand_field_parameters(
    assignments: Sequence[
        LigandAssignment | MetalAssignment | HyperfineAssignment | GeometryAssignment
    ],
) -> Iterator[ResolvedLigandFieldParameters]:
    parameter_assignments = map_assignments(
        assignments=cast(list[Assignments], assignments)
    )
    for parameters in resolve(parameter_assignments=parameter_assignments):
        metal_parameters: dict[str, Any] = {}
        hyperfine_parameters: dict[str, Any] = {}
        ligand_parameters: dict[str, dict[str, Any]] = defaultdict(dict)
        geometry_parameters: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in parameters.items():
            parsed_key = key_parser(key=key)
            if parsed_key.obj == "ligand":
                if not parsed_key.child:
                    raise ValueError(
                        f"ligand name was not found in key: {key=}. "
                        "This should not happen!"
                    )
                ligand_parameters[parsed_key.child][parsed_key.parameter] = value
            elif parsed_key.obj == "metal":
                metal_parameters[parsed_key.parameter] = value
            elif parsed_key.obj == "hyperfine":
                hyperfine_parameters[parsed_key.parameter] = value
            elif parsed_key.obj == "atom":
                if not parsed_key.child:
                    raise ValueError(
                        f"ligand name was not found in key: {key=}. "
                        "This should not happen!"
                    )
                geometry_parameters[parsed_key.child][parsed_key.parameter] = value
            else:
                raise ValueError(f"Unrecognised object key: {key=}")

        yield ResolvedLigandFieldParameters(
            variation={
                k: parameters[k] for k in parameter_assignments.varied_parameters.keys()
            },
            metal=MetalParameters(**metal_parameters),
            ligands=[
                LigandParameters(**values) for values in ligand_parameters.values()
            ],
            hyperfine=HyperfineParameters(**hyperfine_parameters),
            geometry=[
                GeometryParameters(**values) for values in geometry_parameters.values()
            ],
        )


def get_dipole_parameters(
    assignments: list[DipoleAssignment],
) -> Iterator[list[DipoleParameters]]:
    parameter_assignments = map_assignments(
        assignments=cast(list[Assignments], assignments)
    )
    for parameters in resolve(parameter_assignments=parameter_assignments):
        dipole_parameters: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in parameters.items():
            parsed_key = key_parser(key=key)
            if parsed_key.obj == "ligand":
                if not parsed_key.child:
                    raise ValueError(
                        f"ligand name was not found in key: {key=}. "
                        "This should not happen!"
                    )
                dipole_parameters[parsed_key.child][parsed_key.parameter] = value
            else:
                raise ValueError(f"Unrecognised object key: {key=}")
        yield [DipoleParameters(**values) for values in dipole_parameters.values()]
