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

import dataclasses
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from kestrelpy.calculations.rotation_matrix import get_rotation_matrix
from kestrelpy.calculations.spherical_polar_coordinates import (
    calculate_direction_cosine_matrix,
)
from kestrelpy.parameters import GeometryParameters


@dataclass(frozen=True)
class Atom:
    index: int
    element: str
    coordinates: tuple[float, float, float]


@dataclass(frozen=True)
class Assignment:
    theta: float
    phi: float
    psi: float
    coordinating_atom: Atom
    local_x_orientation: Atom | None = None
    name: str = ""

    @property
    def local_frame(self) -> np.ndarray:
        """Return the local ligand x, y, and z frame."""
        return np.round(
            calculate_direction_cosine_matrix(
                theta=self.theta,
                phi=self.phi,
                psi=self.psi,
                degrees=True,
            ),
            decimals=3,
        )

    @cached_property
    def ligand_field_rotation_matrix(self) -> np.ndarray:
        """Return the 5x5 ligand-field rotation matrix."""
        return get_rotation_matrix(
            tta=self.theta,
            phi=self.phi,
            psi=self.psi,
            degrees=True,
        )


@dataclass(frozen=True)
class MolecularGeometry:
    file_name: str
    xyz_text: str
    atoms: tuple[Atom, ...]


@dataclass(frozen=True)
class Ligand:
    name: str
    atoms: tuple[Assignment, ...]


class InvalidLigandNameError(Exception):
    """Raised when an input ligand name cannot be found."""

    pass


@dataclass(frozen=True)
class Molecule:
    """Molecule class.

    Container to represent a `Molecule`, with a central metal ion
    and surrounding ligands.
    """

    molecular_geometry: MolecularGeometry
    metal: Atom
    ligands: tuple[Ligand, ...]

    def get_ligand(self, name: str) -> Ligand:
        """Return the ligand with the given name. The input name is case sensitive."""
        for ligand in self.ligands:
            if ligand.name == name:
                return ligand
        raise InvalidLigandNameError(
            f"Cannot find ligand with {name=}. Valid ligand names: {[ligand.name for ligand in self.ligands]}"
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def save_molecule(file: str | Path, molecule: Molecule) -> None:
    """Save a molecule to file.

    Args:
        file: The file path.
        molecule: The molecule to save.
    """
    serialised_molecule = molecule.to_json()
    with open(file, "w") as f:
        json.dump(serialised_molecule, f, indent=2)


def load_molecule(file: str | Path) -> Molecule:
    """Load a molecule object from a file.

    Args:
        file: The file path.

    Returns:
        The Molecule object for that file.
    """
    with open(file, "r") as f:
        data = json.load(f)

    molecular_geometry = MolecularGeometry(
        file_name=data["molecular_geometry"]["file_name"],
        xyz_text=data["molecular_geometry"]["xyz_text"],
        atoms=tuple(
            Atom(
                index=item["index"],
                element=item["element"],
                coordinates=tuple(item["coordinates"]),
            )
            for item in data["molecular_geometry"]["atoms"]
        ),
    )
    metal = Atom(
        index=data["metal"]["index"],
        element=data["metal"]["element"],
        coordinates=tuple(data["metal"]["coordinates"]),
    )
    ligands = tuple(
        Ligand(
            name=item["name"],
            atoms=tuple(
                Assignment(
                    theta=subitem["theta"],
                    phi=subitem["phi"],
                    psi=subitem["psi"],
                    coordinating_atom=Atom(
                        index=subitem["coordinating_atom"]["index"],
                        element=subitem["coordinating_atom"]["element"],
                        coordinates=tuple(subitem["coordinating_atom"]["coordinates"]),
                    ),
                    local_x_orientation=Atom(
                        index=subitem["local_x_orientation"]["index"],
                        element=subitem["local_x_orientation"]["element"],
                        coordinates=tuple(
                            subitem["local_x_orientation"]["coordinates"]
                        ),
                    )
                    if subitem.get("local_x_orientation")
                    else None,
                )
                for subitem in item["atoms"]
            ),
        )
        for item in data["ligands"]
    )

    return Molecule(
        molecular_geometry=molecular_geometry,
        metal=metal,
        ligands=ligands,
    )


def change_molecule_geometry(
    molecule: Molecule, geometry_parameters: list[GeometryParameters]
) -> Molecule:
    """Change ligand assignments if a geometric parameter (theta/phi/psi) is changed.

    Args:
        molecule: Molecule class
        geometry_parameters: List of geometric parameters per ligand atom

    Returns:
        Molecule object with updated Ligand Assignment
    """
    parameter_lookup: dict[str, Any] = {g.name: g for g in geometry_parameters}
    new_ligands = []

    def unwrap(x: Any) -> Any:
        return x.value if hasattr(x, "value") else x

    # Update theta/phi/psi for each ligand if changed.
    for ligand in molecule.ligands:
        new_assignments = []
        for i, assignment in enumerate(ligand.atoms):
            key = f"{ligand.name}{i+1}"
            if key in parameter_lookup:
                geometry_param = parameter_lookup[key]
                new_assignments.append(
                    dataclasses.replace(
                        assignment,
                        theta=unwrap(geometry_param.theta),
                        phi=unwrap(geometry_param.phi),
                        psi=unwrap(geometry_param.psi),
                    )
                )
            else:
                new_assignments.append(assignment)

        new_ligands.append(dataclasses.replace(ligand, atoms=tuple(new_assignments)))
    # Replace ligands attribute of the Molecule
    return dataclasses.replace(molecule, ligands=tuple(new_ligands))
