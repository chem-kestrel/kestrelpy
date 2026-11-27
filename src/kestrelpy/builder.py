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

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np

from kestrelpy.calculations.spherical_polar_coordinates import (
    calculate_spherical_polar_coordinates,
)
from kestrelpy.configuration import get_logger
from kestrelpy.molecule import Assignment, Atom, Ligand, MolecularGeometry, Molecule

logger = get_logger(__name__)


def parse_xyz_file(contents: str) -> list[Atom]:
    atoms: list[Atom] = []
    for index, line in enumerate(contents.strip().split("\n")):
        split_line = line.strip().split()

        if len(split_line) != 4:
            logger.warning(
                f"Line {index=}, contents={line} does not split into four parts. "
                "Skipping parsing this line..."
            )
            continue

        element = split_line[0]
        try:
            coordinates = tuple(map(float, split_line[1:]))
        except ValueError:
            logger.error(
                f"Line {index=}, contents={split_line[1:]} could not be mapped to xyz "
                "floating point coordinates."
            )
            continue

        rounded_coordinates = tuple(
            [round(coordinate, 3) for coordinate in coordinates]
        )
        logger.info(
            f"Successfully parsed Atom {len(atoms)}, {element=}, {rounded_coordinates=}"
        )

        atoms.append(
            Atom(
                index=len(atoms),
                element=element,
                coordinates=cast(tuple[float, float, float], coordinates),
            )
        )
    return atoms


def read_xyz_file(file: str | Path) -> str:
    """Read a molecular xyz file.

    Args:
        file: The path to the xyz file.

    Returns:
        The file contents
    """
    try:
        with open(file, "r") as f:
            return f.read()
    except FileNotFoundError:
        return str(file)


class Builder:
    """Setup the ligand-field calculation using a molecule in an xyz file."""

    file_name: str = ""

    def __init__(self, contents: str, metal_index: int) -> None:
        self._contents = contents
        self._atoms = parse_xyz_file(contents=self._contents)
        logger.info(f"Successfully parsed {len(self._atoms)} atoms.")

        self._metal_index = metal_index
        self._metal_atom: Atom = self._atoms[metal_index]
        logger.info(
            f"Metal index = {self._metal_atom.index} "
            f"set as `{self._metal_atom.element}` "
            f"at coordinates: {self._metal_atom.coordinates}"
        )

        self._ligand_data: dict[str, Ligand] = {}

    @classmethod
    def from_xyz_file(cls, file: str | Path, metal_index: int) -> "Builder":
        xyz_file = read_xyz_file(file=file)
        builder = cls(contents=xyz_file, metal_index=metal_index)
        builder.file_name = Path(file).name
        return builder

    @classmethod
    def from_xyz_text(
        cls, xyz_text: str, metal_index: int, file_name: str = ""
    ) -> "Builder":
        builder = cls(contents=xyz_text, metal_index=metal_index)
        builder.file_name = file_name
        return builder

    def add_ligand(self, name: str, *assigned_atoms: Assignment) -> None:
        """Add a new ligand with a unique name.

        The intended use if to combine this method with the `assign_atom` method:
        ```python
        builder = Builder(...)
        builder.add_ligand(
            "NameOfLigand",
            builder.assign_atom(...),
            builder.assign_atom(...),
            ...
        )
        ```

        Args:
            name: The unique name of the ligand.
            *atoms: Iterable of Assignment, which are generated using
                Builder.assign_atom(...) method.
        """
        if name in self._ligand_data:
            raise ValueError(f"{name} is already in use, please choose a unique name")

        if len(assigned_atoms) == 0:
            raise ValueError(
                f"No assignments were made: {len(assigned_atoms)=}. Please add one or more assignments."
            )

        self._ligand_data[name] = Ligand(name=name, atoms=tuple(assigned_atoms))

    def assign_atom(
        self, coordination_index: int, orientation_x_index: int | None = None
    ) -> Assignment:
        """Assign an atom to a ligand.

        This method is intended to be used inside of the `add_ligand` method:
        ```python
        builder = Builder(...)
        builder.add_ligand(
            "NameOfLigand",
            builder.assign_atom(...),
            builder.assign_atom(...),
        )
        ```
        """
        coordination_atom = self._atoms[coordination_index]
        orientation_atom = (
            self._atoms[orientation_x_index] if orientation_x_index else None
        )
        theta, phi, psi = calculate_spherical_polar_coordinates(
            coordinating_atom=np.array(coordination_atom.coordinates),
            local_x_orientation=(
                np.array(orientation_atom.coordinates)
                if orientation_atom is not None
                else None
            ),
            cartesian_origin=np.array(self._metal_atom.coordinates),
        )
        return Assignment(
            theta=round(theta, 3),
            phi=round(phi, 3),
            psi=round(psi, 3),
            coordinating_atom=coordination_atom,
            local_x_orientation=orientation_atom,
        )

    def get_molecule(self) -> Molecule:
        """Build and return the Molecule instance."""
        return Molecule(
            molecular_geometry=MolecularGeometry(
                file_name=self.file_name,
                xyz_text=self._contents,
                atoms=tuple(self._atoms),
            ),
            metal=self._metal_atom,
            ligands=tuple(self._ligand_data.values()),
        )

    def update_coordinates(self) -> None:
        index = self._metal_index
        new_origin = self._atoms[index].coordinates

        atoms = [
            replace(atom, coordinates=tuple(np.array(atom.coordinates) - new_origin))
            for atom in self._atoms
        ]
        self._atoms = atoms
        self._metal_atom = self._atoms[index]
