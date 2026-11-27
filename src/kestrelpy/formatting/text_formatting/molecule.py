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

from textwrap import indent

from kestrelpy.molecule import Assignment, Atom, Ligand, MolecularGeometry, Molecule


def format_molecular_geometry(geometry: MolecularGeometry) -> str:
    xyz_text_section = f"Atoms (XYZ format):\n```\n{geometry.xyz_text.strip()}\n```\n"
    table_rows: list[str] = []
    for atom in geometry.atoms:
        x, y, z = atom.coordinates
        table_rows.append(
            f"  {atom.index:5d}   {atom.element:<7}   {x:10.5f}   {y:10.5f}   {z:10.5f}"
        )
    parsed_atom_table = (
        "Parsed Atoms:\n"
        f"{'-'*60}\n"
        f"  {'Index':>5}   {'Element':<7}   {'X':>10}   {'Y':>10}   {'Z':>10}\n"
        f"{'-'*60}\n" + "\n".join(table_rows) + "\n"
        f"{'-'*60}"
    )

    return "\n".join([xyz_text_section, parsed_atom_table])


def format_atom(atom: Atom) -> str:
    return "\n".join(
        [
            f"- index:       {atom.index}",
            f"  element:     {atom.element}",
            f"  coordinates: {atom.coordinates}",
        ]
    )


def format_ligand_assignment(assignment: Assignment) -> str:
    local_frame = assignment.local_frame
    return "\n".join(
        [
            f"- theta: {assignment.theta: .1f}",
            f"  phi:   {assignment.phi: .1f}",
            f"  psi:   {assignment.psi: .1f}",
            "  coordinating_atom:",
            indent(format_atom(assignment.coordinating_atom), "  "),
            (
                f"  local_x_orientation: {None}"
                if assignment.local_x_orientation is None
                else (
                    "  local_x_orientation:\n"
                    + indent(format_atom(assignment.local_x_orientation), "  ")
                )
            ),
            "  local_frame:",
            indent(", ".join(f"{lf: .3f}" for lf in local_frame[:, 0]), "    "),
            indent(", ".join(f"{lf: .3f}" for lf in local_frame[:, 1]), "    "),
            indent(", ".join(f"{lf: .3f}" for lf in local_frame[:, 2]), "    "),
        ]
    )


def format_ligand(ligand: Ligand) -> str:
    return "\n".join(
        [
            f"Ligand `{ligand.name}`:",
            *[
                f"- Assignment {i}:\n"
                f"{indent(format_ligand_assignment(assignment), '  ')}"
                for i, assignment in enumerate(ligand.atoms)
            ],
        ]
    )


def format_molecule(molecule: Molecule) -> str:
    return "\n".join(
        [
            format_molecular_geometry(geometry=molecule.molecular_geometry),
            "Metal:",
            format_atom(molecule.metal) + "\n",
            *[format_ligand(ligand) + "\n" for ligand in molecule.ligands],
        ]
    )
