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

import csv
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

from kestrelpy.comparison.data import EprData, SpinMultiplicityEnergies
from kestrelpy.comparison.energies import fit_transition_energies
from kestrelpy.models.input_models.input_models import PrintingBlock
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.models.results.parameter_space import (
    EprParameterSpaceInput,
    NonRelativisticEnergiesParameterSpaceInput,
    ParameterSpaceData,
    ParameterSpaceResults,
)
from kestrelpy.output import (
    write_molecule_to_file,
    write_parameter_space_csv,
    write_parameter_space_file,
)
from kestrelpy.parameters import Fixed, LigandAssignment, MetalAssignment, Varied
from kestrelpy.workflows.calculate_electronic_structure import (
    calculate_electronic_structure,
)
from kestrelpy.workflows.calculate_magnetic_properties import calculate_epr_properties
from kestrelpy.workflows.get_molecule import get_molecule
from kestrelpy.workflows.resolve_parameters import get_object_key


def _validate_input_arguments(
    metal: MetalAssignment, ligands: list[LigandAssignment]
) -> None:
    number_of_varied_parameters: int = 0
    varied_keys: list[str] = []
    for assignment in [metal, *ligands]:
        for f in fields(assignment):
            value = getattr(assignment, f.name)
            if isinstance(value, Varied):
                varied_keys.append(get_object_key(assignment=assignment))  # type: ignore[arg-type]
                number_of_varied_parameters += 1
    if number_of_varied_parameters < 1:
        raise ValueError(
            "You must vary at least one parameter, "
            f"instead got: {number_of_varied_parameters=}"
        )


def slice_non_relativistic_energies(
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    spin_multiplicity_assignments: list[SpinMultiplicityEnergies],
    printing: PrintingBlock | None = None,
) -> ParameterSpaceResults:
    _validate_input_arguments(metal, ligands)

    # Validate spin-orbit coupling is zero!
    if metal.spin_orbit_coupling != Fixed(0.0):
        raise ValueError(
            "metal.spin_orbit_coupling must be set to 0.0, instead got: "
            f"{metal.spin_orbit_coupling=}"
        )

    _molecule = get_molecule(instructions=molecule)
    data: list[dict[str, float]] = []
    varied_values: dict[str, list[float]] = defaultdict(list)
    for electronic_structure in calculate_electronic_structure(
        molecule=_molecule,
        metal=metal,
        ligands=ligands,
    ):
        # varied values...
        for k, v in electronic_structure.variation.items():
            if v not in varied_values[k]:
                varied_values[k].append(float(v))
        rms = fit_transition_energies(
            spin_multiplicity_assignments=spin_multiplicity_assignments,
            eigenfunctions=electronic_structure.calculation.eigenfunctions,
        )
        data.append({**electronic_structure.variation, "rms": rms})

    results = ParameterSpaceResults(
        input=NonRelativisticEnergiesParameterSpaceInput(
            molecule=molecule,
            metal=metal,
            ligands=ligands,
            printing=printing,
            spin_multiplicity_assignments=spin_multiplicity_assignments,
        ),
        molecule=_molecule,
        data=ParameterSpaceData(
            varied_values=varied_values,
            rows=data,
        ),
    )

    if printing:
        write_parameter_space_csv(file_path=printing.output_file, results=data)
        write_parameter_space_file(file_path=printing.output_file, results=results)
    if printing and printing.molecule_as_json:
        write_molecule_to_file(file_path=printing.output_file, molecule=_molecule)

    return results


def slice_epr_properties(
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    epr_data: EprData,
    printing: PrintingBlock | None = None,
) -> ParameterSpaceResults:
    _validate_input_arguments(metal, ligands)
    _molecule = get_molecule(instructions=molecule)
    data: list[dict[str, float]] = []
    varied_values: dict[str, list[float]] = defaultdict(list)
    for electronic_structure in calculate_electronic_structure(
        molecule=_molecule,
        metal=metal,
        ligands=ligands,
    ):
        # varied values...
        for k, v in electronic_structure.variation.items():
            if v not in varied_values[k]:
                varied_values[k].append(float(v))

        eigenfunctions = [
            f.index
            for f in electronic_structure.calculation.eigenfunctions.levels[
                epr_data.level
            ].eigenfunctions
        ]
        start, final = min(eigenfunctions), max(eigenfunctions)
        epr_properties = calculate_epr_properties(
            calculation=electronic_structure.calculation,
            start=start,
            final=final,
        )
        rms = (
            (1 / 3)
            * (
                sum(
                    (calc - data) ** 2
                    for calc, data in zip(
                        epr_properties.g_factors.values, epr_data.g_factors, strict=True
                    )
                )
            )
        ) ** 0.5
        data.append({**electronic_structure.variation, "rms": rms})

    results = ParameterSpaceResults(
        input=EprParameterSpaceInput(
            molecule=molecule,
            metal=metal,
            ligands=ligands,
            printing=printing,
            epr_data=epr_data,
        ),
        molecule=_molecule,
        data=ParameterSpaceData(
            varied_values=varied_values,
            rows=data,
        ),
    )

    if printing:
        write_parameter_space_csv(file_path=printing.output_file, results=data)
        write_parameter_space_file(file_path=printing.output_file, results=results)
    if printing and printing.molecule_as_json:
        write_molecule_to_file(file_path=printing.output_file, molecule=_molecule)

    return results


def get_parameter_space_from_file(file_path: str | Path) -> ParameterSpaceData:
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)[:-1]  # we do not want the "rms" column name

        varied_values: dict[str, list[float]] = defaultdict(list)

        data: list[dict[str, float]] = []
        for row in reader:
            row_data: dict[str, float] = {}
            for item, header_label in zip(row[:-1], headers, strict=True):
                row_data[header_label] = float(item)

                if float(item) not in varied_values[header_label]:
                    varied_values[header_label].append(float(item))
            row_data["rms"] = float(row[-1])
            data.append(row_data)

    # Sort values
    varied_values = {k: sorted(v) for k, v in varied_values.items()}

    return ParameterSpaceData(
        varied_values=varied_values,
        rows=data,
    )
