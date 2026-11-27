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

from kestrelpy.calculations.eigenfunction_indices import get_thermal_weighted_epr_levels
from kestrelpy.calculations.magnetism.susceptibilities import (
    ParamagneticSusceptibilityTensors,
)
from kestrelpy.models.input_models.input_models import (
    CalculationBlock,
    EPRBlock,
    IntensityBlock,
    PrintingBlock,
    SusceptibilityBlock,
)
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.models.results.calculations import (
    CalculationResults,
    ElectronicStructure,
    EPRProperties,
    Input,
    IntensityResults,
    Output,
)
from kestrelpy.molecule import Molecule
from kestrelpy.output import (
    write_electronic_structure_results,
    write_molecule_to_file,
    write_spectra_to_files,
)
from kestrelpy.parameters import LigandAssignment, MetalAssignment
from kestrelpy.workflows.calculate_electronic_structure import (
    calculate_electronic_structure,
)
from kestrelpy.workflows.calculate_intensities import calculate_intensities
from kestrelpy.workflows.calculate_magnetic_properties import (
    calculate_epr_properties,
    calculate_susceptibilities,
    calculate_thermal_epr_properties,
)
from kestrelpy.workflows.get_molecule import get_molecule


def run_calculation_with_molecule(
    molecule: Molecule,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    calculation: CalculationBlock = CalculationBlock(),
    epr: EPRBlock | None = None,
    susceptibilities: SusceptibilityBlock | None = None,
    intensities: IntensityBlock | None = None,
) -> list[CalculationResults]:
    electronic_structure_calculations = calculate_electronic_structure(
        molecule=molecule,
        metal=metal,
        ligands=ligands,
        hyperfine=getattr(epr, "hyperfine_parameters", None),
        calculation=calculation,
    )

    results: list[CalculationResults] = []
    for electronic_structure_calculation in electronic_structure_calculations:
        # Initialise potential results
        intensity_results: list[IntensityResults] = []
        epr_results: list[EPRProperties] = []
        magnetism_result: ParamagneticSusceptibilityTensors | None = None

        if intensities is not None:
            intensity_results = calculate_intensities(
                calculation=electronic_structure_calculation,
                intensities=intensities,
            )
        if epr and epr.selection:
            # Determine EPR states from user-input manifolds or levels
            if epr.selection.manifolds:
                for indices in epr.selection.manifolds:
                    start, final = indices
                    epr_results.append(
                        calculate_epr_properties(
                            calculation=electronic_structure_calculation.calculation,
                            system=electronic_structure_calculation,
                            start=start,
                            final=final,
                            calculate_hfc=epr.hyperfine_mode,
                            predict_p=epr.predict_p,
                            predict_kappa=epr.predict_kappa,
                        ),
                    )
            levels = (
                electronic_structure_calculation.calculation.eigenfunctions.get_levels(
                    energetic_difference=epr.degeneracy_tolerance
                )
            )
            for level_index in epr.selection.levels:
                eigenfunctions = [f.index for f in levels[level_index].eigenfunctions]
                start, final = min(eigenfunctions), max(eigenfunctions)
                epr_results.append(
                    calculate_epr_properties(
                        calculation=electronic_structure_calculation.calculation,
                        system=electronic_structure_calculation,
                        start=start,
                        final=final,
                        calculate_hfc=epr.hyperfine_mode,
                        predict_p=epr.predict_p,
                        predict_kappa=epr.predict_kappa,
                    ),
                )
        elif epr:
            # Determine states from thermally populated levels if no levels are specified
            soc_levels, nosoc_levels = get_thermal_weighted_epr_levels(
                calculation=electronic_structure_calculation.calculation,
                temperature_k=epr.temperature,
            )
            epr_results.append(
                calculate_thermal_epr_properties(
                    calculation=electronic_structure_calculation.calculation,
                    epr_levels_soc=soc_levels,
                    epr_levels_nosoc=nosoc_levels,
                    g_factor_type=epr.g_type,
                    system=electronic_structure_calculation,
                    calculate_hfc=epr.hyperfine_mode,
                    predict_p=epr.predict_p,
                    predict_kappa=epr.predict_kappa,
                )
            )

        if susceptibilities is not None:
            magnetism_result = calculate_susceptibilities(
                calculation=electronic_structure_calculation.calculation,
                temperatures=susceptibilities.temperatures,
                second_order_zeeman_energy_cutoff=susceptibilities.soz_max_energetic_limit,
            )

        results.append(
            CalculationResults(
                metal=electronic_structure_calculation.metal_parameters,
                ligands=electronic_structure_calculation.ligand_parameters,
                calculation=ElectronicStructure(
                    parameters=electronic_structure_calculation.calculation.parameters,
                    eigenfunctions=electronic_structure_calculation.calculation.eigenfunctions,
                ),
                epr=epr_results,
                susceptibilities=magnetism_result,
                intensities=intensity_results,
            )
        )

    return results


def run_calculation(
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    calculation: CalculationBlock = CalculationBlock(),
    intensities: IntensityBlock | None = None,
    epr: EPRBlock | None = None,
    susceptibilities: SusceptibilityBlock | None = None,
    printing: PrintingBlock | None = None,
) -> Output:
    _molecule: Molecule = get_molecule(instructions=molecule)
    results = run_calculation_with_molecule(
        molecule=_molecule,
        metal=metal,
        ligands=ligands,
        calculation=calculation,
        epr=epr,
        susceptibilities=susceptibilities,
        intensities=intensities,
    )

    output = Output(
        input=Input(
            molecule=molecule,
            metal=metal,
            ligands=ligands,
            calculation=calculation,
            epr=epr,
            susceptibilities=susceptibilities,
            intensities=intensities,
        ),
        molecule=_molecule,
        results=results,
    )

    if printing:
        write_electronic_structure_results(
            file_path=printing.output_file, output=output
        )
    if printing and printing.write_spectra:
        write_spectra_to_files(file_path=printing.output_file, output=output)
    if printing and printing.molecule_as_json:
        write_molecule_to_file(file_path=printing.output_file, molecule=_molecule)

    return output
