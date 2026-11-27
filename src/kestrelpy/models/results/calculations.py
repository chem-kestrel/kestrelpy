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

from dataclasses import dataclass, field

from kestrelpy.calculations.intensities.absorption import Absorption
from kestrelpy.calculations.intensities.circular_dichroism import CircularDichroism
from kestrelpy.calculations.intensities.electronic_transition_dipole_moment import (
    GlobalTransitionDipoleMultipoleExpansion,
)
from kestrelpy.calculations.intensities.magnetic_circular_dichroism import (
    MagneticCircularDichroism,
)
from kestrelpy.calculations.intensities.spectroscopy import Spectrum
from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.magnetism.susceptibilities import (
    ParamagneticSusceptibilityTensors,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ManyElectronEigenfunctions,
)
from kestrelpy.models.input_models.input_models import (
    CalculationBlock,
    EPRBlock,
    IntensityBlock,
    Parameters,
    PrintingBlock,
    SusceptibilityBlock,
)
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.molecule import Molecule
from kestrelpy.parameters import (
    DipoleParameters,
    LigandAssignment,
    LigandParameters,
    MetalAssignment,
    MetalParameters,
)
from kestrelpy.workflows.calculate_hyperfine_interaction import HyperfineResults


@dataclass(frozen=True)
class ElectronicStructure:
    parameters: Parameters
    eigenfunctions: ManyElectronEigenfunctions


@dataclass
class EPRProperties:
    start: int
    final: int
    g_factors: GSquaredTensor
    temperature: float | None = None
    metal_hyperfine_coupling: HyperfineResults | None = None


@dataclass(frozen=True)
class AbsorptionResults:
    results: Absorption
    spectra: list[Spectrum]


@dataclass(frozen=True)
class CircularDichroismResults:
    results: CircularDichroism
    spectra: list[Spectrum]


@dataclass(frozen=True)
class MagneticCircularDichroismResults:
    results: MagneticCircularDichroism
    spectra: list[Spectrum]


@dataclass(frozen=True)
class IntensityResults:
    dipole_parameters: list[DipoleParameters]
    global_transition_dipole_moment_expansion: GlobalTransitionDipoleMultipoleExpansion
    electronic_transition_dipole_moment_expansion: list[list[complex]]

    # results
    absorption: AbsorptionResults | None = None
    cd: CircularDichroismResults | None = None
    mcd: list[MagneticCircularDichroismResults] = field(default_factory=list)


@dataclass(frozen=True)
class CalculationResults:
    metal: MetalParameters
    ligands: list[LigandParameters]
    calculation: ElectronicStructure
    epr: list[EPRProperties] = field(default_factory=list)
    susceptibilities: ParamagneticSusceptibilityTensors | None = None
    intensities: list[IntensityResults] = field(default_factory=list)


@dataclass(frozen=True)
class Input:
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile
    metal: MetalAssignment
    ligands: list[LigandAssignment]
    calculation: CalculationBlock
    intensities: IntensityBlock | None = None
    epr: EPRBlock | None = None
    susceptibilities: SusceptibilityBlock | None = None
    printing: PrintingBlock | None = None


@dataclass(frozen=True)
class Output:
    input: Input
    molecule: Molecule
    results: list[CalculationResults]
