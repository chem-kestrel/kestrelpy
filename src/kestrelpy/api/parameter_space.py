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

from pathlib import Path

from kestrelpy.comparison.data import EprData, SpinMultiplicityEnergies
from kestrelpy.models.input_models.input_models import PrintingBlock
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.models.results.parameter_space import (
    ParameterSpaceData,
    ParameterSpaceResults,
)
from kestrelpy.parameters import LigandAssignment, MetalAssignment
from kestrelpy.workflows.parameter_space import get_parameter_space_from_file
from kestrelpy.workflows.parameter_space import (
    slice_epr_properties as _slice_epr_properties,
)
from kestrelpy.workflows.parameter_space import (
    slice_non_relativistic_energies as _slice_non_relativistic_energies,
)


def analyse_parameter_space_from_file(file_path: str | Path) -> ParameterSpaceData:
    """Get the parameter space data from a parameter space run CSV file.

    Example:

    >>> import kestrelpy as kp
    >>> parameter_space = kp.slice_non_relativistic_energies(
        molecule=...,
        metal=kp.MetalAssignment(
            d_electrons=7,
            racah_b=kp.Varied([500.0, 600.0, 700.0]),
        ),
        ligands=[
            kp.LigandAssignment(
                name="NH3",
                esigma=kp.Varied([5000.0, 5500.0, 6000.0]),
            ),
            kp.LigandAssignment(
                name="Cl",
                esigma=kp.Varied([4000.0, 4500.0, 5000.0]),
                epix=...,
                epiy=...,
            ),
        ],
        ...,
        printing=PrintingBlock(output_file="parameter_space.out"),
    )
    >>> data = kp.analyse_parameter_space_from_file(
        file_path="parameter_space.csv",
    )
    >>> x, y, z = data.get_2d_slice(
        "ligand.NH3.esigma", "ligand.Cl.esigma", fixed_values={"metal.racah_b": 500.0}
    )

    and then plot using which ever library you wish. For example, with matplotlib.pyplot:
    ```
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    c = ax.contourf(x, y, z, cmap='RdBu', levels=100)

    # set the limits of the plot to the limits of the data
    # and add a colourbar
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()
    ```
    """
    return get_parameter_space_from_file(file_path=file_path)


def slice_non_relativistic_energies(
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    spin_multiplicity_assignments: list[SpinMultiplicityEnergies],
    printing: PrintingBlock | None = None,
) -> ParameterSpaceResults:
    """
    Vary parameters and compute the root-mean-square of the non-relativistic transition energies.

    NOTE: non-relativistic means that spin-orbit coupling is set to 0.0. Attempts to vary or set
    a non-zero value for the spin-orbit coupling will raise an error.
    """
    return _slice_non_relativistic_energies(
        molecule=molecule,
        metal=metal,
        ligands=ligands,
        spin_multiplicity_assignments=spin_multiplicity_assignments,
        printing=printing,
    )


def slice_epr_data(
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    epr_data: EprData,
    printing: PrintingBlock | None = None,
) -> ParameterSpaceResults:
    """Vary parameters and compute the root-mean-square diff of the EPR properties."""
    return _slice_epr_properties(
        molecule=molecule,
        metal=metal,
        ligands=ligands,
        epr_data=epr_data,
        printing=printing,
    )
