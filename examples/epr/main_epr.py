"""Simple example of Cu(II) ethylenediamine (Cu(en)2)

This script constructs the molecule from a .xyz file, assigns parameter values
and then calculates its EPR properties. These are compared to experimental
and DFT values.

.xyz coordinates, experimental and DFT calculated data are taken from:
    1. https://doi.org/10.1063/1.1540619
        J. Chem. Phys., 118, 3939-3948,
    2. https://pubs.acs.org/doi/10.1021/acs.jctc.8b00513
        J. Chem. Theory Comput. 2018, 14, 9, 4662-4677
"""

from pathlib import Path

import kestrelpy as kp
from kestrelpy.models.input_models.input_models import HyperfineMode

WORKING_DIRECTORY = Path(__file__).parent
XYZ_FILE = str(WORKING_DIRECTORY / "cu_en2.xyz")

calculation = kp.run_calculation(
    # Build the molecule from the xyz file
    molecule=kp.BuildMoleculeFromFile(
        file_path=XYZ_FILE,
        metal_index=0,
        ligands=[
            kp.BuildLigand(
                name="en",
                atoms=[
                    kp.AssignAtom(coordinating_atom=1),
                    kp.AssignAtom(coordinating_atom=4),
                    kp.AssignAtom(coordinating_atom=13),
                    kp.AssignAtom(coordinating_atom=20),
                ],
            )
        ],
    ),
    # Define the metal centre.
    # The metal is Cu(II) which is a d9 system. Spin-orbit coupling
    # is set to the free-ion value of 830 cm-1. Racah B/C can be set to 0 cm-1
    # as there is only one unpaired electron
    metal=kp.MetalAssignment(
        d_electrons=9,
        racah_b=kp.Fixed(0.0),
        racah_c=kp.Fixed(0.0),
        spin_orbit_coupling=kp.Fixed(830.0),
        kiso=kp.Fixed(0.69),
    ),
    # Ethylenediamine is expected to be a (strong) sigma-donor only ligand
    ligands=[kp.LigandAssignment(name="en", esigma=kp.Fixed(7400.0))],
    # We will turn off configuration projection for this calculation
    calculation=kp.CalculationBlock(configuration_projection=False),
    # For a d1/d9 system, EPR is calculated over the ground state (levels 0 and 1).
    # This can be set manually with the selection option or, if no levels are provided
    # determined from the thermal population and spin projections of each state.
    # Here we use the latter.
    #
    # - For hyperfine calculations we will use a calculated value for the
    #   parameters P and kappa, this is the recommended approach for initial fits.
    # - After initial fitting it is reasonable to vary kappa by +-10% to obtain
    #   the best fit.
    # - We will calculate the hyperfine values for each contribution (Fermi contact,
    #   dipolar and orbital) along with their signs using HyperfineMode.ALL
    epr=kp.EPRBlock(
        hyperfine_mode=HyperfineMode.ALL,
        predict_p=True,
        predict_kappa=True,
        hyperfine_parameters=kp.HyperfineAssignment(metal="63Cu"),  # Use 63Cu isotope
    ),
    # Write the output to 'cu_en2.out'
    printing=kp.PrintingBlock(
        output_file=str(WORKING_DIRECTORY / "cu_en2.out"), molecule_as_json=True
    ),
)

# HYPERFINE CALCULATION RESULTS
print(calculation.results[0].epr[0].g_factors)
# Print a table giving g values and the g tensor orientation.
# For comparison, experimental g-values are: 2.040, 2.040, 2.205  (ref 2)

print(calculation.results[0].epr[0].metal_hyperfine_coupling.sign)  # type: ignore
# Print a table showing the calculated A values, their fermi contact, dipolar
# and spin-orbit contributions and the orientation of the A tensor.

# Experimental A values and contribution values from DFT (B3LYP functional)
# for comparison with output (all values in MHz) (ref 1):
#   A values: A1 = 83, A2 = 83, A3 = 608
#   Fermi contact: -347
#   Dipolar coupling: Adip1 = 228, Adip2 = 229, Adip3 = -458
#   Orbital contribution: Aorb1 = 42, Aorb2 = 42, Aorb3 = 150
