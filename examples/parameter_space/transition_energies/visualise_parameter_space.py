"""
Simple example for analysing the EPR parameter space example.

Will need matplotlib.pyplot installed to see how this runs. Simply install
in your virtual environment using:
>>> pip install matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt

from kestrelpy.workflows.parameter_space import get_parameter_space_from_file

WORKING_DIRECTORY = Path(__file__).parent
PARAMETER_SPACE_CSV = WORKING_DIRECTORY / "cocl4.csv"


if __name__ == "__main__":
    data = get_parameter_space_from_file(PARAMETER_SPACE_CSV)

    for row in data.filter(rms=10.0):
        # Print rows where the root-mean-square was < 0.005.
        print(row)

    # Take a 2D slice by varying ligand.Cl.esigma and ligand.Cl.epix.
    # You must specify the other values as fixed, otherwise you will
    # get an error.
    x, y, z = data.get_2d_slice(
        "ligand.Cl.esigma", "ligand.Cl.epix", fixed_values={"metal.racah_b": 900.0}
    )

    # Plot using Matplotlib. See the docstring for installation.
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, z, cmap="RdBu", levels=100)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()
