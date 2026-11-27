import numpy as np
from tabulate import tabulate

from kestrelpy.calculations.ligand_field_matrix import DOrbitalBasis


def format_ligand_field_matrix_comparison(
    calculated_matrix: np.ndarray,
    reference_matrix: np.ndarray,
) -> str:
    A = calculated_matrix
    B = reference_matrix

    rows = []
    for i in range(5):
        for j in range(i, 5):
            rows.append(
                [
                    f"< {DOrbitalBasis(i).name:^5} | V | {DOrbitalBasis(j).name:^5} >",
                    f"{A[i, j]:.1f}",
                    f"{B[i, j]:.1f}",
                    f"{B[i, j] - A[i, j]:.1f}",
                ]
            )

    return tabulate(
        tabular_data=rows,
        headers=[
            "<Di|V|Dj>",
            "Calculated (cm-1)",
            "Reference (cm-1)",
            "Residual (cm-1)",
        ],
    )
