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
from pathlib import Path
from typing import Iterator

from kestrelpy.formatting import (
    convert_spectra_to_json,
    format_csv_for_parameter_space,
    format_electronic_structure_file_contents,
    format_fit_ligand_field_matrix_file_contents,
    format_parameter_space_results,
    get_json_string,
)
from kestrelpy.models.results.calculations import Output
from kestrelpy.models.results.optimisation import LigandFieldOptimisationResult
from kestrelpy.models.results.parameter_space import ParameterSpaceResults
from kestrelpy.molecule import Molecule


def write_text_to_file(file_path: str | Path, file_contents: str) -> None:
    with open(Path(file_path), "w") as f:
        f.write(file_contents)


def write_molecule_to_file(file_path: str | Path, molecule: Molecule) -> None:
    _file_path = Path(file_path).with_suffix("")
    new_file_path = f"{_file_path}_molecule.json"
    write_text_to_file(
        file_path=new_file_path, file_contents=get_json_string(molecule.to_json())
    )


def write_electronic_structure_results(file_path: str | Path, output: Output) -> None:
    write_text_to_file(
        file_path=file_path,
        file_contents=format_electronic_structure_file_contents(output),
    )


def write_spectra_to_files(file_path: str | Path, output: Output) -> None:
    _file_path = Path(file_path).with_suffix("")
    for file_contents in convert_spectra_to_json(output):
        run_id = file_contents["calculation_id"]
        intensity_id = file_contents["intensity_id"]

        new_file_path = (
            f"{_file_path}_run{run_id:03d}_intensity{intensity_id:03d}_spectra.json"
        )

        write_text_to_file(
            file_path=new_file_path, file_contents=get_json_string(file_contents)
        )


def write_fit_ligand_field_matrix_results(
    file_path: str | Path, results: LigandFieldOptimisationResult
) -> None:
    write_text_to_file(
        file_path=file_path,
        file_contents=format_fit_ligand_field_matrix_file_contents(results=results),
    )


def write_csv(file_path: str | Path, rows: Iterator[list[str]]) -> None:
    csv_path = Path(file_path).with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_parameter_space_csv(
    file_path: str | Path, results: list[dict[str, float]]
) -> None:
    write_csv(
        file_path=file_path,
        rows=format_csv_for_parameter_space(results=results),
    )


def write_parameter_space_file(
    file_path: str | Path, results: ParameterSpaceResults
) -> None:
    write_text_to_file(
        file_path=Path(file_path).with_suffix(".out"),
        file_contents=format_parameter_space_results(results=results),
    )
