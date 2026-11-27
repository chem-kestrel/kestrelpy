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

from dataclasses import asdict
from typing import Iterator

from kestrelpy.formatting.json_formatting.utils import JsonDict
from kestrelpy.models.results.calculations import Output


def convert_spectra_to_json(output: Output) -> Iterator[JsonDict]:
    for run_id, results in enumerate(output.results):
        for intensity_run_id, intensities in enumerate(results.intensities):
            absorption_spectra = []
            cd_spectra = []
            mcd_spectra = []
            if intensities.absorption:
                for spectra in intensities.absorption.spectra:
                    absorption_spectra.append(asdict(spectra))
            if intensities.cd:
                for spectra in intensities.cd.spectra:
                    cd_spectra.append(asdict(spectra))
            if intensities.mcd:
                for mcd in intensities.mcd:
                    for spectra in mcd.spectra:
                        mcd_spectra.append(asdict(spectra))
            data = {
                "calculation_id": run_id + 1,
                "intensity_id": intensity_run_id + 1,
                "absorption": absorption_spectra,
                "cd": cd_spectra,
                "mcd": mcd_spectra,
            }
            yield data
