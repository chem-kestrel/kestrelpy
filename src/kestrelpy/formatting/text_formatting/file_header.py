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

from kestrelpy.configuration import get_package_version

FILE_BANNER_TEMPLATE = """
                                         KESTREL
                                A ligand-field program for
                              3d transition-metal complexes

{divider}
{version_copyright}
{university}
{divider}

    References:
        1.  Nunn, George (2023) Development of a contemporary ligand-field
            program for 3d transition-metal complexes. PhD thesis,
            University of York.
        2.  P. Comba, G. Nunn, F. Scherz and P. H. Walton, Faraday Discuss., 2021, 234, 232-244.
        3.  S. Mace, B. S. Grace, A. Goodfellow, M. A. Bates, M. A. Halcrow, P. H .Walton,
            Y. Xu and B. N. Nguyen, Inorg. Chem. Front., 2026, 13 (17), 7301-7315.


    Contributors:
        - George Nunn: Former PhD student in the PHW group.
        - Ben Grace: PhD student in the PHW group.
        - Martin Bates: PhD project supervisor.
        - Paul Walton: PhD project supervisor.


{calculation_type}
"""


def get_file_banner(calculation_type: str) -> str:
    return FILE_BANNER_TEMPLATE.format(
        version_copyright=f"|        Version: {get_package_version().strip()}, Copyright (c) 2026        |".center(
            90
        ),
        divider="----------------------------------------------------".center(90),
        university="| Department of Chemistry, University of York (UK) |".center(90),
        calculation_type=f"*** {calculation_type} ***".center(90),
    )
