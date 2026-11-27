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

from kestrelpy.comparison.data import EprData, SpinMultiplicityEnergies


def format_energies(energies: SpinMultiplicityEnergies) -> str:
    return "\n".join(
        [
            f"spin_multiplicity:    {energies.spin_multiplicity}",
            f"is_ordered:           {energies.is_ordered}",
            "energies:",
            *[f"  {energy}" for energy in energies.energies],
            "",
        ]
    )


def format_spin_multiplicty_energies(energies: list[SpinMultiplicityEnergies]) -> str:
    return "\n".join(
        [
            "*** Spin-Multiplicity Energies: ***",
            "",
            *[format_energies(block) for block in energies],
        ]
    )


def format_epr_data(epr_data: EprData) -> str:
    return "\n".join(
        [
            "*** EPR Data: ***",
            "",
            f"Electronic level:     {epr_data.level}",
            f"g1:                   {epr_data.g_factors[0]:}",
            f"g2:                   {epr_data.g_factors[1]:}",
            f"g3:                   {epr_data.g_factors[2]:}",
        ]
    )
