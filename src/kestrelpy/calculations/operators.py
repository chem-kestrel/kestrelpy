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

from enum import Enum


class Operators:
    class SphericalHarmonic(str, Enum):
        Y_2_2 = "Y_2_2"
        Y_2_1 = "Y_2_1"
        Y_2_0 = "Y_2_0"
        Y_2_n1 = "Y_2_n1"
        Y_2_n2 = "Y_2_n2"
        Y_4_4 = "Y_4_4"
        Y_4_3 = "Y_4_3"
        Y_4_2 = "Y_4_2"
        Y_4_1 = "Y_4_1"
        Y_4_0 = "Y_4_0"
        Y_4_n1 = "Y_4_n1"
        Y_4_n2 = "Y_4_n2"
        Y_4_n3 = "Y_4_n3"
        Y_4_n4 = "Y_4_n4"

    class InterelectronicRepulsion(str, Enum):
        F2 = "F2"
        F4 = "F4"

    class SpinOrbitCoupling(str, Enum):
        SOC = "SOC"

    class OrbitalAngularMomentum(str, Enum):
        LX = "LX"
        LY = "LY"
        LZ = "LZ"

    class SpinAngularMomentum(str, Enum):
        SX = "SX"
        SY = "SY"
        SZ = "SZ"
