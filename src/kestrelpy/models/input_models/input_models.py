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
from enum import Enum
from typing import Optional

from kestrelpy.calculations.interelectronic_repulsion import InterelectronRepulsion
from kestrelpy.calculations.ligand_field_matrix import LigandField
from kestrelpy.calculations.parameters import (
    MagneticField,
    OrbitalReductionFactors,
    SpinOrbitCoupling,
)
from kestrelpy.parameters import (
    DipoleAssignment,
    HyperfineAssignment,
    HyperfineParameters,
)


@dataclass(frozen=True)
class AbsorptionOptions:
    temperatures: list[float]


@dataclass(frozen=True)
class CDOptions:
    temperatures: list[float]


@dataclass(frozen=True)
class MCDOptions:
    magnetic_fields: list[float]
    temperatures: list[float]
    n_angular_steps: int = 3
    num_processes: int = 0


class HyperfineMode(str, Enum):
    NONE = "none"
    FERMI = "fermi"
    ANISO = "dipolar+orbital"
    ALL = "dipolar+orbital+fermi+signs"


class GFactorType(str, Enum):
    REAL = "real"
    EFFECTIVE = "effective"

    @classmethod
    def _missing_(cls, value: object) -> "GFactorType":
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return cls.EFFECTIVE


@dataclass(frozen=True)
class IntensityBlock:
    dipoles: list[DipoleAssignment]
    absorption: AbsorptionOptions | None = None
    cd: CDOptions | None = None
    mcd: MCDOptions | None = None

    def has_options(self) -> bool:
        return bool(self.absorption or self.cd or self.mcd)


@dataclass(frozen=True)
class EprSelection:
    levels: list[int] = field(default_factory=list)
    manifolds: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.levels and not self.manifolds:
            raise ValueError("Must set 'levels' and/or 'manifolds")
        if not isinstance(self.levels, list):
            raise TypeError(
                f"Expected a {list} for 'levels', instead got {type(self.levels)}"
            )
        for index in self.levels:
            if not isinstance(index, int):
                raise TypeError(
                    f"level index must be a {int}, instead got: {type(index)}"
                )
            if index < 0:
                raise ValueError(
                    f"Each level must be a positive integer, instead got: {index=}"
                )
        if not isinstance(self.manifolds, list):
            raise TypeError(
                f"Expected a {list} for 'manifolds', instead got: {type(self.indices)=}"
            )
        for item in self.manifolds:
            if not isinstance(item, tuple):
                raise TypeError(
                    "Expected a tuple[int] for `manifolds` items, instead got: "
                    f"{item=}"
                )
            if len(item) != 2:
                raise ValueError(
                    "Expected tuple[int] to have a length of 2, instead got: "
                    f"{len(item)=}"
                )
            start, final = item
            if not isinstance(start, int) or not isinstance(final, int):
                raise TypeError(
                    f"Expected pair of indices to be {int}, instead got: "
                    f"{type(start)=}, {type(final)=}"
                )
            if start > final:
                raise ValueError(
                    f"Expected `start` <= `final`, instead got: {start=} > {final=}"
                )


@dataclass(frozen=True)
class EPRBlock:
    hyperfine_parameters: Optional[HyperfineAssignment] = None
    selection: EprSelection | None = None
    g_type: GFactorType = GFactorType.EFFECTIVE
    degeneracy_tolerance: float = 0.1
    hyperfine_mode: HyperfineMode = HyperfineMode.NONE
    predict_p: bool = False
    predict_kappa: bool = False
    temperature: float = 10.0


@dataclass(frozen=True)
class SusceptibilityBlock:
    temperatures: list[float]
    soz_max_energetic_limit: float = 30000.0


@dataclass(frozen=True)
class CalculationBlock:
    configuration_projection: bool = False


@dataclass(frozen=True)
class PrintingBlock:
    output_file: str
    write_spectra: bool = True
    molecule_as_json: bool = True


@dataclass
class Parameters:
    """Class to store parameters required for ligand field calculations"""

    d_electrons: int
    ligand_field: LigandField
    interelectronic_repulsion: InterelectronRepulsion
    spin_orbit_coupling: SpinOrbitCoupling
    magnetic_field: MagneticField
    orbital_reduction_factors: OrbitalReductionFactors
    hyperfine: HyperfineParameters | None = None
