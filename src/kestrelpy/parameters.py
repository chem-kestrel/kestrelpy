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

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Literal, TypeAlias

import numpy as np
from tabulate import tabulate


class ParameterValueError(Exception):
    """Raised when an invalid parameter value is received."""

    pass


@dataclass(frozen=True)
class BaseTypeChecker:
    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, f.type):
                raise TypeError(
                    f"Invliad parameter type found: {type(value)}. Should be {f.type}"
                )


@dataclass(frozen=True)
class LigandParameters(BaseTypeChecker):
    """
    Encapsulates the ligand-field parameters for a ligand using Angular Overlap Model parameters.

    Each parameter quantifies the strength of a specific metal-ligand interaction in
    wavenumbers (cm-1). These parameters are used to construct the ligand-field
    contribution to the electronic structure of a transition-metal complex.

    The sign of the bonding parameters describes the donor/acceptor behaviour of the interaction.
    A positive e-parameter is a donor contribution. A negative e-parameter is an electron-accepting
    interaction.

    The magntiude of the parameter is a measure of the total bonding strength and encapsulates
    both electrostatic and electrodynamic contributions.

    Attributes:
        name (str):
            The identifying name of the ligand, as defined in the `Molecule` object. The
            name is case-sensitive.

        esigma (float):
            Strength of the sigma-bonding interaction between the ligand and metal.
            This parameter is typically positive and has a magnitude in the range of
            1000 to 10,000 cm-1. This parameter can be negative in regions where there
            are weakly coordinated ligands (e.g. axial ligands in tetragonally elongated
            systems) or an absence of a coordinating atom.

        epix (float):
            Strength of the pi-bondinng interaction between the ligand and metal in the local
            x-direction of the ligand. The parameter can be either positive or negative and
            has a magnitude typically in the range of 0.0 to 3000.0 cm-1. Effective "sigma-only"
            ligands, such as ammonia, have epi fixed at 0.0 cm-1.

        epiy (float):
            Strength of the pi-bondinng interaction between the ligand and metal in the local
            y-direction of the ligand. The parameter can be either positive or negative and
            has a magnitude typically in the range of 0.0 to 3000.0 cm-1. Effective "sigma-only"
            ligands, such as ammonia, have epi fixed at 0.0 cm-1.

        edxy (float):
            Strength of the delta xy interaction. This parameter is typically assumed to be 0 cm-1
            for most ligands. Delta interactions are possible for conjugated arene ligands,
            or when bonding with other metals.

        edx2y2 (float):
            Strength of the delta x2-y2 interaction. This parameter is typically assumed to be 0 cm-1
            for most ligands. Delta interactions are possible for conjugated arene ligands,
            or when bonding with other metals.

        eds (float):
            The magnitude of the stabilising effect from ndz2-(n+1)s mixing. This value can only be
            positive.

    Example:
    >>> ligand_parameters = LigandParameters(esigma=5000.0, epix=2000.0)
    >>> ligand_parameters.esigma
    5000.0
    >>> ligand_parameters.ligand_field_matrix()
    array(
        [
            [   0.,    0.,    0.,    0.,    0.],
            [   0.,    0.,    0.,    0.,    0.],
            [   0.,    0., 5000.,    0.,    0.],
            [   0.,    0.,    0., 2000.,    0.],
            [   0.,    0.,    0.,    0.,    0.],
        ]
    )
    """

    name: str
    esigma: float = 0.0
    epix: float = 0.0
    epiy: float = 0.0
    edxy: float = 0.0
    edx2y2: float = 0.0
    eds: float = 0.0

    def __post_init__(self) -> None:
        if self.eds < 0.0:
            raise ParameterValueError(
                f"Invalid `eds` value: {self.eds=}. Must be a positive number."
            )

    @property
    def ligand_field_matrix(self) -> np.ndarray:
        """Return the 5x5 ligand-field matrix, constructed from the e-paameters.

        The 5x5 ligand-field matrix represents the electronic perturbation of the metal by the ligand,
        in the local ligand-frame. NOTE: this function does not include the eds parameter.

        Returns:
            ligand_field_matrix: np.ndarray, shape(5, 5)
        """
        return np.array(
            [
                [self.edxy, 0.0, 0.0, 0.0, 0.0],
                [0.0, self.epiy, 0.0, 0.0, 0.0],
                [0.0, 0.0, self.esigma, 0.0, 0.0],
                [0.0, 0.0, 0.0, self.epix, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.edx2y2],
            ]
        )


@dataclass(frozen=True)
class DipoleParameters(BaseTypeChecker):
    name: str
    psigma: float = 0.0
    ppix: float = 0.0
    ppiy: float = 0.0
    fsigma: float = 0.0
    fpix: float = 0.0
    fpiy: float = 0.0


@dataclass(frozen=True)
class MetalParameters:
    d_electrons: int
    racah_b: float = 0.0
    racah_c: float = 0.0
    spin_orbit_coupling: float = 0.0
    kiso: float = 1.0

    def __post_init__(self) -> None:
        if not 1 <= self.d_electrons <= 9:
            raise ParameterValueError(
                f"Invalid `d_electrons` value: {self.d_electrons=}. Must be between 1 and 9 inclusive."
            )


@dataclass
class HyperfineParameters:
    """Stores the hyperfine parameters P (scaling factor) and kappa (Fermi Contact parameter)

    Attributes:
        metal: 3d transition metal
        p: Hyperfine scaling parameter calculated as::
            p_prime (nuclear g value * electronic g value * electron/nuclear bohr magneton)
            multiplied by <r-3>
        kappa: Isotropic Fermi contact constant (variable parameter)
                Expresses the contribution from the s electron density
                at the nucleus to the hyperfine coupling
    """

    metal: str = "M"
    p: float = 0.0
    kappa: float = 0.0
    units: str | None = ""
    s_admixture: float | None = None


@dataclass(frozen=True)
class GeometryParameters(BaseTypeChecker):
    """Store the geometric parameters theta, phi and psi which describe the orientation
    of a ligand with respect to the metal centre"""

    name: str = ""
    theta: float = 0.0
    psi: float = 0.0
    phi: float = 0.0


class ParameterType(str, Enum):
    FIXED = "fixed"
    VARY = "vary"
    LINK = "link"
    BOUND = "bound"


@dataclass(frozen=True)
class Fixed:
    value: float
    kind: Literal[ParameterType.FIXED] = field(
        default=ParameterType.FIXED, init=False, repr=False
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fixed":
        return cls(value=data["value"])

    def __post_init__(self) -> None:
        if not isinstance(self.value, float):
            raise TypeError(f"Expected a float, instead got {type(self.value)=}")


@dataclass(frozen=True)
class Varied:
    value: list[float]
    kind: Literal[ParameterType.VARY] = field(
        default=ParameterType.VARY, init=False, repr=False
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Varied":
        return cls(value=data["value"])

    def __post_init__(self) -> None:
        if not isinstance(self.value, list):
            raise TypeError(f"Expected a list, instead got: {type(self.value)=}")
        for item in self.value:
            if not isinstance(item, float):
                raise TypeError(f"Expected a list[float], instead got: {type(item)=}")


@dataclass(frozen=True)
class Bound:
    lower_bound: float
    upper_bound: float
    starting_value: float
    kind: Literal[ParameterType.BOUND] = field(
        default=ParameterType.BOUND, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, float)
            for value in (self.lower_bound, self.upper_bound, self.starting_value)
        ):
            raise TypeError(
                "`lower_bound`, `upper_bound` and `starting_value` must all be "
                f"floating-point values. Instead got: {type(self.lower_bound)=}, "
                f"{type(self.upper_bound)=}, and {type(self.starting_value)=}"
            )


class LinkType(str, Enum):
    LINEAR = "linear"


@dataclass(frozen=True)
class Link(BaseTypeChecker):
    object_reference: str
    link_type: LinkType
    link_arguments: dict[str, float]
    kind: Literal[ParameterType.LINK] = field(
        default=ParameterType.LINK, init=False, repr=False
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Link":
        return cls(
            object_reference=data["object_reference"],
            link_type=LinkType(data["link_type"]),
            link_arguments=data["link_arguments"],
        )

    @classmethod
    def linear(
        cls, object_reference: str, gradient: float = 1.0, constant: float = 0.0
    ) -> "Link":
        return cls(
            object_reference=object_reference,
            link_type=LinkType.LINEAR,
            link_arguments={
                "gradient": gradient,
                "constant": constant,
            },
        )

    def apply_link(self, value: float) -> float:
        """Apply the link to a value."""
        match self.link_type:
            case LinkType.LINEAR:
                return float(
                    value * self.link_arguments["gradient"]
                    + self.link_arguments["constant"]
                )
            case _:
                raise Exception(
                    f"{self.link_type=} is not accounted for. Please add the functionality."
                )

    def __post_init__(self) -> None:
        if not isinstance(self.object_reference, str):
            raise TypeError(
                f"Expected {str} for `object_reference`, instead got: {type(self.object_reference)=}"
            )
        if not isinstance(self.link_type, (LinkType, str)):
            raise TypeError(
                f"Expected {LinkType} for `link_type`, instead got: {type(self.link_type)=}"
            )
        if not isinstance(self.link_arguments, dict):
            raise TypeError(
                f"Expected {dict} for `link_type`, instead got: {type(self.link_arguments)=}"
            )


@dataclass(frozen=True)
class Assignment(BaseTypeChecker):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Assignment":
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def display(self) -> str:
        rows: list[tuple[str, str]] = []
        for f in fields(self):
            value = getattr(self, f.name)
            display_value = value.display() if hasattr(value, "display") else str(value)
            rows.append((f.name, display_value))
        return f"{self.__class__.__name__}:\n" + tabulate(
            tabular_data=rows, headers=["Field", "Value"]
        )


ParameterVariation: TypeAlias = Fixed | Varied | Link


@dataclass(frozen=True)
class LigandAssignment(Assignment):
    name: str
    esigma: ParameterVariation = field(default=Fixed(value=0.0))
    epix: ParameterVariation = field(default=Fixed(value=0.0))
    epiy: ParameterVariation = field(default=Fixed(value=0.0))
    edxy: ParameterVariation = field(default=Fixed(value=0.0))
    edx2y2: ParameterVariation = field(default=Fixed(value=0.0))
    eds: ParameterVariation = field(default=Fixed(value=0.0))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Assignment":
        attributes = {"name": data.pop("name")}
        for key, assignment in data.items():
            match assignment["kind"]:
                case ParameterType.FIXED:
                    attributes[key] = Fixed.from_dict(assignment)
                case ParameterType.VARY:
                    attributes[key] = Varied.from_dict(assignment)
                case ParameterType.LINK:
                    attributes[key] = Link.from_dict(assignment)
        return cls(**attributes)


@dataclass(frozen=True)
class MetalAssignment(Assignment):
    d_electrons: int
    racah_b: ParameterVariation = field(default=Fixed(value=0.0))
    racah_c: ParameterVariation = field(default=Fixed(value=0.0))
    spin_orbit_coupling: ParameterVariation = field(default=Fixed(value=0.0))
    kiso: ParameterVariation = field(default=Fixed(value=1.0))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Assignment":
        attributes = {"d_electrons": data.pop("d_electrons")}
        for key, assignment in data.items():
            match assignment["kind"]:
                case ParameterType.FIXED:
                    attributes[key] = Fixed.from_dict(assignment)
                case ParameterType.VARY:
                    attributes[key] = Varied.from_dict(assignment)
                case ParameterType.LINK:
                    attributes[key] = Link.from_dict(assignment)
        return cls(**attributes)


@dataclass(frozen=True)
class GeometryAssignment(Assignment):
    name: str
    theta: ParameterVariation = field(default=Fixed(value=0.0))
    psi: ParameterVariation = field(default=Fixed(value=0.0))
    phi: ParameterVariation = field(default=Fixed(value=0.0))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Assignment":
        attributes = {"name": data.pop("name")}
        for key, assignment in data.items():
            match assignment["kind"]:
                case ParameterType.FIXED:
                    attributes[key] = Fixed.from_dict(assignment)
                case ParameterType.VARY:
                    attributes[key] = Varied.from_dict(assignment)
                case ParameterType.LINK:
                    attributes[key] = Link.from_dict(assignment)
        return cls(**attributes)


@dataclass(frozen=True)
class DipoleAssignment(Assignment):
    name: str
    psigma: ParameterVariation = field(default=Fixed(value=0.0))
    ppix: ParameterVariation = field(default=Fixed(value=0.0))
    ppiy: ParameterVariation = field(default=Fixed(value=0.0))
    fsigma: ParameterVariation = field(default=Fixed(value=0.0))
    fpix: ParameterVariation = field(default=Fixed(value=0.0))
    fpiy: ParameterVariation = field(default=Fixed(value=0.0))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DipoleAssignment":
        attributes = {"name": data.pop("name")}
        for key, assignment in data.items():
            match assignment["kind"]:
                case ParameterType.FIXED:
                    attributes[key] = Fixed.from_dict(assignment)
                case ParameterType.VARY:
                    attributes[key] = Varied.from_dict(assignment)
                case ParameterType.LINK:
                    attributes[key] = Link.from_dict(assignment)
        return cls(**attributes)


ParameterOptimisation: TypeAlias = Fixed | Bound | Link


@dataclass(frozen=True)
class HyperfineAssignment(Assignment):
    metal: str | None = None
    units: str = ""
    p: ParameterVariation = field(default=Fixed(value=0.0))
    kappa: ParameterVariation = field(default=Fixed(value=0.0))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Assignment":
        attributes = {"metal": data.pop("metal"), "units": data.pop("units")}
        for key, assignment in data.items():
            match assignment["kind"]:
                case ParameterType.FIXED:
                    attributes[key] = Fixed.from_dict(assignment)
                case ParameterType.VARY:
                    attributes[key] = Varied.from_dict(assignment)
                case ParameterType.LINK:
                    attributes[key] = Link.from_dict(assignment)
        return cls(**attributes)


@dataclass(frozen=True)
class LigandOptimisation(Assignment):
    name: str
    esigma: ParameterOptimisation = field(default=Fixed(value=0.0))
    epix: ParameterOptimisation = field(default=Fixed(value=0.0))
    epiy: ParameterOptimisation = field(default=Fixed(value=0.0))
    edxy: ParameterOptimisation = field(default=Fixed(value=0.0))
    edx2y2: ParameterOptimisation = field(default=Fixed(value=0.0))
    eds: ParameterOptimisation = field(default=Fixed(value=0.0))


@dataclass(frozen=True)
class MetalOptimisation(Assignment):
    d_electrons: int
    racah_b: ParameterOptimisation = field(default=Fixed(value=0.0))
    racah_c: ParameterOptimisation = field(default=Fixed(value=0.0))
    spin_orbit_coupling: ParameterOptimisation = field(default=Fixed(value=0.0))
    kiso: ParameterOptimisation = field(default=Fixed(value=1.0))
