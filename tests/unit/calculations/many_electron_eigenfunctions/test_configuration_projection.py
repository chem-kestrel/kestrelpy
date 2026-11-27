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

import numpy as np
import pytest

from kestrelpy.calculations.many_electron_eigenfunctions.configuration_projection import (
    Configuration,
    ConfigurationProjection,
    _get_spin_eigenfunction_indices,
    get_eigenfunction_configuration_projection,
    get_level_configuration_projection,
)


def test_configuration_str_representation() -> None:
    config = Configuration(
        occupation=(0, 1, 2), contribution=0.9534232132, spin_multiplicity=4
    )
    expected_string_representation = "4: 11100 (0.953)"
    assert config.get_string_representation() == expected_string_representation


@pytest.mark.parametrize(
    ("spin_multiplicity", "expected_output"),
    [
        (1, np.array([0, 1, 4])),
        (3, np.array([2])),
        (5, np.array([3])),
    ],
)
def test_get_spin_eigenfunction_indices(
    spin_multiplicity: int, expected_output: np.ndarray
) -> None:
    spin_multiplicities = [1, 3, 5]
    spin_projections = np.vstack(
        (
            np.array([1, 1, 0, 0, 1]),  # Spin singlets at indices 0, 1, and 4
            np.array([0, 0, 1, 0, 0]),  # Spin triplets at index 2
            np.array([0, 0, 0, 1, 0]),  # Spin quintets at index 3
        )
    )

    # Get indices for spin singlets
    result = _get_spin_eigenfunction_indices(
        spin_multiplicities=spin_multiplicities,
        spin_projections=spin_projections,
        spin_multiplicity=spin_multiplicity,
    )
    assert np.array_equal(result, expected_output)


@pytest.mark.parametrize(
    ("index", "expected_configurations"),
    [
        pytest.param(
            0,
            [
                Configuration(
                    occupation=(1, 0, 0), contribution=0.75, spin_multiplicity=3
                ),
                Configuration(
                    occupation=(0, 0, 1), contribution=0.15, spin_multiplicity=3
                ),
                Configuration(
                    occupation=(0, 1, 0), contribution=0.10, spin_multiplicity=3
                ),
            ],
        ),
        pytest.param(
            1,
            [
                Configuration(
                    occupation=(0, 0, 1), contribution=0.85, spin_multiplicity=3
                ),
                Configuration(
                    occupation=(1, 0, 0), contribution=0.15, spin_multiplicity=3
                ),
                # Third entry missing because the contribution < 0.0005
            ],
        ),
        pytest.param(
            2,
            [
                Configuration(
                    occupation=(0, 1, 0), contribution=0.90, spin_multiplicity=3
                ),
                Configuration(
                    occupation=(1, 0, 0), contribution=0.10, spin_multiplicity=3
                ),
                # Third entry missing because the contribution < 0.0005
            ],
        ),
    ],
)
def test_get_eigenfunction_configuration_projection(
    index: int, expected_configurations: list[Configuration]
) -> None:
    configurations = [
        Configuration(occupation=(1, 0, 0), contribution=0.0, spin_multiplicity=3),
        Configuration(occupation=(0, 1, 0), contribution=0.0, spin_multiplicity=3),
        Configuration(occupation=(0, 0, 1), contribution=0.0, spin_multiplicity=3),
    ]
    dummy_mapping = np.vstack(
        (
            np.array([0.75, 0.15, 0.10]),
            np.array([0.10, 0.00, 0.90]),
            np.array([0.15, 0.85, 0.00]),
        )
    )

    configuration_projection = ConfigurationProjection(
        configurations=configurations, mapping=dummy_mapping
    )

    # for a doublet level, with eigenfunctions 0 and 1.
    # final = 2 as final = start + degeneracy
    configurations = get_eigenfunction_configuration_projection(
        index=index, configuration_projection=configuration_projection
    )
    assert configurations == expected_configurations


def test_get_level_configuration_projection() -> None:
    configurations = [
        Configuration(occupation=(1, 0, 0), contribution=0.0, spin_multiplicity=3),
        Configuration(occupation=(0, 1, 0), contribution=0.0, spin_multiplicity=3),
        Configuration(occupation=(0, 0, 1), contribution=0.0, spin_multiplicity=3),
    ]
    dummy_mapping = np.vstack(
        (
            np.array([0.75, 0.15, 0.10]),
            np.array([0.10, 0.00, 0.90]),
            np.array([0.15, 0.85, 0.00]),
        )
    )

    configuration_projection = ConfigurationProjection(
        configurations=configurations, mapping=dummy_mapping
    )

    # for a doublet level, with eigenfunctions 0 and 1.
    # final = 2 as final = start + degeneracy
    configurations = get_level_configuration_projection(
        start=0, final=2, configuration_projection=configuration_projection
    )
    normalise = (
        2  # Contributions need to be sum of rows for the columns divided by degeneracy.
    )
    expected_configurations = [
        Configuration(
            occupation=(0, 0, 1), contribution=1.0 / normalise, spin_multiplicity=3
        ),
        Configuration(
            occupation=(1, 0, 0), contribution=0.9 / normalise, spin_multiplicity=3
        ),
        Configuration(
            occupation=(0, 1, 0), contribution=0.1 / normalise, spin_multiplicity=3
        ),
    ]
    assert configurations == expected_configurations

    # For a single level, with eigenfunctions 2.
    configurations = get_level_configuration_projection(
        start=2, final=3, configuration_projection=configuration_projection
    )
    expected_configurations = [
        Configuration(occupation=(0, 1, 0), contribution=0.9, spin_multiplicity=3),
        Configuration(occupation=(1, 0, 0), contribution=0.1, spin_multiplicity=3),
        # Missing third configuration as its contribution is 0.0
    ]
    assert configurations == expected_configurations
