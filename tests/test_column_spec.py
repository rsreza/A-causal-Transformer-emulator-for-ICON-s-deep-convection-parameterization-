"""Sanity tests for the shared column specification."""
import numpy as np

from src.column_spec import (
    N_LEVELS, N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
    INPUT_VARS, OUTPUT_VARS, PRECIP_VARS,
    Z_INTERFACES, Z_CENTERS,
)


def test_shapes():
    assert len(INPUT_VARS) == N_INPUT_VARS == 10
    assert len(OUTPUT_VARS) == N_OUTPUT_VARS == 6
    assert len(PRECIP_VARS) == N_PRECIP_VARS == 2
    assert len(Z_INTERFACES) == N_LEVELS + 1
    assert len(Z_CENTERS) == N_LEVELS


def test_vertical_grid_monotonic():
    assert np.all(np.diff(Z_INTERFACES) > 0)
    assert Z_INTERFACES[0] == 0.0
    assert Z_CENTERS[0] > 0.0


def test_variable_order_stable():
    # Guard against accidental reordering — the synthetic generator and the
    # real-data preprocessor both rely on this exact order.
    assert INPUT_VARS[0] == "T"
    assert INPUT_VARS[1] == "q"
    assert OUTPUT_VARS[0] == "dT_dt"
    assert OUTPUT_VARS[1] == "dq_dt"
