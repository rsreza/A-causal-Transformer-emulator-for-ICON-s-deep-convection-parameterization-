"""Sanity tests for regime definitions."""
import numpy as np

from src.regimes import (
    Regime, REGIME_NAMES, REGIME_PROFILES,
    regime_mass_flux_shape, regime_memory_length,
)


def test_all_regimes_have_profiles():
    for r in Regime:
        assert r in REGIME_PROFILES
        assert r in REGIME_NAMES


def test_suppressed_is_zero_flux():
    shape = regime_mass_flux_shape(Regime.SUPPRESSED)
    assert np.allclose(shape, 0.0)


def test_shallow_peaks_lower_than_deep():
    s = regime_mass_flux_shape(Regime.SHALLOW)
    d = regime_mass_flux_shape(Regime.DEEP)
    assert np.argmax(d) > np.argmax(s)


def test_memory_lengths_ordered():
    # Physical expectation: organized >= deep >= shallow >= suppressed
    assert regime_memory_length(Regime.ORGANIZED) >= regime_memory_length(Regime.DEEP)
    assert regime_memory_length(Regime.DEEP) >= regime_memory_length(Regime.SHALLOW)
    assert regime_memory_length(Regime.SHALLOW) >= regime_memory_length(Regime.SUPPRESSED)
