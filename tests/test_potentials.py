"""
Tests for potentials module.
"""

import numpy as np
import pytest
from pymd.potentials import lennard_jones

def test_lj_at_sigma():
    """Test LJ potential at r = sigma."""
    sigma = 1.0
    epsilon = 1.0
    r = sigma
    r2 = np.array([r**2])

    pot, factor = lennard_jones(r2, epsilon, sigma)

    # V(sigma) = 0
    assert np.isclose(pot[0], 0.0)

    # F(sigma) = 24 * epsilon / r * (2 - 1) = 24 * epsilon / sigma
    # factor = F/r = 24 * epsilon / sigma^2
    expected_factor = 24.0 * epsilon / (sigma**2)
    assert np.isclose(factor[0], expected_factor)

def test_lj_at_minimum():
    """Test LJ potential at minimum r = 2^(1/6) sigma."""
    sigma = 1.0
    epsilon = 1.0
    r = 2**(1/6) * sigma
    r2 = np.array([r**2])

    pot, factor = lennard_jones(r2, epsilon, sigma)

    # V(r_min) = -epsilon
    assert np.isclose(pot[0], -epsilon)

    # F(r_min) = 0
    # factor = F/r = 0
    assert np.isclose(factor[0], 0.0, atol=1e-10)

def test_lj_large_distance():
    """Test LJ potential at large distance."""
    sigma = 1.0
    epsilon = 1.0
    r = 100.0 * sigma
    r2 = np.array([r**2])

    pot, factor = lennard_jones(r2, epsilon, sigma)

    # V -> 0
    assert np.isclose(pot[0], 0.0, atol=1e-10)

    # F -> 0
    assert np.isclose(factor[0], 0.0, atol=1e-10)
