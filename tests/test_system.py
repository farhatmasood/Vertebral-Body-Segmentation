"""
Tests for System module.
"""

import numpy as np
import pytest
from pymd.system import System

def test_pbc():
    """Test Periodic Boundary Conditions."""
    sys = System(n_particles=1, box_length=10.0)
    sys.positions[0] = [12.0, -2.0, 5.0]

    sys.apply_pbc()

    # 12 -> 2
    # -2 -> 8
    # 5 -> 5
    assert np.allclose(sys.positions[0], [2.0, 8.0, 5.0])

def test_kinetic_energy():
    """Test kinetic energy calculation."""
    sys = System(n_particles=2, box_length=10.0, mass=2.0)
    sys.velocities[0] = [1.0, 0.0, 0.0]
    sys.velocities[1] = [0.0, 1.0, 0.0]

    # KE = 0.5 * m * v1^2 + 0.5 * m * v2^2
    # KE = 0.5 * 2.0 * 1.0 + 0.5 * 2.0 * 1.0 = 1.0 + 1.0 = 2.0

    ke = sys.kinetic_energy()
    assert np.isclose(ke, 2.0)

def test_temperature():
    """Test temperature calculation."""
    sys = System(n_particles=2, box_length=10.0, mass=1.0)
    sys.velocities[0] = [1.0, 0.0, 0.0]
    sys.velocities[1] = [-1.0, 0.0, 0.0]

    # KE = 0.5 * 1.0 * 1.0 + 0.5 * 1.0 * 1.0 = 1.0
    # T = 2 * KE / (3 * N) = 2 * 1.0 / (3 * 2) = 1/3

    temp = sys.temperature()
    assert np.isclose(temp, 1.0/3.0)

def test_force_calculation_two_particles():
    """Test force between two particles at r=sigma."""
    sys = System(n_particles=2, box_length=10.0, epsilon=1.0, sigma=1.0)
    sys.positions[0] = [1.0, 1.0, 1.0]
    sys.positions[1] = [2.0, 1.0, 1.0] # r = 1.0 (sigma) along x

    pe = sys.calculate_forces()

    # PE at r=sigma is 0 for unshifted LJ.
    # But System applies shift V_shift = V(cutoff).
    # Actual PE = V(r) - V(cutoff) = 0.0 - V(cutoff).

    rc = 2.5 * 1.0
    v_cut = 4.0 * 1.0 * ((1.0/rc)**12 - (1.0/rc)**6)
    expected_pe = 0.0 - v_cut

    assert np.isclose(pe, expected_pe)

    # Force on 0 from 1: r_01 = p0 - p1 = [-1, 0, 0]
    # Factor = 24 * epsilon / sigma^2 = 24
    # F_0 = Factor * r_01 = [-24, 0, 0]
    # Force on 1 from 0: r_10 = p1 - p0 = [1, 0, 0]
    # F_1 = Factor * r_10 = [24, 0, 0] (Wait, check sign)

    # F/r = 24 * epsilon * (2 - 1) / r^2 = 24 / 1 = 24.
    # Force on 0 = F/r * (pos[0] - pos[1]) = 24 * [-1, 0, 0] = [-24, 0, 0].

    assert np.allclose(sys.forces[0], [-24.0, 0.0, 0.0])
    assert np.allclose(sys.forces[1], [24.0, 0.0, 0.0])
