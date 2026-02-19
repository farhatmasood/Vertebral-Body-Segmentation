"""
Tests for integrator module.
"""

import numpy as np
import pytest
from pymd.system import System
from pymd.integrator import VelocityVerlet

def test_integration_step():
    """Test one integration step."""
    sys = System(n_particles=2, box_length=10.0, mass=1.0)
    sys.positions[0] = [4.0, 5.0, 5.0]
    sys.positions[1] = [6.0, 5.0, 5.0] # r=2.0

    sys.velocities[0] = [0.1, 0.0, 0.0]
    sys.velocities[1] = [-0.1, 0.0, 0.0]

    integrator = VelocityVerlet(dt=0.01)

    # Calculate initial energy
    initial_pe = sys.calculate_forces()
    initial_ke = sys.kinetic_energy()
    initial_total = initial_ke + initial_pe

    # Run one step
    new_pe = integrator.step(sys)
    new_ke = sys.kinetic_energy()
    new_total = new_ke + new_pe

    # Check positions updated
    # r(t+dt) = r(t) + v(t+dt/2)*dt
    assert not np.allclose(sys.positions[0], [4.0, 5.0, 5.0])

    # Check energy conservation
    assert np.isclose(initial_total, new_total, atol=1e-3)

def test_energy_conservation_long_run():
    """Test energy conservation over multiple steps."""
    sys = System(n_particles=2, box_length=10.0, mass=1.0)
    sys.positions[0] = [4.0, 5.0, 5.0]
    sys.positions[1] = [5.5, 5.0, 5.0] # r=1.5

    sys.velocities[0] = [0.1, 0.0, 0.0]
    sys.velocities[1] = [-0.1, 0.0, 0.0]

    integrator = VelocityVerlet(dt=0.005)

    # Initial setup
    initial_pe = sys.calculate_forces()
    initial_total = sys.kinetic_energy() + initial_pe

    pe = initial_pe
    for _ in range(100):
        pe = integrator.step(sys)

    final_total = sys.kinetic_energy() + pe

    # Should conserve energy within tolerance
    assert np.isclose(initial_total, final_total, rtol=1e-2)
