"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
from pymd.system import System

@pytest.fixture
def empty_system():
    """Create a system with 2 particles in a large box."""
    return System(n_particles=2, box_length=10.0, mass=1.0, epsilon=1.0, sigma=1.0, cutoff=2.5)

@pytest.fixture
def two_particle_system():
    """Create a system with 2 particles at specific positions."""
    sys = System(n_particles=2, box_length=10.0)
    sys.positions[0] = [0.0, 0.0, 0.0]
    sys.positions[1] = [1.0, 0.0, 0.0] # r = 1.0 (sigma)
    return sys
