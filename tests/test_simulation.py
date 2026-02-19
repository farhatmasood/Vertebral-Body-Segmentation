"""
Tests for simulation module.
"""

import numpy as np
import pytest
from pymd.system import System
from pymd.integrator import VelocityVerlet
from pymd.simulation import Simulation

def test_simulation_run(tmp_path):
    """Test full simulation run and output."""
    sys = System(n_particles=2, box_length=10.0)
    sys.positions[0] = [4.0, 5.0, 5.0]
    sys.positions[1] = [6.0, 5.0, 5.0]
    sys.velocities[0] = [0.1, 0.0, 0.0]
    sys.velocities[1] = [-0.1, 0.0, 0.0]

    integrator = VelocityVerlet(dt=0.01)

    sim = Simulation(sys, integrator)

    # Run simulation
    sim.run(n_steps=10, output_interval=5, verbose=False)

    # Check if results are stored
    assert len(sim.energies) == 3 # 0, 5, 10
    assert len(sim.trajectory) == 3

    # Save results to tmp path
    output_prefix = tmp_path / "test_sim"
    sim.save_results(str(output_prefix))

    # Check if files created
    assert (tmp_path / "test_sim_energy.csv").exists()
    assert (tmp_path / "test_sim_trajectory.npy").exists()
