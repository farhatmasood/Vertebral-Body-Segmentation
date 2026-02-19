"""
Example script: Simulate Liquid Argon.

This script demonstrates how to set up and run a simulation of Liquid Argon
using the Lennard-Jones potential in reduced units.

Parameters:
    N = 108 (Particles)
    rho = 0.844 (Reduced density)
    T = 0.728 (Reduced temperature)
    dt = 0.005 (Reduced time step)
"""

import numpy as np
import time
from pymd.system import System
from pymd.integrator import VelocityVerlet
from pymd.simulation import Simulation

def main():
    # Parameters
    n_particles = 108
    density = 0.844
    temperature = 0.728
    dt = 0.005
    n_steps = 500

    # Calculate box length
    box_length = (n_particles / density) ** (1/3)
    print(f"Simulation of Liquid Argon (N={n_particles}, rho={density}, T={temperature})")
    print(f"Box Length: {box_length:.3f} sigma")

    # Initialize System
    system = System(n_particles=n_particles, box_length=box_length)

    # Place particles on a lattice
    print("Initializing positions...")
    system.initialize_positions_lattice()

    # Set initial velocities
    print("Initializing velocities...")
    system.initialize_velocities(temperature=temperature)

    # Initialize Integrator
    integrator = VelocityVerlet(dt=dt)

    # Initialize Simulation
    sim = Simulation(system, integrator)

    # Run
    print(f"Starting simulation for {n_steps} steps...")
    start_time = time.time()
    sim.run(n_steps=n_steps, output_interval=50)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    # Save results
    sim.save_results("argon_simulation")

    # Basic Analysis of Final State
    # Verify energy conservation
    initial_energy = sim.energies[0]["total"]
    final_energy = sim.energies[-1]["total"]
    drift = abs(final_energy - initial_energy) / abs(initial_energy)
    print(f"Energy Drift: {drift:.2%}")

    if drift < 0.01:
        print("Energy conservation: GOOD")
    else:
        print("Energy conservation: POOR")

if __name__ == "__main__":
    main()
