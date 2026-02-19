"""
Command Line Interface

This module provides the entry point for running simulations from the command line.
"""

import argparse
import sys
from pymd.system import System
from pymd.integrator import VelocityVerlet
from pymd.simulation import Simulation

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a molecular dynamics simulation.")

    parser.add_argument(
        "-n", "--n-particles", type=int, default=100, help="Number of particles (default: 100)"
    )
    parser.add_argument(
        "-b", "--box-length", type=float, default=10.0, help="Length of the simulation box (default: 10.0)"
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=1.0, help="Target temperature (reduced units, default: 1.0)"
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=1000, help="Number of simulation steps (default: 1000)"
    )
    parser.add_argument(
        "-d", "--dt", type=float, default=0.005, help="Time step (default: 0.005)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="simulation_output", help="Prefix for output files (default: simulation_output)"
    )
    parser.add_argument(
        "--density", type=float, default=None, help="Target density (overrides box length if specified)"
    )

    args = parser.parse_args()

    # Calculate box length from density if provided
    if args.density is not None:
        box_length = (args.n_particles / args.density) ** (1/3)
        print(f"Calculated box length from density {args.density}: {box_length:.3f}")
    else:
        box_length = args.box_length

    print(f"Initializing system with {args.n_particles} particles in box L={box_length:.3f}...")

    # Initialize System
    system = System(n_particles=args.n_particles, box_length=box_length)

    # Initialize positions on a lattice to avoid overlaps
    system.initialize_positions_lattice()

    # Initialize velocities
    system.initialize_velocities(temperature=args.temperature)

    # Initialize Integrator
    integrator = VelocityVerlet(dt=args.dt)

    # Initialize Simulation
    sim = Simulation(system, integrator)

    print(f"Starting simulation for {args.steps} steps (dt={args.dt})...")

    # Run
    sim.run(n_steps=args.steps, output_interval=max(1, args.steps // 100))

    # Save results
    sim.save_results(args.output)

    print("Simulation complete.")

if __name__ == "__main__":
    main()
