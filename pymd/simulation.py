"""
Simulation Module

This module orchestrates the molecular dynamics simulation.
It manages the System, Integrator, and data output.
"""

import numpy as np
import time
import csv
from typing import List, Optional, Dict
from .system import System
from .integrator import VelocityVerlet, Integrator

class Simulation:
    """
    Manages the MD simulation run.

    Attributes:
        system (System): The particle system.
        integrator (Integrator): The integration scheme.
        trajectory (List[np.ndarray]): Stored trajectory (positions).
        energies (List[Dict[str, float]]): Stored energy data.
    """

    def __init__(self, system: System, integrator: Integrator):
        self.system = system
        self.integrator = integrator
        self.trajectory: List[np.ndarray] = []
        self.energies: List[Dict[str, float]] = []

    def run(self, n_steps: int, output_interval: int = 100, verbose: bool = True) -> None:
        """
        Run the simulation.

        Args:
            n_steps (int): Total number of time steps.
            output_interval (int): Steps between outputs.
            verbose (bool): Whether to print progress.
        """
        start_time = time.time()

        # Initial force calculation (t=0)
        # Note: calculate_forces returns PE at current configuration
        pot_energy = self.system.calculate_forces()

        # Save initial state (t=0)
        self._record_state(0, pot_energy, verbose=False)

        for step in range(1, n_steps + 1):
            # Step forward to t = step * dt
            # The integrator updates system state to t+dt and returns PE at t+dt
            pot_energy = self.integrator.step(self.system)

            if step % output_interval == 0:
                self._record_state(step, pot_energy, verbose=verbose, start_time=start_time, total_steps=n_steps)

    def _record_state(
        self,
        step: int,
        pot_energy: float,
        verbose: bool = False,
        start_time: float = 0.0,
        total_steps: int = 0
    ) -> None:
        """Helper to record system state."""
        kin_energy = self.system.kinetic_energy()
        total_energy = kin_energy + pot_energy
        temperature = self.system.temperature()

        self.energies.append({
            "step": step,
            "kinetic": kin_energy,
            "potential": pot_energy,
            "total": total_energy,
            "temperature": temperature
        })

        self.trajectory.append(self.system.positions.copy())

        if verbose:
            elapsed = time.time() - start_time
            print(f"Step {step}/{total_steps} | T: {temperature:.3f} | E_tot: {total_energy:.4e} | Elapsed: {elapsed:.2f}s")

    def save_results(self, filename_prefix: str) -> None:
        """Save simulation data to files."""
        # Save energies
        with open(f"{filename_prefix}_energy.csv", "w", newline="") as f:
            fieldnames = ["step", "kinetic", "potential", "total", "temperature"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.energies)

        # Save trajectory
        np.save(f"{filename_prefix}_trajectory.npy", np.array(self.trajectory))
        print(f"Results saved to {filename_prefix}_energy.csv and {filename_prefix}_trajectory.npy")
