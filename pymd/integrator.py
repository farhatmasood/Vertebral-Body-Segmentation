"""
Integrator Module

This module implements integration schemes for molecular dynamics.
The Velocity Verlet algorithm is the standard choice for its symplectic properties
(energy conservation).
"""

import numpy as np
from typing import Protocol
from .system import System

class Integrator(Protocol):
    def step(self, system: System) -> float:
        """Perform one integration step and return potential energy."""
        ...

class VelocityVerlet:
    """
    Velocity Verlet integrator.

    Equations of motion:
    v(t + dt/2) = v(t) + a(t) * dt/2
    r(t + dt) = r(t) + v(t + dt/2) * dt
    a(t + dt) = F(r(t + dt)) / m
    v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
    """

    def __init__(self, dt: float):
        self.dt = dt

    def step(self, system: System) -> float:
        """
        Perform one step of Velocity Verlet integration.

        Args:
            system (System): The particle system to update.

        Returns:
            float: Potential energy at the new step.
        """
        dt = self.dt
        dt2 = 0.5 * dt
        mass = system.mass

        # 1. Update velocities by half step
        # acceleration = forces / mass
        system.velocities += (system.forces / mass) * dt2

        # 2. Update positions
        system.positions += system.velocities * dt

        # Apply PBC
        system.positions = system.positions % system.box_length

        # 3. Calculate new forces
        potential_energy = system.calculate_forces()

        # 4. Update velocities by another half step with new forces
        system.velocities += (system.forces / mass) * dt2

        return potential_energy
