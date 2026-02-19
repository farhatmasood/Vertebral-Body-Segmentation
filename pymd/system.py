"""
System Module

This module defines the System class which represents the collection of particles
in the simulation. It handles the state (positions, velocities, forces) and
boundary conditions.
"""

import numpy as np
from typing import Optional, Tuple
from .potentials import lennard_jones

class System:
    """
    Represents a system of particles in a periodic box.

    Attributes:
        n_particles (int): Number of particles.
        box_length (float): Length of the simulation box (assumed cubic).
        positions (np.ndarray): Positions of particles (N, 3).
        velocities (np.ndarray): Velocities of particles (N, 3).
        forces (np.ndarray): Forces acting on particles (N, 3).
        mass (float): Mass of each particle.
        epsilon (float): LJ potential depth.
        sigma (float): LJ potential zero-crossing distance.
        cutoff (float): Cutoff radius for potential calculation.
    """

    def __init__(
        self,
        n_particles: int,
        box_length: float,
        mass: float = 1.0,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        cutoff: float = 2.5
    ):
        """
        Initialize the System.

        Args:
            n_particles (int): Number of particles.
            box_length (float): Length of the cubic box.
            mass (float): Mass of particles.
            epsilon (float): LJ parameter epsilon.
            sigma (float): LJ parameter sigma.
            cutoff (float): Cutoff distance (in units of sigma).
        """
        self.n_particles = n_particles
        self.box_length = box_length
        self.mass = mass
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff * sigma  # Actual cutoff distance

        # Calculate potential shift at cutoff to ensure continuity of V(r)
        # V_shift = V_LJ(r_cut)
        # We subtract this from the potential energy of each pair
        # V(r_cut) = 4 * eps * ((sigma/rc)^12 - (sigma/rc)^6)
        sr_cut = self.sigma / self.cutoff
        sr_cut6 = sr_cut**6
        sr_cut12 = sr_cut6 * sr_cut6
        self.potential_shift = 4.0 * self.epsilon * (sr_cut12 - sr_cut6)

        # Initialize arrays
        self.positions = np.zeros((n_particles, 3))
        self.velocities = np.zeros((n_particles, 3))
        self.forces = np.zeros((n_particles, 3))

    def initialize_positions_lattice(self, lattice_constant: Optional[float] = None) -> None:
        """
        Initialize particle positions on a simple cubic lattice.
        If lattice_constant is not provided, it is estimated from density.
        """
        # Estimate lattice constant based on density
        if lattice_constant is None:
            lattice_constant = self.box_length / (self.n_particles**(1/3))

        n_side = int(np.ceil(self.n_particles**(1/3)))
        spacing = self.box_length / n_side

        count = 0
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    if count < self.n_particles:
                        self.positions[count] = np.array([i, j, k]) * spacing + spacing/2
                        count += 1

    def initialize_velocities(self, temperature: float) -> None:
        """
        Initialize velocities according to the Maxwell-Boltzmann distribution.

        Args:
            temperature (float): Target temperature (k_B = 1).
        """
        # Standard deviation for each component: sqrt(k_B * T / m)
        std_dev = np.sqrt(temperature / self.mass)
        self.velocities = np.random.normal(0, std_dev, (self.n_particles, 3))

        # Remove center of mass motion
        v_cm = np.mean(self.velocities, axis=0)
        self.velocities -= v_cm

    def apply_pbc(self) -> None:
        """
        Apply periodic boundary conditions to positions.
        """
        self.positions = self.positions % self.box_length

    def calculate_forces(self) -> float:
        """
        Calculate forces on all particles and the potential energy of the system.

        Returns:
            float: Total potential energy.
        """
        self.forces[:] = 0.0
        potential_energy = 0.0

        # Pairwise differences with PBC (Minimum Image Convention)
        # diff[i, j] = pos[i] - pos[j]
        # Memory intensive for large N, but simple and correct for N < 1000

        # Broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]

        # Apply Minimum Image Convention
        diff = diff - np.round(diff / self.box_length) * self.box_length

        # Squared distances: (N, N)
        r2 = np.sum(diff**2, axis=-1)

        # Avoid self-interaction (r=0) and double counting
        # We can use a mask for strictly upper triangular (j > i)
        mask = np.triu(np.ones((self.n_particles, self.n_particles), dtype=bool), k=1)

        valid_r2 = r2[mask]
        valid_diff = diff[mask] # (M, 3)

        # Apply cutoff
        cutoff2 = self.cutoff**2
        in_cutoff = valid_r2 < cutoff2

        # Filter interactions within cutoff
        interactions_r2 = valid_r2[in_cutoff]
        interactions_diff = valid_diff[in_cutoff]

        # Calculate forces and energy
        pot_ij, force_factor = lennard_jones(interactions_r2, self.epsilon, self.sigma)

        # Apply potential shift
        pot_ij -= self.potential_shift

        # Accumulate potential energy
        potential_energy = np.sum(pot_ij)

        # Forces
        # Force on i due to j is F_ij = factor * r_ij
        # Force on j due to i is -F_ij
        # interactions_diff corresponds to r_ij = pos[i] - pos[j]

        force_vec = interactions_diff * force_factor[:, np.newaxis] # (M_filtered, 3)

        # We need to map these back to the particles
        # Get indices of interacting pairs
        i_indices, j_indices = np.where(mask)
        # Filter by cutoff
        i_indices = i_indices[in_cutoff]
        j_indices = j_indices[in_cutoff]

        # Accumulate forces
        # np.add.at is unbuffered in-place add
        np.add.at(self.forces, i_indices, force_vec)
        np.add.at(self.forces, j_indices, -force_vec)

        return potential_energy

    def kinetic_energy(self) -> float:
        """Calculate total kinetic energy."""
        v2 = np.sum(self.velocities**2, axis=1)
        return 0.5 * self.mass * np.sum(v2)

    def temperature(self) -> float:
        """Calculate instantaneous temperature."""
        ke = self.kinetic_energy()
        # Equipartition theorem: KE = 3/2 N k_B T -> T = 2 KE / (3 N k_B)
        # Here k_B = 1
        return 2.0 * ke / (3.0 * self.n_particles)
