"""
Analysis Module

This module provides tools for analyzing the results of molecular dynamics simulations.
It includes functions for calculating the Radial Distribution Function (RDF)
and thermodynamic properties.
"""

import numpy as np
from typing import Tuple, List

def compute_rdf(
    positions: np.ndarray,
    box_length: float,
    r_max: float,
    dr: float,
    n_particles: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Radial Distribution Function (RDF) g(r).

    g(r) = (count in shell) / (ideal gas count in shell)
    ideal gas count = rho * 4 * pi * r^2 * dr

    Args:
        positions (np.ndarray): Particle positions (N, 3).
        box_length (float): Length of the simulation box.
        r_max (float): Maximum radius for RDF calculation (usually box_length/2).
        dr (float): Bin width.
        n_particles (int): Number of particles.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - bins (np.ndarray): The center positions of the bins.
            - gr (np.ndarray): The value of g(r) for each bin.
    """
    # Create histogram bins
    edges = np.arange(0, r_max + dr, dr)
    bins = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(len(bins))

    # Calculate pairwise distances
    # Using the same vectorized logic as in System (careful with memory for large N)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    diff = diff - np.round(diff / box_length) * box_length
    r2 = np.sum(diff**2, axis=-1)

    # Exclude self-interaction (diagonal is 0)
    # Get upper triangle
    mask = np.triu(np.ones((n_particles, n_particles), dtype=bool), k=1)
    dists = np.sqrt(r2[mask])

    # Histogram
    counts, _ = np.histogram(dists, bins=edges)

    # Normalization
    # Volume of shell = 4/3 * pi * ((r+dr)^3 - r^3)
    # approximation: 4 * pi * r^2 * dr

    rho = n_particles / (box_length**3)
    shell_volumes = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)

    # Number of particles expected in shell for ideal gas
    # We count PAIRS, so each pair (i,j) contributes 2 to the total counts if we iterate all pairs.
    # But here we only count unique pairs (i < j).
    # Total pairs = N*(N-1)/2.
    # Density of pairs?
    # Actually, g(r) is defined relative to a reference particle.
    # N(r) = sum over i (sum over j!=i delta(r - r_ij))
    # Our counts are from unique pairs. The total counts if we iterated all would be 2 * counts.
    # Average number of neighbors at r per particle is (2 * counts) / N.
    # Density is N / V.
    # g(r) = ( (2 * counts) / N ) / ( Volume_shell * rho )

    gr = (2.0 * counts / n_particles) / (shell_volumes * rho)

    return bins, gr
