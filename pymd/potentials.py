"""
Potentials Module

This module defines the interaction potentials between particles.
The primary potential implemented is the Lennard-Jones 12-6 potential.
"""

import numpy as np
from typing import Tuple

def lennard_jones(
    r2: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Lennard-Jones potential and force magnitude.

    The Lennard-Jones potential is defined as:
        V(r) = 4 * epsilon * ((sigma/r)^12 - (sigma/r)^6)

    The force magnitude is given by:
        F(r) = -dV/dr = 24 * epsilon * (2*(sigma/r)^12 - (sigma/r)^6) / r

    However, for vector calculation, we often compute F/r to multiply by the vector r_ij later.
    Let ff = F/r.
    ff = 24 * epsilon * (2*(sigma/r)^12 - (sigma/r)^6) / r^2
       = 24 * epsilon * (2*(sigma^12/r^14) - (sigma^6/r^8))
       = 24 * epsilon * (2 * (sigma^6/r^6)^2 - (sigma^6/r^6)) / r^2

    Args:
        r2 (np.ndarray): Squared distance between particles (r^2).
        epsilon (float): Depth of the potential well.
        sigma (float): Finite distance at which the inter-particle potential is zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - potential_energy (np.ndarray): The calculated potential energy.
            - force_factor (np.ndarray): The factor F(r)/r to be multiplied by the displacement vector.
    """
    # Avoid division by zero for r2=0 (self-interaction), usually handled by caller or masking
    # But here we assume r2 > 0.

    # Calculate (sigma^2 / r^2)^3 = (sigma/r)^6
    inv_r2 = 1.0 / r2
    sr2 = sigma * sigma * inv_r2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6

    potential = 4.0 * epsilon * (sr12 - sr6)

    # F/r = (24 * epsilon / r^2) * (2 * (sigma/r)^12 - (sigma/r)^6)
    #     = 24 * epsilon * inv_r2 * (2 * sr12 - sr6)
    force_factor = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6)

    return potential, force_factor
