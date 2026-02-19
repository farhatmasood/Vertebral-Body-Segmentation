# pymd: Python Molecular Dynamics

A high-fidelity, deterministic molecular dynamics simulation package for academic research.

## Overview
`pymd` is a pure-Python library (with NumPy acceleration) for simulating particle systems using the Lennard-Jones potential. It implements the Velocity Verlet integration scheme, providing energy conservation and accurate thermodynamic analysis.

## Features
- **Deterministic Simulation**: Strictly classical mechanics (Zero AI).
- **Accurate Physics**: Lennard-Jones 12-6 potential.
- **Robust Integration**: Velocity Verlet algorithm (Symplectic, Time-Reversible).
- **Thermodynamic Analysis**: Calculate Temperature, Kinetic/Potential Energy, Radial Distribution Function (RDF).
- **Professional Engineering**: Type-hinted, tested, and documented codebase.

## Installation

```bash
pip install .
```

## Quick Start

You can run a simulation directly from the command line:

```bash
python -m pymd.cli -n 100 -t 0.85 --density 0.8 -s 1000 -o output
```

Or use the Python API:

```python
from pymd.system import System
from pymd.integrator import VelocityVerlet
from pymd.simulation import Simulation

# Initialize Argon-like system (in reduced units)
system = System(n_particles=100, box_length=5.0)
system.initialize_positions_lattice()
system.initialize_velocities(temperature=0.85)

# Setup Integrator
integrator = VelocityVerlet(dt=0.005)

# Run Simulation
sim = Simulation(system, integrator)
sim.run(n_steps=1000, output_interval=10)
sim.save_results("argon_run")
```

For development:

```bash
pip install -e .[dev]
python -m pytest tests/
```

## Theoretical Background

### Interaction Potential
The simulation uses the **Lennard-Jones 12-6 potential**:
$$ V(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right] $$
where $\epsilon$ is the depth of the potential well and $\sigma$ is the finite distance at which the inter-particle potential is zero.

### Integration Scheme
The equations of motion are solved using the **Velocity Verlet** algorithm, which is symplectic and time-reversible, ensuring long-term energy stability:
1. $v(t + \frac{\Delta t}{2}) = v(t) + \frac{F(t)}{m} \frac{\Delta t}{2}$
2. $r(t + \Delta t) = r(t) + v(t + \frac{\Delta t}{2}) \Delta t$
3. Calculate $F(t + \Delta t)$ using $r(t + \Delta t)$
4. $v(t + \Delta t) = v(t + \frac{\Delta t}{2}) + \frac{F(t + \Delta t)}{m} \frac{\Delta t}{2}$

### Periodic Boundary Conditions (PBC)
The simulation uses a cubic box with periodic boundary conditions. The minimum image convention is applied to calculate distances between particles.

### Units
The code operates in **reduced units**:
- Length: $\sigma$
- Energy: $\epsilon$
- Mass: $m$
- Time: $\sigma \sqrt{m/\epsilon}$
- Temperature: $\epsilon / k_B$

For Argon:
- $\sigma \approx 3.4 \mathring{A}$
- $\epsilon/k_B \approx 120 K$
- $m \approx 39.95 \text{ amu}$

## License
MIT
