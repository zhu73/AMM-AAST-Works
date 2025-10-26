# MCMC Simulation: Particles in a Box with Lennard-Jones Potential

This project implements a Markov Chain Monte Carlo (MCMC) simulation of particles confined in a box and interacting through the Lennard-Jones potential. The simulation uses the Metropolis-Hastings algorithm to sample the canonical ensemble.

## Overview

The Lennard-Jones potential models the interaction between a pair of neutral atoms or molecules:

```
V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
```

where:
- `r` is the distance between two particles
- `ε` (epsilon) is the depth of the potential well
- `σ` (sigma) is the finite distance at which the inter-particle potential is zero

## Features

- **MCMC Simulation**: Metropolis-Hastings algorithm for sampling particle configurations
- **Periodic Boundary Conditions**: Simulates an infinite system by wrapping particles around box edges
- **Energy Calculations**: Efficient computation of total potential energy
- **Radial Distribution Function**: Calculate g(r) to analyze particle correlations
- **Comprehensive Visualization**: Multiple plotting functions for analysis
- **Temperature Control**: Simulate systems at different temperatures
- **Acceptance Rate Monitoring**: Track Monte Carlo move acceptance

## File Structure

```
MCMC particle in box LJ Potential/
├── mcmc_lj_simulation.py      # Core MCMC simulation engine
├── visualize_simulation.py     # Visualization tools
├── examples.py                 # Example demonstrations
└── README.md                   # This file
```

## Installation

Required packages:
```bash
pip install numpy matplotlib
```

## Usage

### Basic Simulation

```python
import numpy as np
from mcmc_lj_simulation import SimulationParameters, LennardJonesSystem

# Set random seed for reproducibility
np.random.seed(42)

# Configure simulation parameters
params = SimulationParameters(
    n_particles=20,           # Number of particles
    box_length=10.0,          # Box size
    temperature=1.0,          # Temperature (reduced units)
    epsilon=1.0,              # LJ energy parameter
    sigma=1.0,                # LJ distance parameter
    max_displacement=0.3,     # Maximum MC move size
    n_steps=10000,            # Production steps
    equilibration_steps=2000  # Equilibration steps
)

# Create and run simulation
system = LennardJonesSystem(params)
results = system.run_simulation(verbose=True)

# Access results
print(f"Mean energy: {results['mean_energy']:.4f}")
print(f"Acceptance rate: {results['acceptance_rate']:.4f}")
```

### Running Examples

The `examples.py` file contains several demonstration scripts:

```bash
# Run all examples
python examples.py

# Run specific example
python examples.py 1  # Basic simulation
python examples.py 2  # Temperature comparison
python examples.py 3  # Density effects
python examples.py 4  # Convergence analysis
```

### Visualization

```python
from visualize_simulation import create_comprehensive_report

# Create all visualizations and save to directory
create_comprehensive_report(system, results, save_dir='simulation_results')

# Or create individual plots
from visualize_simulation import (
    plot_energy_evolution,
    plot_particle_configuration_3d,
    plot_radial_distribution
)

plot_energy_evolution(results['energies'])
plot_particle_configuration_3d(results['final_positions'], params.box_length)

r, g = system.get_radial_distribution_function()
plot_radial_distribution(r, g)
```

## Simulation Details

### Metropolis-Hastings Algorithm

1. **Initialization**: Place particles randomly in the box
2. **Equilibration**: Run MC steps to reach equilibrium (discarded)
3. **Production**: Run MC steps and collect statistics
4. **MC Step**:
   - Select a random particle
   - Propose a random displacement
   - Calculate energy change ΔE
   - Accept with probability min(1, exp(-βΔE)) where β = 1/T

### Periodic Boundary Conditions

The simulation uses periodic boundary conditions to minimize finite-size effects:
- When calculating distances, the minimum image convention is used
- Particles that move outside the box are wrapped to the opposite side

### Energy Calculation

The total potential energy is the sum of all pairwise interactions:

```
E_total = Σ(i=1 to N-1) Σ(j=i+1 to N) V(r_ij)
```

where the sum runs over all unique pairs of particles.

## Parameters Guide

### Temperature Effects
- **Low T (< 0.5)**: System tends toward lowest energy configuration (more solid-like)
- **Medium T (~1.0)**: Liquid-like behavior with some structure
- **High T (> 2.0)**: Gas-like behavior, particles more independent

### Density Effects
- **Low density**: Particles rarely interact, gas-like
- **High density**: Strong correlations, more structured g(r)

### Tuning Acceptance Rate
- **Target**: 30-50% acceptance rate is typically good
- **Too high**: Increase `max_displacement`
- **Too low**: Decrease `max_displacement`

## Example Results

### Energy Evolution
The energy should equilibrate after the equilibration phase and fluctuate around a mean value during production.

### Radial Distribution Function
- **g(r) = 0** for r < σ: Hard-core repulsion prevents overlap
- **g(r) > 1** at r ≈ σ: First coordination shell (nearest neighbors)
- **g(r) → 1** for large r: Long-range correlations decay

### Particle Configurations
Visualizations show the spatial arrangement of particles, which can reveal:
- Clustering at low temperatures
- Liquid-like structure at moderate temperatures
- Uniform distribution at high temperatures

## Physical Units

This simulation uses **reduced units** where:
- Length in units of σ
- Energy in units of ε
- Temperature in units of ε/k_B (where k_B is Boltzmann's constant)

To convert to real units for a specific system (e.g., Argon):
- σ ≈ 3.4 Å
- ε/k_B ≈ 120 K

## Tips and Troubleshooting

1. **System crashes or infinite energies**: 
   - Reduce `max_displacement`
   - Ensure particles don't start too close together

2. **Poor sampling**:
   - Increase `n_steps`
   - Increase `equilibration_steps`
   - Adjust `max_displacement` to achieve ~40% acceptance

3. **High density simulations**:
   - May require longer equilibration
   - Smaller `max_displacement` needed
   - Consider starting from a lattice configuration

## References

1. Frenkel, D., & Smit, B. (2002). Understanding Molecular Simulation: From Algorithms to Applications.
2. Metropolis, N., et al. (1953). Equation of state calculations by fast computing machines. J. Chem. Phys., 21(6), 1087-1092.
3. Allen, M. P., & Tildesley, D. J. (2017). Computer Simulation of Liquids.

## License

This code is provided for educational purposes.

## Author

Created as part of a computational physics/chemistry project.
