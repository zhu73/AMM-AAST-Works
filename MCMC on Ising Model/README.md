# Markov Chain Monte Carlo Simulation of the 2D Ising Model

This repository contains a comprehensive implementation of the 2D Ising model using Markov Chain Monte Carlo (MCMC) simulation with the Metropolis-Hastings algorithm.

## Overview

The Ising model is a mathematical model of ferromagnetism in statistical mechanics, consisting of discrete variables (spins) that can be +1 or -1 arranged on a lattice. This implementation studies the phase transition between ferromagnetic and paramagnetic phases.

## Features

- **Complete MCMC Implementation**: Metropolis-Hastings algorithm for sampling spin configurations
- **Phase Transition Analysis**: Study of critical temperature and thermodynamic properties
- **Comprehensive Visualizations**: Spin configurations, energy evolution, and phase diagrams
- **Advanced Analysis Tools**: Temperature sweeps, finite-size scaling, correlation functions
- **Interactive Jupyter Notebook**: Step-by-step demonstration with explanations

## Files

- `ising_mcmc.py` - Main Ising model implementation with MCMC algorithms
- `visualization.py` - Comprehensive visualization functions for analysis
- `temperature_analysis.py` - Advanced temperature analysis and critical phenomena
- `ising_mcmc_demo.ipynb` - Interactive Jupyter notebook demonstration
- `requirements.txt` - Required Python packages

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook ising_mcmc_demo.ipynb
   ```

3. **Or use the Python modules directly**:
   ```python
   from ising_mcmc import IsingModel
   from visualization import plot_spin_configuration
   
   # Create and run simulation
   model = IsingModel(size=32, temperature=2.5)
   results = model.run_simulation(n_sweeps=500)
   
   # Visualize results
   plot_spin_configuration(results['final_spins'])
   ```

## Key Results

- **Critical Temperature**: T_c ≈ 2.269 (theoretical: 2.269185...)
- **Phase Transition**: Clear second-order transition between ordered and disordered phases
- **Thermodynamic Properties**: Calculation of specific heat and magnetic susceptibility
- **Finite-Size Effects**: Analysis of how lattice size affects critical behavior

## Theory

### Ising Hamiltonian
The energy of the system is given by:
```
H = -J ∑⟨i,j⟩ sᵢsⱼ - h ∑ᵢ sᵢ
```

Where:
- J is the coupling constant (interaction strength)
- h is the external magnetic field
- sᵢ ∈ {-1, +1} represents the spin at site i
- ⟨i,j⟩ denotes nearest neighbor pairs

### Metropolis-Hastings Algorithm
1. Choose a random spin
2. Calculate energy change ΔE for flipping the spin
3. Accept flip if:
   - ΔE ≤ 0 (energy decreases), or
   - Random number < exp(-βΔE) (Boltzmann probability)

## Applications

The Ising model and MCMC methods are used in:
- Statistical physics and phase transitions
- Materials science and magnetism
- Machine learning (Hopfield networks, Boltzmann machines)
- Social sciences (opinion dynamics)
- Biology (protein folding, neural networks)

## Extensions

Possible extensions include:
- 3D Ising model
- Different lattice geometries
- External magnetic fields and hysteresis
- Advanced algorithms (cluster algorithms)
- Other spin models (Potts, XY, Heisenberg)

## Author

Created for MCMC Ising Model demonstration - October 26, 2025

## License

This project is provided for educational purposes.