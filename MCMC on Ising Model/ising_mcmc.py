"""
Markov Chain Monte Carlo simulation of the 2D Ising Model

This module implements the Ising model using the Metropolis-Hastings algorithm
for Monte Carlo sampling. The Ising model is a mathematical model of ferromagnetism
in statistical mechanics.

Author: Created for MCMC Ising Model demonstration
Date: 2025-10-26
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import time


class IsingModel:
    """
    2D Ising Model with Metropolis-Hastings MCMC sampling
    
    The Ising model consists of discrete variables (spins) that can be +1 or -1,
    arranged on a 2D lattice. Each spin interacts with its nearest neighbors.
    """
    
    def __init__(self, size: int = 50, temperature: float = 2.0, 
                 coupling: float = 1.0, external_field: float = 0.0):
        """
        Initialize the Ising model
        
        Parameters:
        -----------
        size : int
            Size of the square lattice (size x size)
        temperature : float
            Temperature in units of J/k_B (J is coupling strength)
        coupling : float
            Coupling strength J between nearest neighbors
        external_field : float
            External magnetic field strength
        """
        self.size = size
        self.temperature = temperature
        self.coupling = coupling
        self.external_field = external_field
        self.beta = 1.0 / temperature  # Inverse temperature
        
        # Initialize random spin configuration
        self.spins = np.random.choice([-1, 1], size=(size, size))
        
        # Precompute Boltzmann factors for efficiency
        self._precompute_boltzmann_factors()
        
        # Statistics tracking
        self.energy_history = []
        self.magnetization_history = []
        self.acceptance_rate = 0.0
        self.total_flips = 0
        self.accepted_flips = 0
    
    def _precompute_boltzmann_factors(self):
        """Precompute Boltzmann factors for all possible energy differences"""
        # Possible energy differences for spin flip: -8J, -4J, 0, +4J, +8J
        # (considering nearest neighbor interactions only)
        self.boltzmann_factors = {}
        for dE in [-8, -4, 0, 4, 8]:
            self.boltzmann_factors[dE] = np.exp(-self.beta * self.coupling * dE)
    
    def calculate_energy(self) -> float:
        """
        Calculate total energy of the current configuration
        
        E = -J * sum(s_i * s_j) - h * sum(s_i)
        where the first sum is over nearest neighbor pairs
        """
        # Nearest neighbor interactions
        energy = 0.0
        
        # Horizontal neighbors
        energy -= self.coupling * np.sum(self.spins * np.roll(self.spins, 1, axis=1))
        # Vertical neighbors  
        energy -= self.coupling * np.sum(self.spins * np.roll(self.spins, 1, axis=0))
        
        # External field contribution
        energy -= self.external_field * np.sum(self.spins)
        
        return energy
    
    def calculate_magnetization(self) -> float:
        """Calculate total magnetization (sum of all spins)"""
        return np.sum(self.spins)
    
    def calculate_local_energy(self, i: int, j: int) -> float:
        """
        Calculate energy contribution from a single spin at position (i,j)
        considering its interaction with nearest neighbors
        """
        # Get neighbors with periodic boundary conditions
        neighbors = (
            self.spins[(i-1) % self.size, j] +
            self.spins[(i+1) % self.size, j] +
            self.spins[i, (j-1) % self.size] +
            self.spins[i, (j+1) % self.size]
        )
        
        return -self.coupling * self.spins[i, j] * neighbors - self.external_field * self.spins[i, j]
    
    def metropolis_step(self) -> bool:
        """
        Perform a single Metropolis step:
        1. Choose a random spin
        2. Calculate energy change if flipped
        3. Accept or reject based on Metropolis criterion
        
        Returns:
        --------
        bool : True if spin flip was accepted, False otherwise
        """
        # Choose random spin
        i, j = np.random.randint(0, self.size, 2)
        
        # Calculate energy difference for flipping this spin
        # Delta E = E_new - E_old = -2 * E_local (since spin changes sign)
        local_energy = self.calculate_local_energy(i, j)
        delta_E = -2 * local_energy
        
        # Metropolis acceptance criterion
        if delta_E <= 0:
            # Always accept if energy decreases
            accept = True
        else:
            # Accept with probability exp(-beta * delta_E)
            # Use precomputed Boltzmann factors for efficiency
            if int(delta_E) in self.boltzmann_factors:
                accept = np.random.random() < self.boltzmann_factors[int(delta_E)]
            else:
                accept = np.random.random() < np.exp(-self.beta * delta_E)
        
        if accept:
            self.spins[i, j] *= -1  # Flip the spin
            self.accepted_flips += 1
        
        self.total_flips += 1
        return accept
    
    def monte_carlo_sweep(self) -> Tuple[float, float]:
        """
        Perform one Monte Carlo sweep (N spin flip attempts where N = size^2)
        
        Returns:
        --------
        Tuple[float, float] : (energy, magnetization) after the sweep
        """
        # Perform size^2 Metropolis steps (one sweep)
        for _ in range(self.size * self.size):
            self.metropolis_step()
        
        # Calculate current energy and magnetization
        energy = self.calculate_energy()
        magnetization = self.calculate_magnetization()
        
        # Store statistics
        self.energy_history.append(energy)
        self.magnetization_history.append(magnetization)
        
        return energy, magnetization
    
    def run_simulation(self, n_sweeps: int = 1000, equilibration_sweeps: int = 100,
                      verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Run the complete MCMC simulation
        
        Parameters:
        -----------
        n_sweeps : int
            Number of Monte Carlo sweeps to perform
        equilibration_sweeps : int
            Number of initial sweeps to discard for equilibration
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        Dict containing simulation results
        """
        if verbose:
            print(f"Running MCMC simulation for 2D Ising model")
            print(f"Lattice size: {self.size}x{self.size}")
            print(f"Temperature: {self.temperature:.2f}")
            print(f"Coupling: {self.coupling:.2f}")
            print(f"External field: {self.external_field:.2f}")
            print(f"Total sweeps: {n_sweeps}")
            print(f"Equilibration sweeps: {equilibration_sweeps}")
            print("-" * 50)
        
        start_time = time.time()
        
        # Reset statistics
        self.energy_history = []
        self.magnetization_history = []
        self.accepted_flips = 0
        self.total_flips = 0
        
        # Run simulation
        for sweep in range(n_sweeps):
            energy, magnetization = self.monte_carlo_sweep()
            
            if verbose and (sweep + 1) % (n_sweeps // 10) == 0:
                acceptance_rate = self.accepted_flips / self.total_flips if self.total_flips > 0 else 0
                print(f"Sweep {sweep + 1:4d}/{n_sweeps}: "
                      f"E = {energy:8.2f}, M = {magnetization:6.1f}, "
                      f"Acceptance = {acceptance_rate:.3f}")
        
        # Calculate final acceptance rate
        self.acceptance_rate = self.accepted_flips / self.total_flips if self.total_flips > 0 else 0
        
        # Remove equilibration period
        equilibrated_energies = np.array(self.energy_history[equilibration_sweeps:])
        equilibrated_magnetizations = np.array(self.magnetization_history[equilibration_sweeps:])
        
        # Calculate statistics
        avg_energy = np.mean(equilibrated_energies)
        avg_magnetization = np.mean(np.abs(equilibrated_magnetizations))  # Use absolute value
        
        # Calculate specific heat and magnetic susceptibility
        energy_var = np.var(equilibrated_energies)
        mag_var = np.var(equilibrated_magnetizations)
        
        specific_heat = energy_var / (self.temperature**2 * self.size**2)
        susceptibility = mag_var / (self.temperature * self.size**2)
        
        end_time = time.time()
        
        if verbose:
            print("-" * 50)
            print(f"Simulation completed in {end_time - start_time:.2f} seconds")
            print(f"Final acceptance rate: {self.acceptance_rate:.3f}")
            print(f"Average energy per spin: {avg_energy / self.size**2:.4f}")
            print(f"Average magnetization per spin: {avg_magnetization / self.size**2:.4f}")
            print(f"Specific heat: {specific_heat:.4f}")
            print(f"Magnetic susceptibility: {susceptibility:.4f}")
        
        return {
            'energy_history': np.array(self.energy_history),
            'magnetization_history': np.array(self.magnetization_history),
            'equilibrated_energies': equilibrated_energies,
            'equilibrated_magnetizations': equilibrated_magnetizations,
            'avg_energy': avg_energy,
            'avg_magnetization': avg_magnetization,
            'specific_heat': specific_heat,
            'susceptibility': susceptibility,
            'acceptance_rate': self.acceptance_rate,
            'final_spins': self.spins.copy()
        }
    
    def set_temperature(self, temperature: float):
        """Change the temperature and update beta"""
        self.temperature = temperature
        self.beta = 1.0 / temperature
        self._precompute_boltzmann_factors()
    
    def reset_spins(self, random_state: Optional[int] = None):
        """Reset spins to random configuration"""
        if random_state is not None:
            np.random.seed(random_state)
        self.spins = np.random.choice([-1, 1], size=(self.size, self.size))


def temperature_sweep(size: int = 32, temperatures: np.ndarray = None, 
                     n_sweeps: int = 1000, equilibration_sweeps: int = 200) -> Dict:
    """
    Perform a temperature sweep to study phase transition
    
    Parameters:
    -----------
    size : int
        Lattice size
    temperatures : np.ndarray
        Array of temperatures to simulate
    n_sweeps : int
        Number of sweeps per temperature
    equilibration_sweeps : int
        Number of equilibration sweeps per temperature
        
    Returns:
    --------
    Dict containing results for all temperatures
    """
    if temperatures is None:
        # Focus around critical temperature (T_c ≈ 2.269 for 2D Ising model)
        temperatures = np.linspace(1.5, 3.5, 20)
    
    results = {
        'temperatures': temperatures,
        'energies': [],
        'magnetizations': [],
        'specific_heats': [],
        'susceptibilities': []
    }
    
    print("Starting temperature sweep...")
    print(f"Temperature range: {temperatures[0]:.2f} to {temperatures[-1]:.2f}")
    
    # Initialize model
    model = IsingModel(size=size)
    
    for i, T in enumerate(temperatures):
        print(f"\nTemperature {i+1}/{len(temperatures)}: T = {T:.3f}")
        
        # Set new temperature
        model.set_temperature(T)
        model.reset_spins()  # Start fresh for each temperature
        
        # Run simulation
        sim_results = model.run_simulation(n_sweeps=n_sweeps, 
                                         equilibration_sweeps=equilibration_sweeps,
                                         verbose=False)
        
        # Store results
        results['energies'].append(sim_results['avg_energy'] / size**2)
        results['magnetizations'].append(sim_results['avg_magnetization'] / size**2)
        results['specific_heats'].append(sim_results['specific_heat'])
        results['susceptibilities'].append(sim_results['susceptibility'])
        
        print(f"  <E> = {results['energies'][-1]:.4f}")
        print(f"  <|M|> = {results['magnetizations'][-1]:.4f}")
        print(f"  C = {results['specific_heats'][-1]:.4f}")
        print(f"  χ = {results['susceptibilities'][-1]:.4f}")
    
    # Convert to numpy arrays
    for key in ['energies', 'magnetizations', 'specific_heats', 'susceptibilities']:
        results[key] = np.array(results[key])
    
    return results


if __name__ == "__main__":
    # Quick demonstration
    print("MCMC Ising Model Demonstration")
    print("=" * 40)
    
    # Create and run a simple simulation
    model = IsingModel(size=20, temperature=2.5)
    results = model.run_simulation(n_sweeps=500, equilibration_sweeps=100)
    
    print(f"\nFinal spin configuration has {np.sum(results['final_spins'] == 1)} up spins and {np.sum(results['final_spins'] == -1)} down spins")