"""
Markov Chain Monte Carlo simulation of particles in a box
with Lennard-Jones potential interactions.

This module implements the Metropolis-Hastings algorithm to simulate
a system of particles interacting via the Lennard-Jones potential.
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SimulationParameters:
    """Parameters for the MCMC simulation."""
    n_particles: int = 20  # Number of particles
    box_length: float = 10.0  # Size of the cubic box
    temperature: float = 1.0  # Temperature (in reduced units)
    epsilon: float = 1.0  # LJ energy parameter
    sigma: float = 1.0  # LJ distance parameter
    max_displacement: float = 0.5  # Maximum displacement for MC moves
    n_steps: int = 10000  # Number of MCMC steps
    equilibration_steps: int = 1000  # Steps for equilibration


class LennardJonesSystem:
    """
    A system of particles in a box with Lennard-Jones interactions.
    """
    
    def __init__(self, params: SimulationParameters):
        """
        Initialize the system.
        
        Args:
            params: Simulation parameters
        """
        self.params = params
        self.positions = self._initialize_positions()
        self.energy_history = []
        self.acceptance_rate = 0
        self.n_accepted = 0
        self.n_attempted = 0
        
    def _initialize_positions(self) -> np.ndarray:
        """
        Initialize particle positions randomly in the box.
        
        Returns:
            Array of shape (n_particles, 3) with particle positions
        """
        return np.random.uniform(
            0, self.params.box_length, 
            (self.params.n_particles, 3)
        )
    
    def _apply_periodic_boundary(self, r: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions.
        
        Args:
            r: Position vector or displacement vector
            
        Returns:
            Position/displacement with periodic boundaries applied
        """
        return r - self.params.box_length * np.round(r / self.params.box_length)
    
    def _lennard_jones_potential(self, r: float) -> float:
        """
        Calculate Lennard-Jones potential for a given distance.
        
        V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
        
        Args:
            r: Distance between two particles
            
        Returns:
            Potential energy
        """
        if r < 1e-10:  # Avoid division by zero
            return np.inf
        
        sigma_over_r = self.params.sigma / r
        sigma_over_r_6 = sigma_over_r ** 6
        sigma_over_r_12 = sigma_over_r_6 ** 2
        
        return 4 * self.params.epsilon * (sigma_over_r_12 - sigma_over_r_6)
    
    def calculate_total_energy(self, positions: Optional[np.ndarray] = None) -> float:
        """
        Calculate total potential energy of the system.
        
        Args:
            positions: Particle positions (uses self.positions if None)
            
        Returns:
            Total potential energy
        """
        if positions is None:
            positions = self.positions
            
        energy = 0.0
        n = len(positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate distance with periodic boundary conditions
                r_vec = positions[i] - positions[j]
                r_vec = self._apply_periodic_boundary(r_vec)
                r = np.linalg.norm(r_vec)
                
                # Add pair potential
                energy += self._lennard_jones_potential(r)
        
        return energy
    
    def _propose_move(self) -> Tuple[int, np.ndarray]:
        """
        Propose a random move for a randomly selected particle.
        
        Returns:
            Tuple of (particle_index, new_position)
        """
        # Select random particle
        particle_idx = np.random.randint(0, self.params.n_particles)
        
        # Generate random displacement
        displacement = np.random.uniform(
            -self.params.max_displacement,
            self.params.max_displacement,
            3
        )
        
        # Calculate new position with periodic boundaries
        new_position = self.positions[particle_idx] + displacement
        new_position = new_position % self.params.box_length
        
        return particle_idx, new_position
    
    def _calculate_energy_change(self, particle_idx: int, new_position: np.ndarray) -> float:
        """
        Calculate change in energy for a proposed move.
        
        Args:
            particle_idx: Index of the particle to move
            new_position: Proposed new position
            
        Returns:
            Energy change (E_new - E_old)
        """
        old_energy = 0.0
        new_energy = 0.0
        
        old_position = self.positions[particle_idx]
        
        # Calculate energy change only for pairs involving the moved particle
        for j in range(self.params.n_particles):
            if j == particle_idx:
                continue
            
            # Old energy
            r_vec_old = old_position - self.positions[j]
            r_vec_old = self._apply_periodic_boundary(r_vec_old)
            r_old = np.linalg.norm(r_vec_old)
            old_energy += self._lennard_jones_potential(r_old)
            
            # New energy
            r_vec_new = new_position - self.positions[j]
            r_vec_new = self._apply_periodic_boundary(r_vec_new)
            r_new = np.linalg.norm(r_vec_new)
            new_energy += self._lennard_jones_potential(r_new)
        
        return new_energy - old_energy
    
    def metropolis_step(self) -> bool:
        """
        Perform one Metropolis-Hastings step.
        
        Returns:
            True if move was accepted, False otherwise
        """
        # Propose a move
        particle_idx, new_position = self._propose_move()
        
        # Calculate energy change
        delta_E = self._calculate_energy_change(particle_idx, new_position)
        
        # Metropolis acceptance criterion
        beta = 1.0 / self.params.temperature
        acceptance_probability = min(1.0, np.exp(-beta * delta_E))
        
        self.n_attempted += 1
        
        if np.random.random() < acceptance_probability:
            # Accept the move
            self.positions[particle_idx] = new_position
            self.n_accepted += 1
            return True
        else:
            # Reject the move
            return False
    
    def run_simulation(self, verbose: bool = True) -> dict:
        """
        Run the full MCMC simulation.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing simulation results
        """
        if verbose:
            print("Starting MCMC simulation...")
            print(f"Particles: {self.params.n_particles}")
            print(f"Temperature: {self.params.temperature}")
            print(f"Steps: {self.params.n_steps}")
            print(f"Equilibration: {self.params.equilibration_steps}")
            print()
        
        # Equilibration phase
        if verbose:
            print("Equilibration phase...")
        for step in range(self.params.equilibration_steps):
            self.metropolis_step()
        
        # Reset acceptance counters after equilibration
        self.n_accepted = 0
        self.n_attempted = 0
        
        # Production phase
        if verbose:
            print("Production phase...")
        
        energies = []
        positions_history = []
        
        for step in range(self.params.n_steps):
            self.metropolis_step()
            
            # Record energy
            current_energy = self.calculate_total_energy()
            energies.append(current_energy)
            
            # Optionally save positions (every 100 steps to save memory)
            if step % 100 == 0:
                positions_history.append(self.positions.copy())
            
            # Print progress
            if verbose and (step + 1) % 1000 == 0:
                current_acceptance = self.n_accepted / self.n_attempted if self.n_attempted > 0 else 0
                print(f"Step {step + 1}/{self.params.n_steps}, "
                      f"Energy: {current_energy:.2f}, "
                      f"Acceptance: {current_acceptance:.3f}")
        
        self.acceptance_rate = self.n_accepted / self.n_attempted if self.n_attempted > 0 else 0
        self.energy_history = energies
        
        if verbose:
            print()
            print("Simulation complete!")
            print(f"Final acceptance rate: {self.acceptance_rate:.3f}")
            print(f"Average energy: {np.mean(energies):.2f}")
            print(f"Final energy: {energies[-1]:.2f}")
        
        return {
            'energies': np.array(energies),
            'positions_history': positions_history,
            'final_positions': self.positions.copy(),
            'acceptance_rate': self.acceptance_rate,
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies)
        }
    
    def get_radial_distribution_function(self, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the radial distribution function g(r).
        
        Args:
            n_bins: Number of bins for the histogram
            
        Returns:
            Tuple of (r_values, g_values)
        """
        max_r = self.params.box_length / 2
        bin_edges = np.linspace(0, max_r, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        dr = bin_edges[1] - bin_edges[0]
        
        histogram = np.zeros(n_bins)
        
        # Calculate all pairwise distances
        n = len(self.positions)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = self.positions[i] - self.positions[j]
                r_vec = self._apply_periodic_boundary(r_vec)
                r = np.linalg.norm(r_vec)
                
                if r < max_r:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        histogram[bin_idx] += 2  # Count both i-j and j-i
        
        # Normalize by ideal gas
        volume = self.params.box_length ** 3
        number_density = self.params.n_particles / volume
        
        g = np.zeros(n_bins)
        for i in range(n_bins):
            r = bin_centers[i]
            shell_volume = 4 * np.pi * r**2 * dr
            expected_count = number_density * shell_volume * self.params.n_particles
            if expected_count > 0:
                g[i] = histogram[i] / expected_count
        
        return bin_centers, g


def main():
    """Example usage of the MCMC LJ simulation."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create simulation parameters
    params = SimulationParameters(
        n_particles=20,
        box_length=10.0,
        temperature=1.0,
        epsilon=1.0,
        sigma=1.0,
        max_displacement=0.3,
        n_steps=10000,
        equilibration_steps=2000
    )
    
    # Create and run simulation
    system = LennardJonesSystem(params)
    results = system.run_simulation(verbose=True)
    
    print("\n" + "="*50)
    print("Simulation Results Summary")
    print("="*50)
    print(f"Mean energy: {results['mean_energy']:.4f}")
    print(f"Std energy: {results['std_energy']:.4f}")
    print(f"Acceptance rate: {results['acceptance_rate']:.4f}")
    

if __name__ == "__main__":
    main()
