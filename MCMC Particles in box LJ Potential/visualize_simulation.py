"""
Visualization tools for the MCMC Lennard-Jones simulation.

This module provides functions to visualize:
- Energy evolution over time
- Particle configurations in 2D and 3D
- Radial distribution function
- Acceptance rate statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List
import mcmc_lj_simulation as mcmc


def plot_energy_evolution(energies: np.ndarray, 
                         equilibration_steps: int = 0,
                         save_path: Optional[str] = None):
    """
    Plot the evolution of energy over MCMC steps.
    
    Args:
        energies: Array of energies at each step
        equilibration_steps: Number of equilibration steps (for reference line)
        save_path: Path to save the figure (if None, displays instead)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    steps = np.arange(len(energies))
    
    # Full energy trace
    ax1.plot(steps, energies, alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('MCMC Step', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title('Energy Evolution During MCMC Simulation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Running average
    window_size = min(100, len(energies) // 10)
    if window_size > 1:
        running_avg = np.convolve(energies, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(steps[window_size-1:], running_avg, 'r-', linewidth=2, 
                label=f'Running avg (window={window_size})')
        ax1.legend()
    
    # Histogram of energies
    ax2.hist(energies, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(energies), color='r', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(energies):.2f}')
    ax2.axvline(np.median(energies), color='g', linestyle='--', linewidth=2,
                label=f'Median = {np.median(energies):.2f}')
    ax2.set_xlabel('Energy', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Energy Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Energy plot saved to {save_path}")
    else:
        plt.show()


def plot_particle_configuration_3d(positions: np.ndarray,
                                   box_length: float,
                                   save_path: Optional[str] = None):
    """
    Plot particle configuration in 3D.
    
    Args:
        positions: Array of particle positions (n_particles, 3)
        box_length: Size of the simulation box
        save_path: Path to save the figure (if None, displays instead)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot particles
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c='blue', marker='o', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Draw box
    # Define box corners
    corners = np.array([
        [0, 0, 0], [box_length, 0, 0], [box_length, box_length, 0], [0, box_length, 0],
        [0, 0, box_length], [box_length, 0, box_length], 
        [box_length, box_length, box_length], [0, box_length, box_length]
    ])
    
    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for edge in edges:
        points = corners[edge]
        ax.plot3D(*points.T, 'k-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Particle Configuration ({len(positions)} particles)', 
                fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    ax.set_xlim([0, box_length])
    ax.set_ylim([0, box_length])
    ax.set_zlim([0, box_length])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D configuration plot saved to {save_path}")
    else:
        plt.show()


def plot_particle_configuration_2d(positions: np.ndarray,
                                   box_length: float,
                                   save_path: Optional[str] = None):
    """
    Plot particle configuration in 2D (xy projection).
    
    Args:
        positions: Array of particle positions (n_particles, 3)
        box_length: Size of the simulation box
        save_path: Path to save the figure (if None, displays instead)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot particles (xy projection)
    scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                        c=positions[:, 2], cmap='viridis',
                        s=200, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Draw box
    box = plt.Rectangle((0, 0), box_length, box_length, 
                        fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Colorbar for z-coordinate
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Z coordinate', fontsize=12)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Particle Configuration - XY Projection ({len(positions)} particles)',
                fontsize=14, fontweight='bold')
    ax.set_xlim([-0.5, box_length + 0.5])
    ax.set_ylim([-0.5, box_length + 0.5])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D configuration plot saved to {save_path}")
    else:
        plt.show()


def plot_radial_distribution(r: np.ndarray, 
                            g: np.ndarray,
                            save_path: Optional[str] = None):
    """
    Plot radial distribution function g(r).
    
    Args:
        r: Array of radial distances
        g: Array of g(r) values
        save_path: Path to save the figure (if None, displays instead)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(r, g, linewidth=2, color='navy')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, 
              label='Ideal gas (g(r) = 1)', alpha=0.7)
    
    ax.set_xlabel('Distance r', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title('Radial Distribution Function', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([0, r[-1]])
    ax.set_ylim([0, max(3, np.max(g) * 1.1)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RDF plot saved to {save_path}")
    else:
        plt.show()


def create_comprehensive_report(system: mcmc.LennardJonesSystem,
                                results: dict,
                                save_dir: Optional[str] = None):
    """
    Create a comprehensive visualization report of the simulation.
    
    Args:
        system: The LennardJonesSystem object after simulation
        results: Results dictionary from run_simulation
        save_dir: Directory to save figures (if None, displays instead)
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Energy evolution
    save_path = os.path.join(save_dir, 'energy_evolution.png') if save_dir else None
    plot_energy_evolution(results['energies'], 
                         system.params.equilibration_steps,
                         save_path)
    
    # 2. 3D particle configuration
    save_path = os.path.join(save_dir, 'configuration_3d.png') if save_dir else None
    plot_particle_configuration_3d(results['final_positions'],
                                   system.params.box_length,
                                   save_path)
    
    # 3. 2D particle configuration
    save_path = os.path.join(save_dir, 'configuration_2d.png') if save_dir else None
    plot_particle_configuration_2d(results['final_positions'],
                                   system.params.box_length,
                                   save_path)
    
    # 4. Radial distribution function
    r, g = system.get_radial_distribution_function()
    save_path = os.path.join(save_dir, 'radial_distribution.png') if save_dir else None
    plot_radial_distribution(r, g, save_path)
    
    # 5. Summary statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy autocorrelation
    ax = axes[0, 0]
    energies = results['energies']
    max_lag = min(1000, len(energies) // 4)
    autocorr = np.correlate(energies - np.mean(energies), 
                           energies - np.mean(energies), 
                           mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr[:max_lag] / autocorr[0]
    ax.plot(autocorr, linewidth=2)
    ax.set_xlabel('Lag', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.set_title('Energy Autocorrelation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Energy statistics
    ax = axes[0, 1]
    stats_text = f"""
    Simulation Statistics
    ──────────────────────
    Particles: {system.params.n_particles}
    Temperature: {system.params.temperature:.2f}
    Box Length: {system.params.box_length:.2f}
    
    Mean Energy: {results['mean_energy']:.4f}
    Std Energy: {results['std_energy']:.4f}
    Min Energy: {np.min(energies):.4f}
    Max Energy: {np.max(energies):.4f}
    
    Acceptance Rate: {results['acceptance_rate']:.3f}
    Total Steps: {len(energies)}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    ax.axis('off')
    
    # Pairwise distance distribution
    ax = axes[1, 0]
    positions = results['final_positions']
    distances = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r_vec = r_vec - system.params.box_length * np.round(r_vec / system.params.box_length)
            distances.append(np.linalg.norm(r_vec))
    
    ax.hist(distances, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Pairwise Distance', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Pairwise Distance Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Energy per particle over time (smoothed)
    ax = axes[1, 1]
    energy_per_particle = energies / system.params.n_particles
    window = min(50, len(energy_per_particle) // 20)
    if window > 1:
        smoothed = np.convolve(energy_per_particle, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, linewidth=2, color='darkgreen')
    else:
        ax.plot(energy_per_particle, linewidth=1, alpha=0.7, color='darkgreen')
    ax.set_xlabel('MCMC Step', fontsize=11)
    ax.set_ylabel('Energy per Particle', fontsize=11)
    ax.set_title('Energy per Particle (Smoothed)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'statistics_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics summary saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running MCMC simulation for visualization...")
    
    # Set random seed
    np.random.seed(42)
    
    # Create simulation
    params = mcmc.SimulationParameters(
        n_particles=30,
        box_length=10.0,
        temperature=1.0,
        max_displacement=0.3,
        n_steps=5000,
        equilibration_steps=1000
    )
    
    system = mcmc.LennardJonesSystem(params)
    results = system.run_simulation(verbose=True)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comprehensive_report(system, results, save_dir='simulation_results')
    print("\nAll visualizations complete!")
