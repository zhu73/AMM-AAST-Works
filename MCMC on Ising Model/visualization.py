"""
Visualization functions for the MCMC Ising Model

This module provides comprehensive visualization tools for analyzing
the results of MCMC simulations on the 2D Ising model.

Author: Created for MCMC Ising Model demonstration
Date: 2025-10-26
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from ising_mcmc import IsingModel


# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_spin_configuration(spins: np.ndarray, title: str = "Spin Configuration", 
                          figsize: Tuple[int, int] = (8, 8), save_path: Optional[str] = None):
    """
    Plot the current spin configuration as a 2D lattice
    
    Parameters:
    -----------
    spins : np.ndarray
        2D array of spins (+1 or -1)
    title : str
        Title for the plot
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure (if provided)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap: blue for -1 (down), red for +1 (up)
    colors = ['#3498db', '#e74c3c']  # Blue, Red
    cmap = ListedColormap(colors)
    
    # Plot spins
    im = ax.imshow(spins, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 1], shrink=0.8)
    cbar.set_ticklabels(['↓ (-1)', '↑ (+1)'])
    cbar.set_label('Spin Direction', fontsize=12)
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_energy_magnetization_evolution(energy_history: np.ndarray, 
                                       magnetization_history: np.ndarray,
                                       equilibration_sweeps: int = 0,
                                       figsize: Tuple[int, int] = (12, 5),
                                       save_path: Optional[str] = None):
    """
    Plot the evolution of energy and magnetization during the simulation
    
    Parameters:
    -----------
    energy_history : np.ndarray
        Array of energy values over time
    magnetization_history : np.ndarray
        Array of magnetization values over time
    equilibration_sweeps : int
        Number of equilibration sweeps to mark on the plot
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure (if provided)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    sweeps = np.arange(len(energy_history))
    
    # Plot energy evolution
    ax1.plot(sweeps, energy_history, 'b-', alpha=0.7, linewidth=1)
    ax1.axvline(x=equilibration_sweeps, color='red', linestyle='--', alpha=0.7, 
                label=f'Equilibration ({equilibration_sweeps} sweeps)')
    ax1.set_xlabel('Monte Carlo Sweep')
    ax1.set_ylabel('Total Energy')
    ax1.set_title('Energy Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot magnetization evolution
    ax2.plot(sweeps, magnetization_history, 'g-', alpha=0.7, linewidth=1)
    ax2.axvline(x=equilibration_sweeps, color='red', linestyle='--', alpha=0.7,
                label=f'Equilibration ({equilibration_sweeps} sweeps)')
    ax2.set_xlabel('Monte Carlo Sweep')
    ax2.set_ylabel('Total Magnetization')
    ax2.set_title('Magnetization Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_histograms(equilibrated_energies: np.ndarray, 
                   equilibrated_magnetizations: np.ndarray,
                   figsize: Tuple[int, int] = (12, 5),
                   save_path: Optional[str] = None):
    """
    Plot histograms of equilibrated energy and magnetization values
    
    Parameters:
    -----------
    equilibrated_energies : np.ndarray
        Energy values after equilibration
    equilibrated_magnetizations : np.ndarray
        Magnetization values after equilibration
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure (if provided)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Energy histogram
    ax1.hist(equilibrated_energies, bins=30, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(equilibrated_energies), color='red', linestyle='--', 
                linewidth=2, label=f'Mean = {np.mean(equilibrated_energies):.1f}')
    ax1.set_xlabel('Energy')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Energy Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Magnetization histogram
    ax2.hist(equilibrated_magnetizations, bins=30, density=True, alpha=0.7, 
             color='lightgreen', edgecolor='black')
    ax2.axvline(np.mean(equilibrated_magnetizations), color='red', linestyle='--', 
                linewidth=2, label=f'Mean = {np.mean(equilibrated_magnetizations):.1f}')
    ax2.set_xlabel('Magnetization')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Magnetization Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_autocorrelation(data: np.ndarray, max_lag: int = 100, 
                        variable_name: str = "Observable",
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None):
    """
    Plot autocorrelation function to assess correlation time
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    max_lag : int
        Maximum lag to compute autocorrelation
    variable_name : str
        Name of the variable for labeling
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure (if provided)
    """
    def autocorrelation(x, max_lag):
        """Calculate autocorrelation function"""
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]  # Normalize
        return autocorr[:max_lag+1]
    
    lags = np.arange(max_lag + 1)
    autocorr = autocorrelation(data, max_lag)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(lags, autocorr, 'o-', markersize=3, linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.7, 
               label=r'$1/e \approx 0.368$')
    
    # Find correlation time (when autocorr drops to 1/e)
    try:
        tau_idx = np.where(autocorr <= 1/np.e)[0][0]
        ax.axvline(x=tau_idx, color='red', linestyle='--', alpha=0.7,
                   label=f'τ ≈ {tau_idx}')
    except IndexError:
        pass
    
    ax.set_xlabel('Lag (Monte Carlo Sweeps)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation Function - {variable_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_temperature_sweep_results(results: Dict, figsize: Tuple[int, int] = (15, 10),
                                  save_path: Optional[str] = None):
    """
    Plot results from temperature sweep analysis
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from temperature_sweep function
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure (if provided)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    temperatures = results['temperatures']
    
    # Energy per spin
    ax1.plot(temperatures, results['energies'], 'bo-', markersize=5, linewidth=2)
    ax1.axvline(x=2.269, color='red', linestyle='--', alpha=0.7, 
                label='Theoretical $T_c = 2.269$')
    ax1.set_xlabel('Temperature (J/k_B)')
    ax1.set_ylabel('Energy per spin')
    ax1.set_title('Average Energy vs Temperature')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Magnetization per spin
    ax2.plot(temperatures, results['magnetizations'], 'go-', markersize=5, linewidth=2)
    ax2.axvline(x=2.269, color='red', linestyle='--', alpha=0.7, 
                label='Theoretical $T_c = 2.269$')
    ax2.set_xlabel('Temperature (J/k_B)')
    ax2.set_ylabel('|Magnetization| per spin')
    ax2.set_title('Average |Magnetization| vs Temperature')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Specific heat
    ax3.plot(temperatures, results['specific_heats'], 'ro-', markersize=5, linewidth=2)
    ax3.axvline(x=2.269, color='red', linestyle='--', alpha=0.7, 
                label='Theoretical $T_c = 2.269$')
    ax3.set_xlabel('Temperature (J/k_B)')
    ax3.set_ylabel('Specific Heat')
    ax3.set_title('Specific Heat vs Temperature')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Magnetic susceptibility
    ax4.plot(temperatures, results['susceptibilities'], 'mo-', markersize=5, linewidth=2)
    ax4.axvline(x=2.269, color='red', linestyle='--', alpha=0.7, 
                label='Theoretical $T_c = 2.269$')
    ax4.set_xlabel('Temperature (J/k_B)')
    ax4.set_ylabel('Magnetic Susceptibility')
    ax4.set_title('Magnetic Susceptibility vs Temperature')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('Phase Transition Analysis - 2D Ising Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_animation(model: IsingModel, n_frames: int = 100, interval: int = 100,
                    figsize: Tuple[int, int] = (8, 8), save_path: Optional[str] = None):
    """
    Create an animation showing the evolution of the spin configuration
    
    Parameters:
    -----------
    model : IsingModel
        The Ising model instance
    n_frames : int
        Number of frames in the animation
    interval : int
        Interval between frames in milliseconds
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the animation (if provided, should end with .gif or .mp4)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap
    colors = ['#3498db', '#e74c3c']  # Blue, Red
    cmap = ListedColormap(colors)
    
    # Initialize plot
    im = ax.imshow(model.spins, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Ising Model Evolution (T = {model.temperature:.2f})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 1], shrink=0.8)
    cbar.set_ticklabels(['↓ (-1)', '↑ (+1)'])
    cbar.set_label('Spin Direction', fontsize=12)
    
    def animate(frame):
        # Perform several Monte Carlo sweeps between frames
        for _ in range(5):
            model.monte_carlo_sweep()
        
        # Update image
        im.set_array(model.spins)
        
        # Update title with current statistics
        energy = model.calculate_energy()
        magnetization = model.calculate_magnetization()
        ax.set_title(f'Ising Model Evolution (T = {model.temperature:.2f})\n'
                    f'Frame {frame+1}, E = {energy:.0f}, M = {magnetization:.0f}',
                    fontsize=12)
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                  interval=interval, blit=True, repeat=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        else:
            print("Warning: save_path should end with .gif or .mp4")
    
    plt.tight_layout()
    plt.show()
    
    return anim


def plot_comparison_temperatures(temperatures: List[float], size: int = 32, 
                               n_sweeps: int = 500, figsize: Tuple[int, int] = (15, 5),
                               save_path: Optional[str] = None):
    """
    Plot spin configurations at different temperatures for comparison
    
    Parameters:
    -----------
    temperatures : List[float]
        List of temperatures to compare
    size : int
        Lattice size
    n_sweeps : int
        Number of sweeps to run for each temperature
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure (if provided)
    """
    n_temps = len(temperatures)
    fig, axes = plt.subplots(1, n_temps, figsize=figsize)
    
    if n_temps == 1:
        axes = [axes]
    
    # Create custom colormap
    colors = ['#3498db', '#e74c3c']  # Blue, Red
    cmap = ListedColormap(colors)
    
    for i, T in enumerate(temperatures):
        model = IsingModel(size=size, temperature=T)
        
        # Run simulation
        model.run_simulation(n_sweeps=n_sweeps, equilibration_sweeps=n_sweeps//5, 
                           verbose=False)
        
        # Plot final configuration
        im = axes[i].imshow(model.spins, cmap=cmap, vmin=-1, vmax=1, 
                           interpolation='nearest')
        axes[i].set_title(f'T = {T:.2f}', fontsize=12, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Calculate and display magnetization
        mag_per_spin = np.abs(model.calculate_magnetization()) / size**2
        axes[i].text(0.05, 0.95, f'|M| = {mag_per_spin:.3f}', 
                    transform=axes[i].transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add a colorbar to the last subplot
    cbar = plt.colorbar(im, ax=axes[-1], ticks=[-1, 1], shrink=0.8)
    cbar.set_ticklabels(['↓ (-1)', '↑ (+1)'])
    
    plt.suptitle(f'Spin Configurations at Different Temperatures ({size}×{size} lattice)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Quick demonstration of visualization functions
    print("MCMC Ising Model Visualization Demo")
    print("=" * 40)
    
    # Create a model and run simulation
    model = IsingModel(size=32, temperature=2.5)
    results = model.run_simulation(n_sweeps=300, equilibration_sweeps=50, verbose=False)
    
    # Demonstrate some visualizations
    print("Plotting spin configuration...")
    plot_spin_configuration(results['final_spins'], "Final Spin Configuration")
    
    print("Plotting energy and magnetization evolution...")
    plot_energy_magnetization_evolution(results['energy_history'], 
                                      results['magnetization_history'], 
                                      equilibration_sweeps=50)