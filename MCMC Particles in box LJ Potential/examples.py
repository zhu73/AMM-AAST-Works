"""
Example demonstrations of MCMC simulation with Lennard-Jones potential.

This script runs several example simulations with different parameters
to demonstrate the capabilities of the MCMC LJ simulation.
"""

import numpy as np
import mcmc_lj_simulation as mcmc
import visualize_simulation as viz
import matplotlib.pyplot as plt


def example_1_basic_simulation():
    """
    Example 1: Basic simulation with moderate density.
    """
    print("="*70)
    print("EXAMPLE 1: Basic MCMC Simulation")
    print("="*70)
    print("This demonstrates a basic simulation with 20 particles\n")
    
    np.random.seed(42)
    
    params = mcmc.SimulationParameters(
        n_particles=20,
        box_length=10.0,
        temperature=1.0,
        epsilon=1.0,
        sigma=1.0,
        max_displacement=0.3,
        n_steps=10000,
        equilibration_steps=2000
    )
    
    system = mcmc.LennardJonesSystem(params)
    results = system.run_simulation(verbose=True)
    
    # Visualize results
    print("\nCreating visualizations...")
    viz.plot_energy_evolution(results['energies'])
    viz.plot_particle_configuration_3d(results['final_positions'], params.box_length)
    
    return system, results


def example_2_temperature_comparison():
    """
    Example 2: Compare systems at different temperatures.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Temperature Comparison")
    print("="*70)
    print("Comparing systems at T=0.5, T=1.0, and T=2.0\n")
    
    temperatures = [0.5, 1.0, 2.0]
    colors = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    results_list = []
    
    for T, color in zip(temperatures, colors):
        print(f"\nRunning simulation at T = {T}...")
        np.random.seed(42)  # Same initial configuration
        
        params = mcmc.SimulationParameters(
            n_particles=20,
            box_length=10.0,
            temperature=T,
            max_displacement=0.2,
            n_steps=8000,
            equilibration_steps=2000
        )
        
        system = mcmc.LennardJonesSystem(params)
        results = system.run_simulation(verbose=False)
        results_list.append((T, system, results))
        
        # Plot energy evolution
        ax = axes[0, 0]
        steps = np.arange(len(results['energies']))
        ax.plot(steps, results['energies'], alpha=0.7, linewidth=0.8, 
               color=color, label=f'T = {T}')
        
        # Plot energy distribution
        ax = axes[0, 1]
        ax.hist(results['energies'], bins=30, alpha=0.5, 
               label=f'T = {T}', color=color, density=True)
        
        # Plot radial distribution function
        ax = axes[1, 0]
        r, g = system.get_radial_distribution_function()
        ax.plot(r, g, linewidth=2, color=color, label=f'T = {T}')
        
        print(f"T = {T}: Mean E = {results['mean_energy']:.2f}, "
              f"Std E = {results['std_energy']:.2f}, "
              f"Acceptance = {results['acceptance_rate']:.3f}")
    
    # Format plots
    axes[0, 0].set_xlabel('MCMC Step')
    axes[0, 0].set_ylabel('Total Energy')
    axes[0, 0].set_title('Energy Evolution at Different Temperatures', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Energy')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title('Energy Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Distance r')
    axes[1, 0].set_ylabel('g(r)')
    axes[1, 0].set_title('Radial Distribution Function', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    
    # Summary table
    ax = axes[1, 1]
    table_data = []
    for T, system, results in results_list:
        table_data.append([
            f"{T:.1f}",
            f"{results['mean_energy']:.2f}",
            f"{results['std_energy']:.2f}",
            f"{results['acceptance_rate']:.3f}"
        ])
    
    table_text = "Temperature Comparison Summary\n" + "─"*45 + "\n"
    table_text += f"{'T':<6} {'Mean E':<12} {'Std E':<12} {'Accept':<10}\n"
    table_text += "─"*45 + "\n"
    for row in table_data:
        table_text += f"{row[0]:<6} {row[1]:<12} {row[2]:<12} {row[3]:<10}\n"
    
    ax.text(0.1, 0.5, table_text, fontsize=11, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results_list


def example_3_density_effects():
    """
    Example 3: Effect of particle density on the system.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Density Effects")
    print("="*70)
    print("Comparing low, medium, and high density systems\n")
    
    # Keep temperature and total volume constant, vary number of particles
    densities = [
        (10, 10.0, "Low"),    # 10 particles, box_length = 10
        (30, 10.0, "Medium"), # 30 particles, box_length = 10
        (60, 10.0, "High")    # 60 particles, box_length = 10
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (n_particles, box_length, density_label) in enumerate(densities):
        rho = n_particles / (box_length ** 3)
        print(f"\n{density_label} density: {n_particles} particles, ρ = {rho:.4f}")
        
        np.random.seed(42)
        
        params = mcmc.SimulationParameters(
            n_particles=n_particles,
            box_length=box_length,
            temperature=1.0,
            max_displacement=0.25,
            n_steps=5000,
            equilibration_steps=1000
        )
        
        system = mcmc.LennardJonesSystem(params)
        results = system.run_simulation(verbose=False)
        
        print(f"Mean E/N = {results['mean_energy']/n_particles:.3f}, "
              f"Acceptance = {results['acceptance_rate']:.3f}")
        
        # Plot radial distribution function
        r, g = system.get_radial_distribution_function()
        axes[idx].plot(r, g, linewidth=2, color='navy')
        axes[idx].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[idx].set_xlabel('Distance r', fontsize=11)
        axes[idx].set_ylabel('g(r)', fontsize=11)
        axes[idx].set_title(f'{density_label} Density\n(N={n_particles}, ρ={rho:.4f})',
                          fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, box_length/2])
    
    plt.tight_layout()
    plt.show()


def example_4_convergence_analysis():
    """
    Example 4: Analyze convergence and equilibration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Convergence Analysis")
    print("="*70)
    print("Analyzing how quickly the system equilibrates\n")
    
    np.random.seed(42)
    
    params = mcmc.SimulationParameters(
        n_particles=25,
        box_length=10.0,
        temperature=1.0,
        max_displacement=0.3,
        n_steps=15000,
        equilibration_steps=0  # We'll analyze this ourselves
    )
    
    system = mcmc.LennardJonesSystem(params)
    
    # Track energy at every step (even during "equilibration")
    energies = []
    print("Running extended simulation...")
    for step in range(params.n_steps):
        system.metropolis_step()
        current_energy = system.calculate_total_energy()
        energies.append(current_energy)
        
        if (step + 1) % 3000 == 0:
            print(f"Step {step + 1}/{params.n_steps}")
    
    energies = np.array(energies)
    
    # Calculate running average and variance
    window_size = 500
    running_avg = np.convolve(energies, np.ones(window_size)/window_size, mode='valid')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Full energy trace
    steps = np.arange(len(energies))
    axes[0].plot(steps, energies, alpha=0.5, linewidth=0.5, color='blue')
    axes[0].plot(steps[window_size-1:], running_avg, 'r-', linewidth=2, 
                label=f'Running average (window={window_size})')
    axes[0].set_xlabel('MCMC Step')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Energy Evolution with Running Average', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Running variance
    running_var = []
    for i in range(window_size, len(energies)):
        chunk = energies[i-window_size:i]
        running_var.append(np.var(chunk))
    
    axes[1].plot(steps[window_size:], running_var, linewidth=2, color='green')
    axes[1].set_xlabel('MCMC Step')
    axes[1].set_ylabel('Variance')
    axes[1].set_title(f'Running Variance (window={window_size})', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Block averaging for error estimation
    block_sizes = [10, 50, 100, 500, 1000]
    block_means = []
    block_errors = []
    
    for block_size in block_sizes:
        n_blocks = len(energies) // block_size
        blocks = energies[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_avgs = np.mean(blocks, axis=1)
        block_means.append(np.mean(block_avgs))
        block_errors.append(np.std(block_avgs) / np.sqrt(n_blocks))
    
    axes[2].errorbar(block_sizes, block_means, yerr=block_errors, 
                    marker='o', markersize=8, capsize=5, linewidth=2)
    axes[2].set_xlabel('Block Size')
    axes[2].set_ylabel('Mean Energy')
    axes[2].set_title('Block Averaging Analysis', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal statistics:")
    print(f"Overall mean energy: {np.mean(energies):.4f}")
    print(f"Last 5000 steps mean: {np.mean(energies[-5000:]):.4f}")
    print(f"Standard error (last 5000): {np.std(energies[-5000:]) / np.sqrt(5000):.4f}")


def run_all_examples():
    """Run all example demonstrations."""
    print("\n" + "="*70)
    print("MCMC Lennard-Jones Simulation - Example Demonstrations")
    print("="*70 + "\n")
    
    # Example 1
    example_1_basic_simulation()
    
    # Example 2
    example_2_temperature_comparison()
    
    # Example 3
    example_3_density_effects()
    
    # Example 4
    example_4_convergence_analysis()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    # Run individual examples or all at once
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_simulation()
        elif example_num == "2":
            example_2_temperature_comparison()
        elif example_num == "3":
            example_3_density_effects()
        elif example_num == "4":
            example_4_convergence_analysis()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python examples.py [1|2|3|4]")
    else:
        # Run all examples
        run_all_examples()
