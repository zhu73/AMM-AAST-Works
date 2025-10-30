#!/usr/bin/env python3
"""
Compare Force Field vs Neural Network Potential Water Simulations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def read_ener_file(filename):
    """Read CP2K energy file - handle multiple entries per step"""
    step, time, kinetic, temperature, potential, conserved, cpu_time = [], [], [], [], [], [], []
    
    with open(filename, 'r', errors='ignore') as f:
        prev_step = -1
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    current_step = int(parts[0])
                    # Only take first entry for each step (avoid duplicates)
                    if current_step != prev_step:
                        step.append(current_step)
                        time.append(float(parts[1]))
                        kinetic.append(float(parts[2]))
                        temperature.append(float(parts[3]))
                        potential.append(float(parts[4]))
                        conserved.append(float(parts[5]))
                        cpu_time.append(float(parts[6]))
                        prev_step = current_step
                except (ValueError, IndexError):
                    continue
    
    return {
        'step': np.array(step),
        'time': np.array(time),
        'kinetic': np.array(kinetic),
        'temperature': np.array(temperature),
        'potential': np.array(potential),
        'conserved': np.array(conserved),
        'cpu_time': np.array(cpu_time)
    }

def calculate_stats(data):
    """Calculate statistics"""
    # Energy conservation (drift)
    E0 = data['conserved'][0]
    E_final = data['conserved'][-1]
    drift = (E_final - E0) / abs(E0) * 100
    
    # Temperature statistics
    temp_avg = np.mean(data['temperature'])
    temp_std = np.std(data['temperature'])
    
    return {
        'energy_drift': drift,
        'temp_avg': temp_avg,
        'temp_std': temp_std,
        'total_time': data['cpu_time'].sum()
    }

# Load data
print("Loading energy data...")
ff_data = read_ener_file('../01-FF-Water/H2O-64-FF-1.ener')
mlp_data = read_ener_file('../02-MLP-Water/H2O-64-NNP-1.ener')

# Calculate statistics
ff_stats = calculate_stats(ff_data)
mlp_stats = calculate_stats(mlp_data)

# Print comparison
print("\n" + "="*70)
print("SIMULATION COMPARISON: Force Field vs Neural Network Potential")
print("="*70)

print("\n{:<30} {:>15} {:>20}".format("Property", "Force Field", "Neural Network"))
print("-"*70)
print("{:<30} {:>15.4f}% {:>20.4f}%".format(
    "Energy Drift", ff_stats['energy_drift'], mlp_stats['energy_drift']))
print("{:<30} {:>15.2f} K {:>20.2f} K".format(
    "Average Temperature", ff_stats['temp_avg'], mlp_stats['temp_avg']))
print("{:<30} {:>15.2f} K {:>20.2f} K".format(
    "Temperature Std Dev", ff_stats['temp_std'], mlp_stats['temp_std']))
print("{:<30} {:>15.1f} s {:>20.1f} s".format(
    "Total CPU Time", ff_stats['total_time'], mlp_stats['total_time']))
print("{:<30} {:>15} {:>20}".format(
    "Steps", len(ff_data['step']), len(mlp_data['step'])))
print("{:<30} {:>15.1f} fs {:>20.1f} fs".format(
    "Simulation Time", ff_data['time'][-1], mlp_data['time'][-1]))

# Skip plotting
print("\n(Plotting disabled)")

# Save summary to file
with open('comparison_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("SIMULATION COMPARISON: Force Field vs Neural Network Potential\n")
    f.write("="*70 + "\n\n")
    f.write("{:<30} {:>15} {:>20}\n".format("Property", "Force Field", "Neural Network"))
    f.write("-"*70 + "\n")
    f.write("{:<30} {:>15.4f}% {:>20.4f}%\n".format(
        "Energy Drift", ff_stats['energy_drift'], mlp_stats['energy_drift']))
    f.write("{:<30} {:>15.2f} K {:>20.2f} K\n".format(
        "Average Temperature", ff_stats['temp_avg'], mlp_stats['temp_avg']))
    f.write("{:<30} {:>15.2f} K {:>20.2f} K\n".format(
        "Temperature Std Dev", ff_stats['temp_std'], mlp_stats['temp_std']))
    f.write("{:<30} {:>15.1f} s {:>20.1f} s\n".format(
        "Total CPU Time", ff_stats['total_time'], mlp_stats['total_time']))
    f.write("{:<30} {:>15} {:>20}\n".format(
        "Steps", len(ff_data['step']), len(mlp_data['step'])))
    f.write("{:<30} {:>15.1f} fs {:>20.1f} fs\n".format(
        "Simulation Time", ff_data['time'][-1], mlp_data['time'][-1]))

print("✅ Summary saved as 'comparison_summary.txt'")
print("\nAnalysis complete!")
