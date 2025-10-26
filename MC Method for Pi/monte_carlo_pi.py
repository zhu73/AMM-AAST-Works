#!/usr/bin/env python3
"""
Estimate the value of Pi using Monte Carlo method.

The Monte Carlo method works by:
1. Generating random points in a unit square (0,0) to (1,1)
2. Counting how many points fall inside a quarter circle with radius 1
3. Using the ratio: π/4 ≈ (points inside circle) / (total points)
4. Therefore: π ≈ 4 * (points inside circle) / (total points)
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np

def estimate_pi_monte_carlo(n_samples):
    """
    Estimate Pi using Monte Carlo method.
    
    Args:
        n_samples (int): Number of random samples to generate
        
    Returns:
        float: Estimated value of Pi
        list: List of points inside the circle
        list: List of points outside the circle
    """
    points_inside = 0
    inside_points = []
    outside_points = []
    
    for _ in range(n_samples):
        # Generate random point in unit square [0,1] x [0,1]
        x = random.random() # Random number between 0 and 1
        y = random.random() # Random number between 0 and 1

        # Check if point is inside quarter circle (x² + y² ≤ 1)
        if x*x + y*y <= 1:
            points_inside += 1
            inside_points.append((x, y))
        else:
            outside_points.append((x, y))
    
    # Estimate Pi: π/4 = points_inside/n_samples, so π = 4 * points_inside/n_samples
    pi_estimate = 4 * points_inside / n_samples
    
    return pi_estimate, inside_points, outside_points

def plot_monte_carlo(inside_points, outside_points, pi_estimate, n_samples):
    """
    Visualize the Monte Carlo simulation.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot points inside circle as red dots
    if inside_points:
        inside_x, inside_y = zip(*inside_points)
        plt.scatter(inside_x, inside_y, c='red', s=1, alpha=0.6, label='Inside circle')
    
    # Plot points outside circle as blue dots
    if outside_points:
        outside_x, outside_y = zip(*outside_points)
        plt.scatter(outside_x, outside_y, c='blue', s=1, alpha=0.6, label='Outside circle')
    
    # Draw the quarter circle boundary
    theta = np.linspace(0, np.pi/2, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    plt.plot(circle_x, circle_y, 'black', linewidth=2, label='Quarter circle')
    
    # Draw the unit square
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'black', linewidth=2)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal')
    plt.title(f'Monte Carlo Estimation of π\n'
              f'Samples: {n_samples:,}, Estimated π: {pi_estimate:.6f}, '
              f'Actual π: {math.pi:.6f}, Error: {abs(pi_estimate - math.pi):.6f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure to the folder
    filename = f'monte_carlo_pi_{n_samples}_samples.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {filename}")
    
    plt.close()  # Close the figure instead of showing it

def main():
    """
    Main function to demonstrate Monte Carlo estimation of π.
    """
    print("Monte Carlo Estimation of π")
    print("=" * 40)
    
    # Three different sample sizes to test
    sample_sizes = [10000, 100000, 1000000]
    
    for n_samples in sample_sizes:
        print(f"\nUsing {n_samples:,} samples:")
        pi_estimate, inside_points, outside_points = estimate_pi_monte_carlo(n_samples)
        
        error = abs(pi_estimate - math.pi)
        percent_error = (error / math.pi) * 100
        
        print(f"Estimated π: {pi_estimate:.6f}")
        print(f"Actual π:    {math.pi:.6f}")
        print(f"Error:       {error:.6f} ({percent_error:.2f}%)")
        
        # Plot for all sample sizes
        plot_monte_carlo(inside_points, outside_points, pi_estimate, n_samples)
        