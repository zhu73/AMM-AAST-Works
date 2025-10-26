"""
Temperature analysis and phase transition studies for the MCMC Ising Model

This module provides advanced analysis tools for studying the phase transition
in the 2D Ising model, including critical temperature estimation and finite-size scaling.

Author: Created for MCMC Ising Model demonstration
Date: 2025-10-26
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ising_mcmc import IsingModel, temperature_sweep


def find_critical_temperature(size: int = 32, temperature_range: Tuple[float, float] = (2.0, 2.5),
                            n_points: int = 20, n_sweeps: int = 1000, 
                            equilibration_sweeps: int = 200) -> Dict:
    """
    Find the critical temperature by analyzing the peak in specific heat
    
    Parameters:
    -----------
    size : int
        Lattice size
    temperature_range : Tuple[float, float]
        Temperature range to search (min, max)
    n_points : int
        Number of temperature points to sample
    n_sweeps : int
        Number of Monte Carlo sweeps per temperature
    equilibration_sweeps : int
        Number of equilibration sweeps per temperature
        
    Returns:
    --------
    Dict containing critical temperature analysis results
    """
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n_points)
    
    print(f"Searching for critical temperature in range {temperature_range}")
    print(f"Using {n_points} temperature points for lattice size {size}×{size}")
    
    # Perform temperature sweep
    results = temperature_sweep(size=size, temperatures=temperatures, 
                              n_sweeps=n_sweeps, equilibration_sweeps=equilibration_sweeps)
    
    # Find temperature with maximum specific heat
    max_c_idx = np.argmax(results['specific_heats'])
    T_c_estimate = results['temperatures'][max_c_idx]
    max_specific_heat = results['specific_heats'][max_c_idx]
    
    # Find temperature with maximum susceptibility
    max_chi_idx = np.argmax(results['susceptibilities'])
    T_c_susceptibility = results['temperatures'][max_chi_idx]
    max_susceptibility = results['susceptibilities'][max_chi_idx]
    
    print(f"\nCritical temperature estimates:")
    print(f"From specific heat peak: T_c = {T_c_estimate:.4f} (C_max = {max_specific_heat:.4f})")
    print(f"From susceptibility peak: T_c = {T_c_susceptibility:.4f} (χ_max = {max_susceptibility:.4f})")
    print(f"Theoretical value (infinite lattice): T_c = 2.2691853...")
    
    return {
        'temperatures': temperatures,
        'specific_heats': results['specific_heats'],
        'susceptibilities': results['susceptibilities'],
        'energies': results['energies'],
        'magnetizations': results['magnetizations'],
        'T_c_specific_heat': T_c_estimate,
        'T_c_susceptibility': T_c_susceptibility,
        'max_specific_heat': max_specific_heat,
        'max_susceptibility': max_susceptibility,
        'size': size
    }


def finite_size_scaling_analysis(sizes: List[int] = [16, 24, 32, 48], 
                               temperature_range: Tuple[float, float] = (2.1, 2.4),
                               n_points: int = 15, n_sweeps: int = 1500,
                               equilibration_sweeps: int = 300) -> Dict:
    """
    Perform finite size scaling analysis to extrapolate critical temperature
    
    Parameters:
    -----------
    sizes : List[int]
        List of lattice sizes to analyze
    temperature_range : Tuple[float, float]
        Temperature range around critical point
    n_points : int
        Number of temperature points per size
    n_sweeps : int
        Number of Monte Carlo sweeps per temperature
    equilibration_sweeps : int
        Number of equilibration sweeps per temperature
        
    Returns:
    --------
    Dict containing finite size scaling results
    """
    print("Starting finite size scaling analysis...")
    print(f"Lattice sizes: {sizes}")
    print(f"Temperature range: {temperature_range}")
    
    results = {
        'sizes': sizes,
        'T_c_estimates': [],
        'max_specific_heats': [],
        'max_susceptibilities': [],
        'all_results': {}
    }
    
    for size in sizes:
        print(f"\nAnalyzing size {size}×{size}...")
        
        # Find critical temperature for this size
        size_results = find_critical_temperature(
            size=size, 
            temperature_range=temperature_range,
            n_points=n_points,
            n_sweeps=n_sweeps,
            equilibration_sweeps=equilibration_sweeps
        )
        
        results['T_c_estimates'].append(size_results['T_c_specific_heat'])
        results['max_specific_heats'].append(size_results['max_specific_heat'])
        results['max_susceptibilities'].append(size_results['max_susceptibility'])
        results['all_results'][size] = size_results
    
    # Convert to numpy arrays
    results['T_c_estimates'] = np.array(results['T_c_estimates'])
    results['max_specific_heats'] = np.array(results['max_specific_heats'])
    results['max_susceptibilities'] = np.array(results['max_susceptibilities'])
    
    # Finite size scaling: T_c(L) = T_c(∞) + a/L^(1/ν)
    # For 2D Ising: ν = 1, so T_c(L) = T_c(∞) + a/L
    # Fit linear relationship with 1/L
    inv_sizes = 1.0 / np.array(sizes)
    
    # Linear fit: T_c(L) = T_c_inf + a * (1/L)
    coeffs = np.polyfit(inv_sizes, results['T_c_estimates'], 1)
    T_c_infinite = coeffs[1]  # Intercept (T_c at L→∞)
    slope = coeffs[0]         # Slope (finite size correction)
    
    print(f"\nFinite size scaling analysis:")
    print(f"T_c(L) = {T_c_infinite:.6f} + {slope:.6f}/L")
    print(f"Extrapolated T_c(∞) = {T_c_infinite:.6f}")
    print(f"Theoretical T_c = 2.269185...")
    print(f"Error = {abs(T_c_infinite - 2.269185):.6f}")
    
    results['T_c_infinite'] = T_c_infinite
    results['finite_size_slope'] = slope
    results['inv_sizes'] = inv_sizes
    results['fit_coefficients'] = coeffs
    
    return results


def hysteresis_analysis(size: int = 32, temperature: float = 2.269, 
                       field_range: Tuple[float, float] = (-0.5, 0.5),
                       n_points: int = 50, n_sweeps: int = 500) -> Dict:
    """
    Analyze hysteresis behavior by sweeping external magnetic field
    
    Parameters:
    -----------
    size : int
        Lattice size
    temperature : float
        Temperature for the analysis
    field_range : Tuple[float, float]
        Range of external magnetic field (min, max)
    n_points : int
        Number of field points
    n_sweeps : int
        Number of sweeps per field value
        
    Returns:
    --------
    Dict containing hysteresis analysis results
    """
    print(f"Hysteresis analysis at T = {temperature}")
    print(f"Field range: {field_range}")
    
    fields = np.linspace(field_range[0], field_range[1], n_points)
    
    # Forward sweep (increasing field)
    forward_magnetizations = []
    model = IsingModel(size=size, temperature=temperature)
    
    print("Forward sweep (increasing field)...")
    for i, h in enumerate(fields):
        model.external_field = h
        model.run_simulation(n_sweeps=n_sweeps, equilibration_sweeps=n_sweeps//4, 
                           verbose=False)
        forward_magnetizations.append(model.calculate_magnetization() / size**2)
        if (i + 1) % (n_points // 10) == 0:
            print(f"  Progress: {i+1}/{n_points}")
    
    # Backward sweep (decreasing field)
    backward_magnetizations = []
    print("Backward sweep (decreasing field)...")
    for i, h in enumerate(reversed(fields)):
        model.external_field = h
        model.run_simulation(n_sweeps=n_sweeps, equilibration_sweeps=n_sweeps//4, 
                           verbose=False)
        backward_magnetizations.append(model.calculate_magnetization() / size**2)
        if (i + 1) % (n_points // 10) == 0:
            print(f"  Progress: {i+1}/{n_points}")
    
    backward_magnetizations = list(reversed(backward_magnetizations))
    
    return {
        'fields': fields,
        'forward_magnetizations': np.array(forward_magnetizations),
        'backward_magnetizations': np.array(backward_magnetizations),
        'temperature': temperature,
        'size': size
    }


def correlation_function_analysis(model: IsingModel, max_distance: int = None) -> Dict:
    """
    Calculate spin-spin correlation function
    
    Parameters:
    -----------
    model : IsingModel
        The Ising model instance with current configuration
    max_distance : int
        Maximum distance to calculate correlations (default: size//4)
        
    Returns:
    --------
    Dict containing correlation function results
    """
    if max_distance is None:
        max_distance = model.size // 4
    
    print(f"Calculating correlation function up to distance {max_distance}")
    
    # Calculate correlation function C(r) = <S_0 S_r> - <S_0><S_r>
    spins = model.spins
    size = model.size
    
    # Choose center point
    center_i, center_j = size // 2, size // 2
    center_spin = spins[center_i, center_j]
    
    distances = []
    correlations = []
    
    for r in range(1, max_distance + 1):
        correlation_sum = 0
        count = 0
        
        # Sample points at distance r from center
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di**2 + dj**2 <= r**2 and di**2 + dj**2 > (r-1)**2:
                    i = (center_i + di) % size
                    j = (center_j + dj) % size
                    correlation_sum += center_spin * spins[i, j]
                    count += 1
        
        if count > 0:
            distances.append(r)
            correlations.append(correlation_sum / count)
    
    # Fit exponential decay: C(r) = C_0 * exp(-r/ξ)
    # where ξ is the correlation length
    try:
        # Take log to linearize: ln(C(r)) = ln(C_0) - r/ξ
        positive_correlations = np.array(correlations)
        positive_correlations = positive_correlations[positive_correlations > 0]
        valid_distances = np.array(distances[:len(positive_correlations)])
        
        if len(positive_correlations) > 3:
            log_correlations = np.log(positive_correlations)
            coeffs = np.polyfit(valid_distances, log_correlations, 1)
            correlation_length = -1.0 / coeffs[0]
            C_0 = np.exp(coeffs[1])
        else:
            correlation_length = np.nan
            C_0 = np.nan
    except:
        correlation_length = np.nan
        C_0 = np.nan
    
    print(f"Estimated correlation length: ξ = {correlation_length:.2f}")
    
    return {
        'distances': np.array(distances),
        'correlations': np.array(correlations),
        'correlation_length': correlation_length,
        'C_0': C_0,
        'temperature': model.temperature
    }


def critical_exponents_analysis(finite_size_results: Dict) -> Dict:
    """
    Extract critical exponents from finite size scaling data
    
    Parameters:
    -----------
    finite_size_results : Dict
        Results from finite_size_scaling_analysis
        
    Returns:
    --------
    Dict containing critical exponent estimates
    """
    print("Analyzing critical exponents...")
    
    sizes = np.array(finite_size_results['sizes'])
    max_specific_heats = finite_size_results['max_specific_heats']
    max_susceptibilities = finite_size_results['max_susceptibilities']
    
    # Critical exponents for 2D Ising model (theoretical values):
    # α = 0 (specific heat logarithmic divergence)
    # γ = 7/4 = 1.75 (susceptibility)
    # ν = 1 (correlation length)
    
    # Finite size scaling:
    # C_max ∝ L^(α/ν) = L^0 = constant (for 2D Ising)
    # χ_max ∝ L^(γ/ν) = L^1.75
    
    # Fit susceptibility scaling: χ_max = A * L^(γ/ν)
    log_sizes = np.log(sizes)
    log_susceptibilities = np.log(max_susceptibilities)
    
    # Linear fit in log space
    coeffs_chi = np.polyfit(log_sizes, log_susceptibilities, 1)
    gamma_over_nu = coeffs_chi[0]
    
    # For 2D Ising, ν = 1, so γ ≈ gamma_over_nu
    gamma_estimate = gamma_over_nu
    
    print(f"Critical exponent estimates:")
    print(f"γ/ν = {gamma_over_nu:.3f} (from susceptibility scaling)")
    print(f"γ ≈ {gamma_estimate:.3f} (assuming ν = 1)")
    print(f"Theoretical γ = 1.75")
    print(f"Error in γ: {abs(gamma_estimate - 1.75):.3f}")
    
    return {
        'gamma_over_nu': gamma_over_nu,
        'gamma_estimate': gamma_estimate,
        'susceptibility_fit_coeffs': coeffs_chi,
        'log_sizes': log_sizes,
        'log_susceptibilities': log_susceptibilities
    }


if __name__ == "__main__":
    print("Temperature Analysis Demo")
    print("=" * 30)
    
    # Quick critical temperature analysis
    print("Finding critical temperature...")
    critical_results = find_critical_temperature(size=24, n_points=10, n_sweeps=300)
    
    print(f"\nEstimated T_c = {critical_results['T_c_specific_heat']:.4f}")
    print("Run the full analysis with larger parameters for better accuracy!")