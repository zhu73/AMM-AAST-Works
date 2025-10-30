#!/usr/bin/env python3
"""
View trajectories using ASE GUI
Usage: python3 view_ase.py <trajectory_file.xyz>
"""
import sys
from ase.io import read
from ase.visualize import view

if len(sys.argv) < 2:
    print("Usage: python3 view_ase.py <trajectory_file.xyz>")
    print("\nExamples:")
    print("  python3 view_ase.py 01-FF-Water/H2O-64-FF-clean.xyz")
    print("  python3 view_ase.py 02-MLP-Water/H2O-64-NNP-clean.xyz")
    sys.exit(1)

trajectory_file = sys.argv[1]
print(f"Loading trajectory from {trajectory_file}...")

# Read all frames from the trajectory
atoms = read(trajectory_file, index=':')
print(f"Loaded {len(atoms)} frames")

# Launch ASE GUI
print("Launching ASE GUI...")
view(atoms)
