#!/usr/bin/env python3
"""
Create PNG images from trajectory frames (no display needed)
Usage: python3 create_images.py <trajectory_file.xyz> <output_dir> [frame_step]
"""
import sys
import os
from ase.io import read, write

if len(sys.argv) < 3:
    print("Usage: python3 create_images.py <trajectory_file.xyz> <output_dir> [frame_step]")
    print("\nExamples:")
    print("  python3 create_images.py 01-FF-Water/H2O-64-FF-clean.xyz images_ff 10")
    print("  python3 create_images.py 02-MLP-Water/H2O-64-NNP-clean.xyz images_mlp 10")
    sys.exit(1)

trajectory_file = sys.argv[1]
output_dir = sys.argv[2]
frame_step = int(sys.argv[3]) if len(sys.argv) > 3 else 10

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print(f"Loading trajectory from {trajectory_file}...")
atoms = read(trajectory_file, index=':')
print(f"Loaded {len(atoms)} frames")

# Save selected frames as images
print(f"Saving every {frame_step}th frame to {output_dir}/")
for i, frame in enumerate(atoms[::frame_step]):
    output_file = f"{output_dir}/frame_{i*frame_step:04d}.png"
    write(output_file, frame, rotation='10z,-80x')
    if i % 10 == 0:
        print(f"  Saved frame {i*frame_step}")

print(f"\nDone! Created {len(atoms[::frame_step])} images in {output_dir}/")
print(f"You can download these images or create a movie using ffmpeg")
