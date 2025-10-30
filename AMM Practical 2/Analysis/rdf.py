import numpy as np
import mdtraj as mdt

top = mdt.load('../init.pdb')
unitcell_lengths = top.unitcell_lengths
unitcell_angles = top.unitcell_angles

# Load the trajectory (using cleaned files)
traj_ff = mdt.load('../01-FF-Water/H2O-64-FF-clean.xyz', top=top)
traj_mlp = mdt.load('../02-MLP-Water/H2O-64-NNP-clean.xyz', top=top)
# inject the cell information
len_trj = len(traj_ff)
traj_ff.unitcell_lengths = unitcell_lengths.repeat(len_trj, axis=0)
traj_ff.unitcell_angles = unitcell_angles.repeat(len_trj, axis=0)

len_trj = len(traj_mlp)
traj_mlp.unitcell_lengths = unitcell_lengths.repeat(len_trj, axis=0)
traj_mlp.unitcell_angles = unitcell_angles.repeat(len_trj, axis=0)

# Compute the RDF
min_dimension = traj_ff[0].unitcell_lengths.min() / 2
pairs = traj_ff.topology.select_pairs('name O', 'name O')
r, rdf_ff = mdt.compute_rdf(traj_ff, pairs=pairs, r_range=(0, min_dimension), bin_width=0.005)
pairs = traj_mlp.topology.select_pairs('name O', 'name O')
r, rdf_mlp = mdt.compute_rdf(traj_mlp, pairs=pairs, r_range=(0, min_dimension), bin_width=0.005)

# Save the RDF
np.savetxt('rdf-ff.txt', np.vstack((r*10, rdf_ff)).T)
np.savetxt('rdf-mlp.txt', np.vstack((r*10, rdf_mlp)).T)
