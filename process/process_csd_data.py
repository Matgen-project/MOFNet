from ccdc.descriptors import MolecularDescriptors as MD, GeometricDescriptors as GD
from ccdc.io import EntryReader
csd = EntryReader('CSD')
import ccdc.molecule
import sys
import os
import numpy as np
import math

import pickle 
from tools.get_atom_features import get_atom_features
from tools.get_bond_features import get_bond_features
from tools.remove_waters import remove_waters, remove_single_oxygen, get_largest_components
import numpy as np
from sklearn.metrics import pairwise_distances

mol_name = sys.argv[1]
mol = csd.molecule(mol_name)


# remove waters
mol = remove_waters(mol)
mol = remove_single_oxygen(mol)

# remove other solvates, here we remove all small components.

if len(mol.components) > 1:
    lg_id = get_largest_components(mol)
    mol = mol.components[lg_id]

mol.remove_hydrogens()

atom_features = np.array([get_atom_features(atom) for atom in mol.atoms])
bond_matrix = get_bond_features(mol)

pos_matrix = np.array([[atom.coordinates.x, atom.coordinates.y, atom.coordinates.z] for atom in mol.atoms])
dist_matrix = pairwise_distances(pos_matrix)

mol_features = [atom_features, bond_matrix, dist_matrix]

save_path = '../data/processed/' + mol_name + '.p'

if not os.path.exists(save_path):
    pickle.dump(mol_features,open(save_path, "wb"))
 
   
