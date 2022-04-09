from ccdc.descriptors import MolecularDescriptors as MD, GeometricDescriptors as GD
from ccdc.io import EntryReader
csd = EntryReader('CSD')
import ccdc.molecule
import sys
import os
import numpy as np
import math

import pickle 
from script.get_atom_features import get_atom_features
from script.get_bond_features import get_bond_features
from script.remove_waters import remove_waters, remove_single_oxygen, get_largest_components

mol_name = sys.argv[1]
mol = csd.molecule(mol_name)

mol = remove_waters(mol)
mol = remove_single_oxygen(mol)
if len(mol.components) > 1:
    lg_id = get_largest_components(mol)
    mol = mol.components[lg_id]

mol.remove_hydrogens()

atom_features = get_atom_features(mol)
bond_features = get_bond_features(mol)

mol_features = [atom_features, bond_features]

save_path = './processed/' + mol_name + '.p'

if not os.path.exists(save_path):
    pickle.dump(mol_features,open(save_path, "wb"))
 
   
