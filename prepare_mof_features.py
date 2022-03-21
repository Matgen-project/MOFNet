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


#print(len(mol.atoms))
# remove waters
#print(len(mol.components))
mol = remove_waters(mol)
#print(len(mol.components))
mol = remove_single_oxygen(mol)
#print(len(mol.components))

#print(mol.components[1]) 
# remove other solvates, here we remove all small components.

if len(mol.components) > 1:
#    compounts = sorted([(x.molecular_weight, x) for x in mol.components])
    #compounts = sorted((x, key=lambda m: x.molecular_weight) for x in mol.components)
    lg_id = get_largest_components(mol)
    mol = mol.components[lg_id]
#print(len(mol.atoms))
#print(len(mol.components))

mol.remove_hydrogens()
#print(len(mol.atoms))

atom_features = get_atom_features(mol)
bond_features = get_bond_features(mol)

mol_features = [atom_features, bond_features]

save_path = './processed/' + mol_name + '.p'

if not os.path.exists(save_path):
    pickle.dump(mol_features,open(save_path, "wb"))
 
   
