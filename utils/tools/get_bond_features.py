import numpy as np
import math

def get_bond_features(mol):
    """Calculate bond features.

    Args:
        mol (ccdc.molecule.bond): An CSD mol object.

    Returns:
        bond matriax.
        bond distance.
    """
    adj_matrix = np.eye(len(mol.atoms))
    dis_matrix = []

    for bond in mol.bonds:
        atom1,atom2 = bond.atoms
        # construct atom matrix.
        adj_matrix[atom1.index, atom2.index] = adj_matrix[atom2.index, atom1.index] = 1

        # calculate bond distance.
        #print(atom1,atom2)
        #a_array = [atom1.coordinates.x, atom1.coordinates.y, atom1.coordinates.z]
        #b_array = [atom2.coordinates.x, atom2.coordinates.y, atom2.coordinates.z]
        #bond_length = calc_distance(a_array, b_array)
        #dis_matrix.append(bond_length)
    
    return adj_matrix

def get_bond_features_en(mol):
    """Calculate bond features.

    Args:
        mol (ccdc.molecule.bond): An CSD mol object.

    Returns:
        bond matriax (coo).
    """
    row, col = [], []

    for bond in mol.bonds:
        atom1,atom2 = bond.atoms
        # construct atom matrix.
        row.append(atom1.index)
        col.append(atom2.index)
        row.append(atom2.index)
        col.append(atom1.index)
    
    return row, col
    
# function to obtain bond distance
def calc_distance(a_array, b_array):
    delt_d = np.array(a_array) -  np.array(b_array)
    distance  = math.sqrt(delt_d[0]**2 + delt_d[1]**2 + delt_d[2]**2)
    return round(distance,3)
    

