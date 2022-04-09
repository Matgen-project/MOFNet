import numpy as np

def get_atom_features(atom):
    attributes = []
    attributes += one_hot_vector(
        atom.atomic_number,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, \
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 19, 30, \
         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
         45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, \
         59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, \
         73, 74, 75, 76, 77, 78, 79, 80, 81, 999]
    )
    # Connected numbers
    attributes += one_hot_vector(
        len(atom.neighbours),
        [0, 1, 2, 3, 4, 5, 6, 999]
    )

    # Test whether or not the atom is a hydrogen bond acceptor
    attributes.append(atom.is_acceptor)
    attributes.append(atom.is_chiral)

    # Test whether the atom is part of a ring system.
    attributes.append(atom.is_cyclic)
    attributes.append(atom.is_metal)

    # Test Whether this is a spiro atom.
    attributes.append(atom.is_spiro)

    return np.array(list(attributes), dtype=np.float32)

def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)
