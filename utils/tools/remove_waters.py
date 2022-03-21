import ccdc.molecule

def get_largest_components(m):
    s = []
    for c in m.components:
        n = len(c.atoms)
        id_n = int(str(c.identifier))
        l = [(n, id_n)]
        s.append(l)
    t = sorted(s, key=lambda k: k[0])
    largest_id = t[-1][0][1] - 1

    return largest_id

def remove_waters(m):
    keep = []
    waters = 0
    for s in m.components:
        ats = [at.atomic_symbol for at in s.atoms]
        if len(ats) == 3:
            ats.sort()
            if ats[0] == 'H' and ats[1] == 'H' and ats[2] == 'O':
                waters += 1
            else:
                keep.append(s)
        else:
            keep.append(s)
    new = ccdc.molecule.Molecule(m.identifier)
    for k in keep:
        new.add_molecule(k)
    return new

def remove_single_oxygen(m):
    keep = []
    waters = 0
    for s in m.components:
        ats = [at.atomic_symbol for at in s.atoms]
        if len(ats) == 1:
            ats.sort()
            if ats[0] == 'O':
                waters += 1
            else:
                keep.append(s)
        else:
            keep.append(s)
    new = ccdc.molecule.Molecule(m.identifier)
    for k in keep:
        new.add_molecule(k)
    return new
