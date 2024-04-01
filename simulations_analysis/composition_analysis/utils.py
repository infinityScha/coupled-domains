import os
import re

import MDAnalysis as md
import numpy as np
from MDAnalysis import Merge
from tqdm import tqdm

LIPIDS_DEF = {}
# read lipid identification folder
with open("simulations_analysis/composition_analysis/lipids.txt") as f:
    lines = f.readlines()
    n = 0
    while n < len(lines):
        name = lines[n].rstrip()
        head = lines[n + 1].rstrip()
        tail1 = lines[n + 2].rstrip()
        tail2 = lines[n + 3].rstrip()
        LIPIDS_DEF[name] = [head, tail1, tail2]
        n += 5


class UniverseLoader(object):

    universe = None
    _normal = None

    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, value):
        self._normal = value

    def __init__(self, grofile, xtcfile, normal=2, by_tails=True):
        self._load_universe(grofile, xtcfile, by_tails)
        self._normal = normal

    def _load_universe(self, grofile, xtcfile, by_tails=True):
        self.universe = md.Universe(grofile, xtcfile)
        if by_tails:
            grofile_w_tails = grofile[:-4] + '_by_tails.gro'
            grofile_w_tails_sorted = grofile_w_tails[:-4] + '_sorted.gro'
            if not os.path.exists(grofile_w_tails_sorted):
                if not os.path.exists(grofile_w_tails):
                    # otherwise make the desired universe
                    self._set_tail_residues()
                    self.universe.atoms.write(grofile_w_tails, reindex=False)
                # reorder the grofile by resids for correct setting of resids
                with open(grofile_w_tails, 'r') as f:
                    grofile_content = f.readlines()
                resids = np.array([int(re.search('[0-9]+', line).group()) for line in grofile_content[2:-1]])
                # sort by resid
                linesort = np.argsort(resids, kind='stable').flatten().astype(int)
                temp = grofile_content.copy()
                for i, j in enumerate(linesort):
                    temp[i+2] = grofile_content[2:-1][j]
                grofile_content = temp
                # write the sorted one to file
                with open(grofile_w_tails_sorted, 'w') as f:
                    f.write("".join(grofile_content))

            self.universe = md.Universe(grofile_w_tails_sorted)
            # reorder the atoms in the universe by ids
            self.universe = Merge(self.universe.atoms[np.argsort(self.universe.atoms.ids)])
            self.universe.load_new(xtcfile)

    # change tails residue function
    def _set_tail_residues(self):
        selection = '(resname CHL1 and name O*)'
        for name, value in LIPIDS_DEF.items():
            selection = selection + ' or (resname ' + name + ' and name ' + value[0] + ')'
        heads = self.universe.select_atoms(selection)
        membrane = heads.residues.atoms
        membrane_not_sterol = membrane.select_atoms('not resname CHL1').residues

        # select the tails seperately
        tail1_sel = ''
        tail2_sel = ''
        for name, value in LIPIDS_DEF.items():
            tail1_sel = tail1_sel + '(resname ' + name + ' and name ' + value[1] + ') or'
            tail2_sel = tail2_sel + '(resname ' + name + ' and name ' + value[2] + ') or'
        tail1_sel = tail1_sel[:-2]
        tail2_sel = tail2_sel[:-2]

        max_resid = self.universe.residues.resids.max()

        for i, lipid in tqdm(enumerate(membrane_not_sterol), total=len(membrane_not_sterol), desc='reindexing residues'):
            # create tail residues
            res1 = self.universe.add_Residue(resname=lipid.resname + '1', segment=lipid.segment, resid=max_resid + 1,
                                        resnum=max_resid + 1)
            res2 = self.universe.add_Residue(resname=lipid.resname + '2', segment=lipid.segment, resid=max_resid + 2,
                                        resnum=max_resid + 2)
            max_resid += 2
            # get both tails
            tail1 = lipid.atoms.select_atoms(tail1_sel)
            tail2 = lipid.atoms.select_atoms(tail2_sel)
            # change tail's residues
            tail1.residues = res1
            tail2.residues = res2

    def set_bonds(self, tails_no_chol):
        # add bonds to ul.universe
        # get all carbons from lipid tails and the 3 atoms after them
        bonds = list()
        three_next_atoms = [self.universe.atoms[id:id + 3] for id in tails_no_chol.ids]
        for carbon, atoms in zip(tails_no_chol, three_next_atoms):
            # don't care for S_CD if connected to an oxygen.
            if ('O' in atoms[0].name) or ('O' in atoms[1].name) or ('O' in atoms[2].name):
                continue
            # add 1st and 2nd carbons.
            bonds.append(md.AtomGroup([carbon, atoms[0]]))
            bonds.append(md.AtomGroup([carbon, atoms[1]]))
            # if there isn't double bond add the third carbon as well
            if ('C' not in atoms[0].name) and ('C' not in atoms[1].name):
                bonds.append(md.AtomGroup([carbon, atoms[2]]))
        self.universe.add_bonds(bonds)

    def divide_to_leaflets(self, tails_res):
        classify = np.array(
            [np.sign(res.atoms[0].position[self.normal] - res.atoms[-1].position[self.normal]) for res in tails_res])
        classify = classify > 0.0
        return tails_res[classify], tails_res[np.invert(classify)], classify > 0.0


def load_input(input_file):
    d = {}
    with open(input_file) as f:
        for line in f:
            # ignore line
            if '#' == line[0] or '\n' == line[0]:
                continue
            splt = line.split()
            for i in range(1, len(splt)):
                if splt[i].isnumeric():
                    splt[i] = int(splt[i])
                elif splt[i].replace('.','',1).isdigit():
                    splt[i] = float(splt[i])
                # otherwise keep as a string
            key = splt[0]
            val = splt[1] if len(splt) == 2 else splt[1:]
            d[key] = val
    return d
