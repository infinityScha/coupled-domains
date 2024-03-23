import itertools

import shutil
import time

from sklearn import svm
import multiprocessing as mp

import os
import numpy as np
import matplotlib.pyplot as plt
import freud
from scipy import ndimage

from tqdm import tqdm
import utils

#####Constants########

# filenames and dirs
folder = '/run/media/itay/Elements/curved-domains/NEW2_FLATBOT/'
grofiles = ['DMPC_FORCEFLAT_075_nw.gro', 'DAPC_FORCEFLAT_075_nw.gro', 'DPPC_FORCEFLAT_075_nw.gro']
xtcfiles = ['DMPC_FORCEFLAT_075_whole.xtc', 'DAPC_FORCEFLAT_075_whole.xtc', 'DPPC_FORCEFLAT_075_whole.xtc']

#####Functions########

def positions():
    pos_list, res_list = list(), list()
    leaflet0, leaflet1 = ul.divide_to_leaflets(tails_res)[:2]
    box = ul.universe.dimensions[:3]
    for leaflet in [leaflet0, leaflet1]:
        leaflet_pos = leaflet.atoms.center(None, compound='residues') % box
        leaflet_res = leaflet.atoms.residues.resnames
        leaflet_pos[:, 2] = 0.0
        pos_list.append(leaflet_pos)
        res_list.append(leaflet_res)
    return pos_list, res_list, box



for grofile, xtcfile in zip(grofiles, xtcfiles):
    print(grofile[:-4])
    ul = utils.UniverseLoader(folder+grofile, folder+xtcfile, normal=2, by_tails=False )
    universe = ul.universe

    dim_arr = list()
    for frame in tqdm(universe.trajectory[20000::10], total=len(universe.trajectory[20000::10]), desc='compositions'):
        dim_arr.append(universe.dimensions[0])
    print('width', np.mean(dim_arr))
    """
    tails_selection = 'all'
    tails = ul.universe.select_atoms(tails_selection)
    tails_res = tails.residues
    
    lo_comp_list, ld_comp_list = [], []

    for frame in tqdm(universe.trajectory[20000::10], total = len(universe.trajectory[20000::10]), desc='compositions'):
        leaflets_pos, leaflets_res, (Lx, Ly, Lz) = positions()
        for res, pos in zip(leaflets_res, leaflets_pos):
            ld_pos = pos[res == 'DUPC']
            lo_pos = pos[res != 'DUPC']

            cl = freud.cluster.Cluster()
            box = freud.box.Box(Lx, Ly, is2D=True)
            cl_props = freud.cluster.ClusterProperties()

            # find COM of LO phase
            system = freud.AABBQuery(box, lo_pos)
            cl.compute(system, neighbors={'r_max': 7.5})
            cl_props.compute(system, cl.cluster_idx)
            lo_com = cl_props.centers[cl_props.sizes.argmax()][0] % Lx
            #plt.scatter(lo_pos[:, 0], lo_pos[:, 1])
            #plt.scatter(lo_pos[cl.cluster_idx==cl_props.sizes.argmax(), 0], lo_pos[cl.cluster_idx==cl_props.sizes.argmax(), 1])
            # find COM of LD phase
            system = freud.AABBQuery(box, ld_pos)
            cl.compute(system, neighbors={'r_max': 12.0})
            cl_props.compute(system, cl.cluster_idx)
            ld_com = cl_props.centers[cl_props.sizes.argmax()][0] % Lx
            #plt.scatter(lo_pos[:, 0], lo_pos[:, 1])
            #plt.scatter(ld_pos[:, 0], ld_pos[:, 1])
            #plt.scatter(ld_pos[cl.cluster_idx==cl_props.sizes.argmax(), 0], ld_pos[cl.cluster_idx==cl_props.sizes.argmax(), 1])
            #plt.scatter([ld_com], [60.0], s=100)
            #plt.scatter([lo_com], [60.0], s=100)
            #plt.vlines([ld_com-10.0, ld_com+10.0, lo_com-10.0, lo_com+10.0], 0.0, 150.0, linewidth=2.0)
            #plt.show()
            #get all resnames in +- 1nm from the phases COM
            dist_from_lo_com = np.abs(pos[:, 0] - lo_com)
            dist_from_lo_com = np.where(dist_from_lo_com > 0.5 * Lx, np.abs(dist_from_lo_com - Lx), dist_from_lo_com)
            lo_comp_list.append(res[dist_from_lo_com < 10.0])
            dist_from_ld_com = np.abs(pos[:, 0] - ld_com)
            dist_from_ld_com = np.where(dist_from_ld_com > 0.5 * Lx, np.abs(dist_from_ld_com - Lx), dist_from_ld_com)
            ld_comp_list.append(res[dist_from_ld_com < 10.0])

    # flatten the lists
    lo_comp_list = [item for sublist in lo_comp_list for item in sublist]
    ld_comp_list = [item for sublist in ld_comp_list for item in sublist]

    # count unique resnames
    resnames, counts = np.unique(lo_comp_list, return_counts=True)
    print('====lo====')
    for resname, c in zip(resnames, counts):
        print(resname, c / counts.sum())
    resnames, counts = np.unique(ld_comp_list, return_counts=True)
    print('====ld====')
    for resname, c in zip(resnames, counts):
        print(resname, c / counts.sum())
    """
