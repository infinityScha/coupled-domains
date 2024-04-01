import numpy as np
import matplotlib.pyplot as plt
import os
import MDAnalysis as mda
from MDAnalysis.transformations import NoJump
from scipy.optimize import curve_fit
from utils import periodic_griddata as pg
from tqdm import tqdm
from freud.density import GaussianDensity
from freud.cluster import Cluster
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import lipyphilic as lpp
from time import sleep, time

# remove warnings
import warnings
warnings.filterwarnings("ignore")

which = "DAPC"

# Parameters
sim_dir = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/{}/".format(which)

I_MIN = 23
I_MAX = 38

UTILILTY_DATA_DIR = "simulations_analysis/utility_data/{}/".format(which)

if not os.path.exists(UTILILTY_DATA_DIR):
    os.makedirs(UTILILTY_DATA_DIR)

for i in range(I_MIN, I_MAX+1):

    grofile = "{}_{}_nw.gro".format(which, i)
    xtcfile = "{}_{}.xtc".format(which, i)

    u = mda.Universe(os.path.join(sim_dir, grofile), os.path.join(sim_dir, xtcfile))
    u.trajectory.add_transformations(NoJump())

    leaflets = lpp.AssignCurvedLeaflets(universe=u,lipid_sel="name GL1 GL2 ROH",midplane_sel="resname CHOL and name ROH C2",midplane_cutoff=10.0, lf_cutoff=10.0)

    def update_leaflets(frame):
        leaflets.run(start=frame, stop=frame+1, step=1)
        u.trajectory[frame]
        assignment = leaflets.leaflets.flatten()
        leaflet1 = u.residues[assignment == 1]
        leaflet2 = u.residues[assignment == -1]
        return leaflet1, leaflet2
    # extract peaks of both leaflets
    count = 0
    peak1_arr = []
    peak2_arr = []
    time0 = time()
    first_atoms1, first_atoms2 = [], []
    for ts in tqdm(u.trajectory, total=len(u.trajectory), desc="run{}".format(i)):
        if count % 100 == 0:
            leaflet1, leaflet2 = update_leaflets(ts.frame)
            first_atoms1 = mda.AtomGroup([res.atoms[0] for res in leaflet1])
            first_atoms2 = mda.AtomGroup([res.atoms[0] for res in leaflet2])
        period = u.dimensions[0]
        pos1 = np.array(first_atoms1.positions)
        pos2 = np.array(first_atoms2.positions)
        x1, z1 = pos1[:, 0] % period, pos1[:, 2]
        x2, z2 = pos2[:, 0] % period, pos2[:, 2]
        peak1 = x1[np.argmax(z1)]/period * 2 *np.pi
        peak2 = x2[np.argmax(z2)]/period * 2 *np.pi
        time0 = time()
        peak1_arr.append(peak1)
        peak2_arr.append(peak2)
        count += 1
    peak1_arr = np.array(peak1_arr)
    peak2_arr = np.array(peak2_arr)
    sin_peak1_arr, cos_peak1_arr = np.sin(peak1_arr), np.cos(peak1_arr)
    sin_peak2_arr, cos_peak2_arr = np.sin(peak2_arr), np.cos(peak2_arr)
    # average the peaks
    sin_peak_arr = 0.5 * (sin_peak1_arr + sin_peak2_arr)
    cos_peak_arr = 0.5 * (cos_peak1_arr + cos_peak2_arr)
    # do moving average to get the average peak
    sin_peak_arr = np.convolve(sin_peak_arr, np.ones(20)/20, mode="valid")
    cos_peak_arr = np.convolve(cos_peak_arr, np.ones(20)/20, mode="valid")
    # deduce peak_arr from sin_peak_arr and cos_peak_arr
    peak_arr = np.arctan2(sin_peak_arr, cos_peak_arr)
    peak_arr %= 2 * np.pi
    np.save(UTILILTY_DATA_DIR + "peak_arr_{}.npy".format(i), peak_arr)