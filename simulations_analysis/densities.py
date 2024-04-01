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
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d

# remove warnings
import warnings
warnings.filterwarnings("ignore")

DEBUG = False

WHICH = "DAPC"
FLATTEN = True


# Parameters
sim_dir = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/{}/".format(WHICH)

I_MIN = 23
I_MAX = 32

SKIP = 10
EQUILIBRIUM = 25000
SMOOTH_BY = 21

RESNAMES = [WHICH, "DUPC", "CHOL"]
COLORS = ["blue", "red", "green"]
COMP_LO = {"DPPC": [0.62, 0.02, 0.36], "DAPC": [0.61, 0.04, 0.35]}
COMP_LD = {"DPPC": [0.04, 0.85, 0.11], "DAPC": [0.02, 0.88, 0.10]}

PLOTS_DIR = "simulations_analysis/plots/{}/".format(WHICH)
DATA_DIR = "simulations_analysis/data/{}/".format(WHICH)

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def moving_average(data, window_size):
    """
    Compute the moving average of a given data array.

    Parameters:
    data (array_like): The input data array.
    window_size (int): The size of the moving window.

    Returns:
    array_like: The moving average array with extrapolation.
    """
    # Pad the data array with first and last element for extrapolation
    data = np.copy(data)
    padded_data = np.pad(data, (int((window_size-1)/2), int((window_size-1)/2)), mode='edge')

    # Compute the moving average using convolution
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(padded_data, weights, mode='valid')

    return moving_avg

def worker(i):

    grofile = "{}_{}_nw.gro".format(WHICH, i)
    xtcfile = "{}_{}.xtc".format(WHICH, i)

    # Load trajectory
    u = mda.Universe(os.path.join(sim_dir, grofile), os.path.join(sim_dir, xtcfile))
    # center first frame and save
    ag = u.select_atoms("name GL1 GL2")
    u.trajectory.add_transformations(lpp.transformations.transformations.center_membrane(ag, min_diff=2.5), NoJump())
    tails = u.select_atoms("name C4A C4B C2")

    def set_leaflets():
        leaflets_file = DATA_DIR + "leaflets_{}.npy".format(i)
        leaflets_no_chol_file = DATA_DIR + "leaflets_no_chol_{}.npy".format(i)
        residues_no_chol = u.select_atoms("name GL1 GL2").residues
        # load leaflets if exist
        if os.path.exists(leaflets_file) and os.path.exists(leaflets_no_chol_file):
            leaflets = np.load(leaflets_file)
            leaflets_no_chol = np.load(leaflets_no_chol_file)
            # check if the trajectory's length is correct
            if leaflets.shape[1] == (len(u.trajectory) - EQUILIBRIUM):
                return leaflets, leaflets_no_chol, residues_no_chol
        # assign leaflets if doesn't exist or 
        leaflets = lpp.AssignCurvedLeaflets(universe=u,lipid_sel="name GL1 GL2 ROH",midplane_sel="resname CHOL and name ROH C2",midplane_cutoff=10.0, lf_cutoff=10.0)
        leaflets_no_chol = lpp.AssignCurvedLeaflets(universe=u,lipid_sel="name GL1 GL2", lf_cutoff=10.0)    
        print("Setting leaflets for", i)
        leaflets.run(start=EQUILIBRIUM)
        print("Setting leaflets without cholesterol for", i)
        leaflets_no_chol.run(start=EQUILIBRIUM)
        print("Finished without cholesterol for", i)
        # save leaflets
        np.save(leaflets_file, leaflets.leaflets)
        np.save(leaflets_no_chol_file, leaflets_no_chol.leaflets)    
        return leaflets.leaflets, leaflets_no_chol.leaflets, residues_no_chol
    
    leaflets, leaflets_no_chol, residues_no_chol = set_leaflets()

    def update_leaflets(frame, cholesterol=True):
        leaflets_ = leaflets if cholesterol else leaflets_no_chol
        curr_assignment = leaflets_[:, frame - EQUILIBRIUM]
        res = u.residues if cholesterol else residues_no_chol
        leaflet1 = res[curr_assignment == 1]
        leaflet2 = res[curr_assignment == -1]
        return leaflet1, leaflet2

    def arc_length(f, x, args=()):
        xs = np.arange(0, x, 0.5)
        if xs.size < 5:
            xs = np.linspace(0, x, 5)
        xs = np.append(xs, x)
        ys = f(xs, *args)
        dxs = np.diff(xs)
        dys = np.diff(ys)
        return np.sum(np.sqrt(dxs**2 + dys**2))

    def fit_the_curve():
        resnames, pos = [], []
        # smooth the midplane
        if FLATTEN:
            tails_pos = tails.positions
            x = tails_pos[:, 0]
            z = tails_pos[:, 2]
            f = smooth_xz(x, z)
        else:
            f = lambda x: x
        for leaflet in [leaflet1, leaflet2]:
            resnames.append([res.resname for res in leaflet])
            pos_ = np.array([arc_length(f, res.atoms[-1].position[0]) for res in leaflet])  
            pos.append(pos_)
        arc_length1 = arc_length(f, u.dimensions[0])
        return resnames, pos, arc_length1

    def density(bins=60):
        # fit the curve for both leaflets
        resnames, pos, period = fit_the_curve()
        # compute the histograms
        hists = [[], []]
        for resname in RESNAMES:
            for (resnames_, pos_, hists_) in zip(resnames, pos, hists):
                mask = np.array(resnames_) == resname
                hist = np.histogram(pos_[mask], bins=bins, range=(0, period))[0]
                hists_.append(hist)
        return hists, period
    
    def smooth_xz(x, z, bins=60):
        # average the z values over int(u.dimensions[0]/4) bins of x
        x_bins = np.linspace(0, u.dimensions[0], int(u.dimensions[0]/8)+1, endpoint=True)
        x_bins_wider = np.linspace(0, u.dimensions[0], 6, endpoint=True)
        z_bins = np.zeros(len(x_bins)-1)
        # find outliers and move them to the group through the pbc
        for i in range(len(x_bins_wider)-1):
            mask = (x >= x_bins_wider[i]) & (x < x_bins_wider[i+1])
            if np.sum(mask) == 0:
                continue
            masked_z = z[mask]
            curr_median = np.median(masked_z)
            half_box = 0.5*u.dimensions[2]
            while np.abs(masked_z - curr_median).max() > half_box:
                masked_z = np.where((masked_z - curr_median) > half_box, masked_z - u.dimensions[2], masked_z)
                masked_z = np.where((masked_z - curr_median) < -half_box, masked_z + u.dimensions[2], masked_z)
            z[mask] = masked_z
        for i in range(len(z_bins)):
            mask = (x >= x_bins[i]) & (x < x_bins[i+1])
            if np.sum(mask) == 0:
                continue
            masked_z = z[mask]
            z_bins[i] = np.mean(z[mask])
        # smooth with np.convolve
        z_bins = convolve1d(z_bins, np.ones(5)/5, mode='wrap')
        # periodic spline interpolation
        x_bins = 0.5 * (x_bins[1:] + x_bins[:-1])
        f = interp1d(x_bins, z_bins, kind='cubic', bounds_error=False, fill_value='extrapolate')
        return f

    def find_peak(x, z):
        f = smooth_xz(x, z)
        # find the peak
        l_period = int(u.dimensions[0])
        x_ = np.linspace(0, u.dimensions[0], l_period*100)
        fx_ = f(x_)
        # get rough derivative by 10% x-dim difference, use pbc
        dfx_ = np.abs(np.roll(fx_, -l_period*5) - np.roll(fx_, l_period*5))
        # peak region
        argmax_x = np.argmax(fx_)
        region_mask = np.zeros_like(x_, dtype=bool)
        # mask -+ 50 points around the peak, use pbc
        region_mask[np.arange(argmax_x-l_period*15, argmax_x+l_period*15) % (l_period*100)] = True
        # estimate peak by derivative closest to 0
        max_x = x_[region_mask][np.argmin(dfx_[region_mask])]
        if DEBUG:
            plt.plot(x, z, "o")
            plt.plot(x_, fx_)
            plt.plot(max_x, f(max_x), "o", color='red', markersize=10)
        return max_x

    # extract peaks of both leaflets
    count = 0
    peak1_arr = np.zeros(len(u.trajectory))
    peak2_arr = np.zeros(len(u.trajectory))
    first_atoms1, first_atoms2 = [], []
    for ts in u.trajectory[EQUILIBRIUM:]:
        if (np.random.rand() < 0.05) or DEBUG:
            print(count, "/", len(u.trajectory[EQUILIBRIUM:]), "for", i)
        if count % SKIP == 0:
            leaflet1, leaflet2 = update_leaflets(ts.frame, cholesterol=False)
            first_atoms1 = mda.AtomGroup([res.atoms[0] for res in leaflet1])
            first_atoms2 = mda.AtomGroup([res.atoms[0] for res in leaflet2])
        period = u.dimensions[0]
        pos1 = np.array(first_atoms1.positions)
        pos2 = np.array(first_atoms2.positions)
        x1, z1 = pos1[:, 0] % period, pos1[:, 2]
        x2, z2 = pos2[:, 0] % period, pos2[:, 2]
        peak1 = find_peak(x1,z1)/period * 2 *np.pi
        peak2 = find_peak(x2,z2)/period * 2 *np.pi
        if DEBUG:
            plt.savefig("test.png")
            plt.close()
            sleep(0.7)
        peak1_arr[ts.frame] = peak1
        peak2_arr[ts.frame] = peak2
        count += 1
    peak1_arr = np.array(peak1_arr)
    peak2_arr = np.array(peak2_arr)
    fig, ax = plt.subplots()
    sin_peak1_arr, cos_peak1_arr = np.sin(peak1_arr), np.cos(peak1_arr)
    sin_peak2_arr, cos_peak2_arr = np.sin(peak2_arr), np.cos(peak2_arr)
    ax.plot(sin_peak1_arr[EQUILIBRIUM:], label="leaflet1")
    ax.plot(sin_peak2_arr[EQUILIBRIUM:], label="leaflet2")
    # average the peaks
    sin_peak_arr = 0.5 * (sin_peak1_arr + sin_peak2_arr)
    cos_peak_arr = 0.5 * (cos_peak1_arr + cos_peak2_arr)
    ax.plot(sin_peak_arr[EQUILIBRIUM:], label="average")
    # do moving average to get the average peak
    sin_peak_arr[EQUILIBRIUM:] = moving_average(sin_peak_arr[EQUILIBRIUM:], SMOOTH_BY)
    cos_peak_arr[EQUILIBRIUM:] = moving_average(cos_peak_arr[EQUILIBRIUM:], SMOOTH_BY)
    # deduce peak_arr from sin_peak_arr and cos_peak_arr
    peak_arr = np.arctan2(sin_peak_arr, cos_peak_arr)
    peak_arr %= 2 * np.pi
    ax.plot(sin_peak_arr[EQUILIBRIUM:], label="average (smoothed)")
    ax.legend()
    plt.savefig("{}peaks_{}.png".format(PLOTS_DIR, i))
    plt.close(fig)

    # add transformation to the trajectory that shift it by peak_arr
    def shift_by_peak(ts):
        pos = np.copy(ts.positions)
        pos[:, 0] -= peak_arr[ts.frame] * ts.dimensions[0] / (2 * np.pi)
        pos[:, 0] %= ts.dimensions[0]
        ts.positions = pos
        return ts

    u = mda.Universe(os.path.join(sim_dir, grofile), os.path.join(sim_dir, xtcfile))
    ag = u.select_atoms("name GL1 GL2 ROH")
    u.trajectory.add_transformations(lpp.transformations.transformations.center_membrane(ag, min_diff=2.5), NoJump(), shift_by_peak)
    leaflets, leaflets_no_chol, residues_no_chol = set_leaflets()
    tails = u.select_atoms("name C4A C4B C2")

    print(i, "start")
    len_traj = len(u.trajectory[EQUILIBRIUM::SKIP])
    
    hists1_arr, hists2_arr = [], []
    period_arr = []

    count = 0
    for ts in u.trajectory[EQUILIBRIUM::SKIP]:
        if (np.random.rand() < 0.3) or DEBUG:
            print(count, "/", len_traj, "for", i)
        leaflet1, leaflet2 = update_leaflets(ts.frame)
        if DEBUG:
            fig, ax = plt.subplots()
            for leaflet in [leaflet1, leaflet2]:
                pos = np.array([res.atoms[0].position for res in leaflet])
                x = pos[:, 0]
                z = pos[:, 2]
                ax.plot(x, z, "o")
            plt.savefig("test1.png")
            plt.close(fig)
            sleep(0.7)
        count += 1
        [hists1, hists2], period = density()
        hists1_arr.append(hists1)
        hists2_arr.append(hists2)
        period_arr.append(period)
    hists1_arr = np.array(hists1_arr)
    hists2_arr = np.array(hists2_arr)
    hists1 = np.mean(hists1_arr, axis=0)
    hists2 = np.mean(hists2_arr, axis=0)
    period = np.mean(period_arr)
    tot_hists1 = np.sum(hists1,axis=0)
    tot_hists2 = np.sum(hists2, axis=0)
    hists1 /= tot_hists1
    hists2 /= tot_hists2
    xs = np.linspace(0, period, hists1.shape[1]+1, endpoint=True)
    xs = (xs[1:] + xs[:-1]) / 2
    print(i, "done")
    return xs, hists1, hists2, period

# run the workers in parallel   
import multiprocessing
pool = multiprocessing.Pool(1 if DEBUG else min([os.cpu_count(), I_MAX-I_MIN+1]))
results = pool.map(worker, range(I_MIN, I_MAX+1))
pool.close()

# plot the results
for i, result in enumerate(results):
    xs, mean_hists1, mean_hists2, period = result
    # for each molecule, plot the local percentage 
    fig, axs = plt.subplots(2, 1)
    # plot histograms
    for (hist1, hist2, color) in zip(mean_hists1, mean_hists2, COLORS):
        axs[0].plot(xs, hist1*100, color=color)
        axs[1].plot(xs, hist2*100, color=color)
    # plot ideal compositions
    for comp_lo, comp_ld, color in zip(COMP_LO[WHICH], COMP_LD[WHICH], COLORS):
        axs[0].axhline(comp_lo*100, color=color, linestyle='--', alpha=0.3)
        axs[0].axhline(comp_ld*100, color=color, linestyle='-.', alpha=0.3)
        axs[1].axhline(comp_lo*100, color=color, linestyle='--', alpha=0.3)
        axs[1].axhline(comp_ld*100, color=color, linestyle='-.', alpha=0.3)
    axs[0].set_title("Relative composition for I={} and {}".format(i+I_MIN, WHICH))
    axs[0].set_ylabel("Composition [%]")
    axs[1].set_xlabel("x [A]")
    axs[1].set_ylabel("Composition [%]")
    axs[0].set_ylim(0, 100)
    axs[1].set_ylim(0, 100)
    axs[0].set_xlim(0, period)
    axs[1].set_xlim(0, period)
    axs[0].legend(RESNAMES)
    filename = "{}composition_{}.png".format(PLOTS_DIR, i+I_MIN) if not FLATTEN else "{}composition_{}_flattened.png".format(PLOTS_DIR, i+I_MIN)
    plt.savefig(filename)
    plt.close()