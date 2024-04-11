import numpy as np
import matplotlib.pyplot as plt
import os
import MDAnalysis as mda
from MDAnalysis.transformations import NoJump
import lipyphilic as lpp
from time import sleep
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from tqdm import tqdm
import sys
import multiprocessing
# remove warnings
import warnings
warnings.filterwarnings("ignore")

# get from arguments
WHICH = sys.argv[1]
FOLDER = sys.argv[2]

DEBUG = False

WHICH_LIST = ["DPPC", "DAPC", "DAPC_LIN", "DAPC_AS", "DPPC_AS"]
FOLDER_LIST = ["DPPC", "DAPC", "DAPC_LIN", "DAPC_POPC", "DPPC_POPC"]
COLORS = ["blue", "red", "green", "purple"]
SHIFT_METHOD = "LD_COM" # "PEAKS" or "LD_COM"
FLATTEN = True

COMP_LO = {"DPPC": [0.63, 0.02, 0.35], "DAPC": [0.62, 0.03, 0.35], "DAPC_LIN": [0.62, 0.03, 0.35, 0.05], "DAPC_AS": [0.62, 0.03, 0.35, 0.0], "DPPC_AS": [0.63, 0.02, 0.35, 0.0]}
COMP_LD = {"DPPC": [0.06, 0.85, 0.09], "DAPC": [0.05, 0.87, 0.08], "DAPC_LIN": [0.05, 0.87, 0.08, 0.05], "DAPC_AS": [0.05, 0.87, 0.08, 0.0], "DPPC_AS": [0.06, 0.85, 0.09, 0.0]}
CHOL_MIDPLANE_DENSITY_LO = {"DPPC": 0.00324, "DAPC": 0.02601, "DAPC_LIN": 0.02601, "DAPC_AS": 0.0, "DPPC_AS": 0.0}
CHOL_MIDPLANE_DENSITY_LD = {"DPPC": 0.05312, "DAPC": 0.04512, "DAPC_LIN": 0.04512, "DAPC_AS": 0.0, "DPPC_AS": 0.0}

I_MIN = 23
I_MAX = 38

SKIP = 10
EQUILIBRIUM = 20000
SMOOTH_BY = 21

# Parameters
sim_dir = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/{}/".format(FOLDER)

PLOTS_DIR = "simulations_analysis/plots/{}/".format(WHICH)
DATA_DIR = "simulations_analysis/data/{}/".format(WHICH)
CSV_DIR = "simulations_analysis/csv/{}/".format(WHICH)

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)

# load a universe just to get lipid names
grofile = "{}_{}_nw.gro".format(WHICH, I_MIN)
u = mda.Universe(os.path.join(sim_dir, grofile))
LIPIDS = np.sort(np.unique(u.residues.resnames)).tolist()
u = None

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

    def set_leaflets():
        leaflets_file = DATA_DIR + "leaflets_{}.npy".format(i)
        leaflets_no_chol_file = DATA_DIR + "leaflets_no_chol_{}.npy".format(i)
        residues_no_chol = u.select_atoms("name GL1 GL2").residues
        # load leaflets if exist
        if os.path.exists(leaflets_file) and os.path.exists(leaflets_no_chol_file):
            leaflets = np.load(leaflets_file).T
            leaflets_no_chol = np.load(leaflets_no_chol_file).T
            # check if the trajectory's length is correct
            if leaflets.shape[1] == (len(u.trajectory) - EQUILIBRIUM):
                return leaflets, leaflets_no_chol, residues_no_chol
        # assign leaflets if doesn't exist or 
        leaflets = lpp.AssignCurvedLeaflets(universe=u,lipid_sel="name GL1 GL2 ROH", midplane_sel="name ROH",midplane_cutoff=10.0, lf_cutoff=10.0)
        leaflets_no_chol = lpp.AssignCurvedLeaflets(universe=u,lipid_sel="name GL1 GL2", lf_cutoff=10.0)
        total_leaflets, total_leaflets_no_chol = [], []
        for frame in tqdm(range(EQUILIBRIUM, len(u.trajectory)), disable=False):
            leaflets.run(start=frame, stop=frame+1, step=1)
            total_leaflets.append(leaflets.leaflets.flatten())
            leaflets_no_chol.run(start=frame, stop=frame+1, step=1)
            total_leaflets_no_chol.append(leaflets_no_chol.leaflets.flatten())
        total_leaflets = np.array(total_leaflets)
        total_leaflets_no_chol = np.array(total_leaflets_no_chol)
        print("Finished for", i)
        #print("Setting leaflets with cholesterol for", i)
        #leaflets.run(start=EQUILIBRIUM, verbose=True)
        #print("Setting leaflets without cholesterol for", i)
        #leaflets_no_chol.run(start=EQUILIBRIUM, verbose=True)
        #print("Finished without cholesterol for", i)
        #total_leaflets = leaflets.leaflets
        #total_leaflets_no_chol = leaflets_no_chol.leaflets
        # save leaflets
        np.save(leaflets_file, total_leaflets)
        np.save(leaflets_no_chol_file, total_leaflets_no_chol)    
        return total_leaflets, total_leaflets_no_chol, residues_no_chol
    
    leaflets, leaflets_no_chol, residues_no_chol = set_leaflets()

    def update_leaflets(frame, cholesterol=True):
        leaflets_ = leaflets if cholesterol else leaflets_no_chol
        curr_assignment = leaflets_[:, frame - EQUILIBRIUM]
        res = u.residues if cholesterol else residues_no_chol
        leaflet1 = res[curr_assignment == 1]
        leaflet2 = res[curr_assignment == -1]
        midplane = res[curr_assignment == 0]
        return leaflet1, leaflet2, midplane

    arc_length_dict = {}

    def arc_length(f, x, maxvalue, args=()):
        if len(arc_length_dict) > 1000:
            arc_length_dict.clear()
        if f not in arc_length_dict.keys():
            xs = np.arange(0, maxvalue, 0.5)
            ys = f(xs, *args)
            dxs = np.diff(xs)
            dys = np.diff(ys)
            ds = np.sqrt(dxs**2 + dys**2)
            s = np.cumsum(ds)
            s = np.insert(s, 0, 0)
            arc_length_dict[f] = [xs, s]
        xs, s = arc_length_dict[f]
        return np.interp(x, xs, s)

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
        for leaflet in [leaflet1, leaflet2, midplane]:
            resnames.append([res.resname for res in leaflet])
            pos_ = np.array([arc_length(f, res.atoms[-1].position[0], u.dimensions[0]) for res in leaflet])  
            pos.append(pos_)
        arc_length1 = arc_length(f, u.dimensions[0], u.dimensions[0])
        return resnames, pos, arc_length1

    def density(bins=60):
        # fit the curve for both leaflets
        resnames, pos, period = fit_the_curve()
        # compute the histograms
        hists = [[], [], []]
        for resname in LIPIDS:
            for (resnames_, pos_, hists_) in zip(resnames[:2], pos[:2], hists[:2]):
                mask = np.array(resnames_) == resname
                hist = np.histogram(pos_[mask], bins=bins, range=(0, period))[0]
                hists_.append(hist)
        # midplane cholesterol plane density
        dx = period/bins
        hists[2] = np.histogram(pos[2], bins=bins, range=(0, period))[0]/dx/u.dimensions[1]*100
        return hists, period, u.dimensions[0]
    
    def smooth_xz(x, z):
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

    def center_point(x, f):
        # get the midplane arclength function
        period = arc_length(f, u.dimensions[0], u.dimensions[0])
        x = np.sort(x)
        x_new = arc_length(f, x, u.dimensions[0])
        hist, bins = np.histogram(x_new, bins=20, range=(0, period))
        # get first fourier mode
        w = np.exp(-2j*np.pi*np.arange(20) / 20)
        f1 = hist.dot(w)
        # get the angle
        angle = np.angle(f1.conj())
        if angle < 0:
            angle += 2*np.pi
        approx_center = period * angle / (2*np.pi)
        if DEBUG:
            plt.plot(x_new, np.zeros_like(x), "o")
            plt.plot(0.5*(bins[1:] + bins[:-1]), hist, "-")
            plt.plot(approx_center, 0, "o", color='red', markersize=10)
            plt.savefig("test_cen1.png")
            plt.close()
            sleep(0.5)
        # shift x so that the max_bin is in the middle
        x_shifted = x_new - approx_center + period/2
        x_shifted %= period
        if DEBUG:
            plt.plot(x_shifted, np.zeros_like(x), "o")
            hist1, bins1 = np.histogram(x_shifted, bins=10, range=(0, period))
            plt.plot(0.5*(bins1[1:] + bins1[:-1]), hist1, "-")
            plt.savefig("test_cen2.png")
            plt.close()
            sleep(0.5)
        # find a point such that half the points to its left and half to the right
        x_shifted = np.sort(x_shifted)
        x_center = 0.5 * (x_shifted[len(x_shifted)//2] + x_shifted[len(x_shifted)//2-1]) + approx_center - period/2
        x_center %= period
        if DEBUG:
            plt.plot(x_new, np.zeros_like(x), "o")
            plt.plot(0.5*(bins[1:] + bins[:-1]), hist, "-")
            plt.plot(x_center, 0, "o", color='red', markersize=10)
            plt.savefig("test_cen3.png")
            plt.close()
            sleep(0.5)
        # find original x_center from arclength
        x_center = np.interp(x_center, x_new, x)
        return x_center
        

    def find_peak(x, f):
        if SHIFT_METHOD == "LD_COM":
            return center_point(x, f)
        # find the peak
        l_period = int(u.dimensions[0])
        x_ = np.linspace(0, u.dimensions[0], l_period*100)
        fx_ = f(x_)
        # get rough derivative by 10% x-dim difference, use pbc
        dfx_ = np.abs(np.roll(fx_, -l_period*5) - np.roll(fx_, l_period*5))
        # peak region
        argmax_x = np.argmax(fx_)
        region_mask = np.zeros_like(x_, dtype=bool)
        # mask -+ 50 points around thet peak, use pbc
        region_mask[np.arange(argmax_x-l_period*15, argmax_x+l_period*15) % (l_period*100)] = True
        # estimate peak by derivative closest to 0
        max_x = x_[region_mask][np.argmin(dfx_[region_mask])]
        if DEBUG:
            plt.plot(x, z, "o")
            plt.plot(x_, fx_)
            plt.plot(max_x, f(max_x), "o", color='red', markersize=10)
        return max_x

    def save_peak_arr(peak_arr):
        if SHIFT_METHOD == "PEAKS": 
            np.save(DATA_DIR + "peak_arr_{}.npy".format(i), peak_arr)
        elif SHIFT_METHOD == "LD_COM":
            np.save(DATA_DIR + "peak_arr_com_{}.npy".format(i), peak_arr)

    def load_peak_arr():
        peak_arr = None
        if os.path.exists(DATA_DIR + "peak_arr_{}.npy".format(i)) and SHIFT_METHOD == "PEAKS":
            peak_arr = np.load(DATA_DIR + "peak_arr_{}.npy".format(i)).T
        elif os.path.exists(DATA_DIR + "peak_arr_com_{}.npy".format(i)) and SHIFT_METHOD == "LD_COM":
            peak_arr = np.load(DATA_DIR + "peak_arr_com_{}.npy".format(i)).T
        return peak_arr

    
    def set_peak_arr():
        # check if already exists
        peak_arr = load_peak_arr()
        # check how much of the trajectory is not zero, should be the same as equilibrium
        if np.sum(peak_arr != 0.0) == len(u.trajectory[EQUILIBRIUM:]):
            return peak_arr
        # extract peaks of both leaflets
        count = 0
        peak1_arr = np.zeros(len(u.trajectory))
        peak2_arr = np.zeros(len(u.trajectory))
        first_atoms1, first_atoms2 = [], []
        for ts in u.trajectory[EQUILIBRIUM:]:
            if (np.random.rand() < 0.05) or DEBUG:
                print(count, "/", len(u.trajectory[EQUILIBRIUM:]), "for", i)
            if count % SKIP == 0:
                leaflet1, leaflet2, midplane = update_leaflets(ts.frame, cholesterol=False)
                first_atoms1 = mda.AtomGroup([res.atoms[0] for res in leaflet1])
                first_atoms2 = mda.AtomGroup([res.atoms[0] for res in leaflet2])
            period = u.dimensions[0]
            pos1 = np.array(first_atoms1.positions)
            pos2 = np.array(first_atoms2.positions)
            x1, z1 = pos1[:, 0] % period, pos1[:, 2]
            x2, z2 = pos2[:, 0] % period, pos2[:, 2]
            peak1 = find_peak(x1 if SHIFT_METHOD == "PEAKS" else x1[first_atoms1.resnames == "DUPC"], smooth_xz(x1, z1))/period * 2 *np.pi
            peak2 = find_peak(x2,smooth_xz(x2, z2))/period * 2 *np.pi if SHIFT_METHOD == "PEAKS" else peak1
            if DEBUG:
                #plt.savefig("test.png")
                #plt.close()
                #sleep(0.7)
                pass
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
        plt.savefig("{}peaks_{}_{}.png".format(PLOTS_DIR, SHIFT_METHOD, i))
        plt.close(fig)
        save_peak_arr(peak_arr)
        return peak_arr
    
    peak_arr = set_peak_arr()

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
    
    hists1_arr, hists2_arr, histsm_arr = [], [], []
    period_arr, x_length_arr = [], []

    count = 0
    for ts in u.trajectory[EQUILIBRIUM::SKIP]:
        if (np.random.rand() < 0.3) or DEBUG:
            print(count, "/", len_traj, "for", i)
        leaflet1, leaflet2, midplane = update_leaflets(ts.frame)
        if DEBUG:
            fig, ax = plt.subplots()
            for leaflet in [leaflet1, leaflet2, midplane]:
                pos = np.array([res.atoms[0].position for res in leaflet])
                x = pos[:, 0]
                z = pos[:, 2]
                ax.plot(x, z, "o")
            plt.savefig("test1.png")
            plt.close(fig)
            sleep(0.7)
        count += 1
        [hists1, hists2, histsm], period, x_length = density()
        hists1_arr.append(hists1)
        hists2_arr.append(hists2)
        histsm_arr.append(histsm)
        period_arr.append(period)
        x_length_arr.append(x_length)
    hists1_arr = np.array(hists1_arr)
    hists2_arr = np.array(hists2_arr)
    histsm_arr = np.array(histsm_arr)
    period_arr = np.array(period_arr)
    x_length_arr = np.array(x_length_arr)
    hists1 = np.mean(hists1_arr, axis=0)
    hists2 = np.mean(hists2_arr, axis=0)
    histsm = np.mean(histsm_arr, axis=0).flatten()
    period = np.mean(period_arr)
    x_length = np.mean(x_length_arr)
    tot_hists1 = np.sum(hists1, axis=0)
    tot_hists2 = np.sum(hists2, axis=0)
    hists1 /= tot_hists1
    hists2 /= tot_hists2
    xs = np.linspace(0, period, hists1.shape[1]+1, endpoint=True)
    xs = (xs[1:] + xs[:-1]) / 2
    print(i, "done")
    return xs, hists1, hists2, histsm, period, x_length

# run the workers in parallel   
if __name__ == '__main__':
    pool = multiprocessing.Pool(1 if DEBUG else min([8, I_MAX-I_MIN+1]))
    results = pool.map(worker, range(I_MIN, I_MAX+1))
    pool.close()

    def find_curvature(period, x_length):
        # convert to nm
        period *= 0.1
        x_length *= 0.1
        # Whewell's formula
        def phi(s, Rmax):
            return - period / (2 * np.pi) * np.cos(2 * np.pi * s / period) / Rmax
        # x = integral of cos(phi(s, Rmax)) ds from 0 to period 
        def x(Rmax):
            ss = np.linspace(0, period, 10000)
            return np.trapz(np.cos(phi(ss, Rmax)), ss)
        # find x to give x_length, x is monotonically increasing
        Rmax = 100000
        while x(Rmax) > x_length:
            Rmax /= 2
        # use bisection to find the root
        a, b = Rmax, 2*Rmax
        while b - a > 1e-6:
            c = (a + b) / 2
            if x(c) < x_length:
                a = c
            else:
                b = c
        return (a + b) / 2


    period_23 = 0.0
    # plot the results
    for i, result in enumerate(results):
        xs, mean_hists1, mean_hists2, histsm, period, x_length = result
        if i == 0:
            period_23 = period
        # for each molecule, plot the local percentage 
        fig, axs = plt.subplots(3, 1)
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
        # plot midplane
        axs[2].plot(xs, histsm, color='black')
        axs[2].axhline(CHOL_MIDPLANE_DENSITY_LO[WHICH], color='black', linestyle='--', alpha=0.3)
        axs[2].axhline(CHOL_MIDPLANE_DENSITY_LD[WHICH], color='black', linestyle='-.', alpha=0.3)
        axs[0].set_title("Relative composition for I={} and {}".format(i+I_MIN, WHICH))
        axs[0].set_ylabel("Composition [%]")
        axs[1].set_xlabel("x [A]")
        axs[1].set_ylabel("Composition [%]")
        axs[2].set_ylabel("Chol dens. [nm^-2]")
        axs[0].set_ylim(0, 100)
        axs[1].set_ylim(0, 100)
        axs[2].set_ylim(0, 0.09)
        axs[0].set_xlim(0, period)
        axs[1].set_xlim(0, period)
        axs[2].set_xlim(0, period)
        axs[0].legend(LIPIDS)
        filename = "{}composition_{}_{}.png".format(PLOTS_DIR, SHIFT_METHOD, i+I_MIN) if not FLATTEN else "{}composition_{}_{}_flattened.png".format(PLOTS_DIR, SHIFT_METHOD, i+I_MIN)
        plt.savefig(filename)
        plt.close()
        # save data to CSV
        data = np.vstack([xs, mean_hists1])
        data = np.vstack([data, mean_hists2])
        data = np.vstack([data, histsm])
        data = data.T
        headers = ["x"]
        for lipid in LIPIDS:
            headers.append(lipid + "_leaflet1")
        for lipid in LIPIDS:
            headers.append(lipid + "_leaflet2")
        headers.append("CHOL_midplane")
        np.savetxt(CSV_DIR + "composition_{}_{}.csv".format(SHIFT_METHOD, i+I_MIN), data, delimiter=",", header=",".join(headers), comments="")
        # find the curvature
        Rmax = find_curvature(period_23, x_length)
        print("Radius of curvature for I={} is".format(i+I_MIN), Rmax, "nm")