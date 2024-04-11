import numpy as np
import matplotlib.pyplot as plt
import os
import MDAnalysis as mda
from MDAnalysis.transformations import nojump
from scipy.optimize import curve_fit
from utils import periodic_griddata as pg
from tqdm import tqdm
from freud.density import GaussianDensity
from freud.cluster import Cluster
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


# remove warnings
import warnings
warnings.filterwarnings("ignore")

lipid = "DAPC"
folder = "CURVED_LINACTANT"
which = "DAPC_LIN"

# Parameters
sim_dir = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/{}/".format(folder)

I_MIN = 23
I_MAX = 38

EPS_LO = 14
MIN_SAMPLES_LO = 5
EPS_LD = 14
MIN_SAMPLES_LD = 4

START = 5000
SKIP = 10
EQUILIBRIUM = 5000

PLOTS_DIR = "simulations_analysis/plots/{}/".format(which)

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# define default colors for the plots to have gradual color
colors = plt.cm.viridis(np.linspace(0, 1, np.abs(I_MAX-I_MIN)+1))

def worker(i):
    lo_arr = []
    comp_arr_lo = []
    comp_arr_ld = []

    grofile = "{}_{}_nw.gro".format(which, i)
    xtcfile = "{}_{}.xtc".format(which, i)

    # Load trajectory
    u = mda.Universe(os.path.join(sim_dir, grofile), os.path.join(sim_dir, xtcfile))

    leaflet1, leaflet2 = [], []

    def update_leaflets():
        leaflet1, leaflet2 = [], []
        for res in u.residues:
            pos1, pos2 = res.atoms.positions[0, 2], res.atoms.positions[-1, 2]
            if pos1 < pos2:
                leaflet1.append(res)
            else:
                leaflet2.append(res) 
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

    def flatten(pos, arc_length_f):
        # flatten the sines to the arc length
        x_flat = np.vectorize(arc_length_f)(pos[:, 0]  % u.dimensions[0])
        pos[:, 0] = x_flat 
        pos[:, 1] %= u.dimensions[1]
        pos[:, 2] = 0.0
        return pos

    def fit_the_curve(which_=0):
        leaflet = leaflet1 if which_ == 0 else leaflet2 
        resnames = [res.resname for res in leaflet]
        pos = []
        for res in leaflet:
            if res.resname == "CHOL":
                pos.append(res.atoms[-1].position)
            else:
                tail2 = res.atoms[-1].position
                tail1 = res.atoms[-6 if lipid == "DAPC" else -5].position
                pos.append((tail1 + tail2) / 2)
        pos = np.array(pos)

        x = pos[:, 0] % u.dimensions[0]
        z = pos[:, 2] % u.dimensions[2]
        period = u.dimensions[0]
        mid = z.mean()
        # fit position to a sine function, with the period and midline as fixed parameters
        def sine(x, A, u):
            return A * np.sin(2 * np.pi * x / period + u) + mid
        # cluster the (x.z) points to omit the outliers with freud
        cl = Cluster()
        xz = np.array([x, z, np.zeros_like(x)]).T
        cl.compute(system=(u.dimensions[:2], xz), neighbors={"r_max": 10.0})
        # find the largest cluster
        max_idx, max_size = 0, 0
        for i in range(cl.num_clusters):
            curr_size = len(cl.cluster_keys[i])
            if curr_size > max_size:
                max_size = curr_size
                max_idx = i
        # get the indices of the largest cluster
        cluster_indices = cl.cluster_keys[max_idx]
        x = x[cluster_indices]
        z = z[cluster_indices]
        # fit the data
        popt, _ = curve_fit(sine, x, z)

        def sine_arc_length(x):
            return arc_length(sine, x, popt)
        return resnames, pos, sine_arc_length, popt[0]

    def _clus_helper(pos_relevant, period, eps, min_samples):
        kdt = cKDTree(pos_relevant[:, :2], boxsize=np.array([period, u.dimensions[1]])+0.001)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        X = kdt.sparse_distance_matrix(kdt, eps)
        dbscan.fit(X)
        return dbscan.labels_

    def dbscan_clus_(resnames, pos, period):
        is_dupc = np.array([resname == "DUPC" for resname in resnames])
        not_dupc = np.invert(is_dupc)
        is_dapc = np.array([resname == lipid for resname in resnames])
        is_chol = np.array([resname == "CHOL" for resname in resnames])
        labels_lo = _clus_helper(pos[not_dupc], period, EPS_LO, MIN_SAMPLES_LO)
        labels_lo[labels_lo > -1] = 1
        labels_lo[labels_lo == -1] = 0
        labels_ld = _clus_helper(pos[np.invert(not_dupc)], period, EPS_LD, MIN_SAMPLES_LD)
        labels_ld[labels_ld != -1] = -2
        labels_ld[labels_ld == -1] = 1
        labels_ld[labels_ld == -2] = 0
        which_domain = np.zeros(len(pos), dtype=int)-1
        which_domain[not_dupc] = labels_lo
        which_domain[is_dupc] = labels_ld
        xx = np.linspace(0, period, int(np.ceil(period)))
        yy = np.linspace(0, u.dimensions[1], int(np.ceil(u.dimensions[1])))
        x_grid, y_grid = np.meshgrid(xx, yy)
        z_grid = pg.periodic_griddata(pos[:, :2], which_domain, (x_grid, y_grid), method="nearest", period=[period, u.dimensions[1]])
        in_lo = which_domain == 1
        lo_arr.append(z_grid.mean())
        dapc_lo = is_dapc[in_lo].mean()
        dupc_lo = is_dupc[in_lo].mean()
        chol_lo = is_chol[in_lo].mean()
        comp_arr_lo.append([dapc_lo, dupc_lo, chol_lo])
        not_in_lo = np.invert(in_lo)
        dapc_ld = is_dapc[not_in_lo].mean()
        dupc_ld = is_dupc[not_in_lo].mean()
        chol_ld = is_chol[not_in_lo].mean()
        comp_arr_ld.append([dapc_ld, dupc_ld, chol_ld])
        return z_grid, 1


    def domains_overlap():
        # flatten the curve
        resnames1, pos1, arch_length_f1, A1 = fit_the_curve(0)
        resnames2, pos2, arch_length_f2, A2 = fit_the_curve(1)
        arch_length_f = lambda x: (arch_length_f1(x) + arch_length_f2(x)) / 2
        pos1 = flatten(pos1, arch_length_f)
        pos2 = flatten(pos2, arch_length_f)
        period = arch_length_f(u.dimensions[0])
        grid1, n1 = dbscan_clus_(resnames1, pos1, period)
        grid2, n2 = dbscan_clus_(resnames2, pos2, period)
        # calculate the overlap
        overlap_arr = np.zeros((n1))
        area_arr = np.zeros(n1)
        
        for i in range(n1):
            area_i = (grid1 == i).sum()
            area_arr[i] = area_i
            for j in range(n2):
                overlap_area = np.logical_and(grid1 == i, grid2 == j).sum()
                overlap_arr[i] += overlap_area
            overlap_arr[i] /= area_i
        return overlap_arr, area_arr, 0.5*(np.abs(A1)+np.abs(A2))*4*np.pi*np.pi/period/period*10

    A_arr = []
    total_overlap_arr = []
    total_area_arr = []
    print(i, "start")
    len_traj = len(u.trajectory[START::SKIP])
    count = 0
    for ts in u.trajectory[START::SKIP]:
        if np.random.rand() < 0.3:
            print(count, "/", len_traj, "for", i)
        leaflet1, leaflet2 = update_leaflets()
        overlap_arr, area_arr, A = domains_overlap()
        total_overlap_arr.append(overlap_arr)
        total_area_arr.append(area_arr)
        A_arr.append(A)
        count += 1
    total_overlap_arr = np.concatenate(total_overlap_arr)
    total_area_arr = np.concatenate(total_area_arr)
    lo_arr = np.array(lo_arr)
    comp_arr_lo = np.array(comp_arr_lo)*100 # convert to percentage
    comp_arr_ld = np.array(comp_arr_ld)*100 # convert to percentage
    lo_area = np.mean(lo_arr[-int(EQUILIBRIUM/SKIP):])
    lo_comp = (np.mean(comp_arr_lo[-int(EQUILIBRIUM/SKIP):], axis=0)*10).astype(int)/10
    ld_comp = (np.mean(comp_arr_ld[-int(EQUILIBRIUM/SKIP):], axis=0)*10).astype(int)/10
    print(i, "done")
    return [lo_area, lo_comp, ld_comp, np.mean(A_arr), total_overlap_arr, lo_arr, comp_arr_lo, comp_arr_ld]


# run the workers in parallel
import multiprocessing
pool = multiprocessing.Pool(os.cpu_count())
results = pool.map(worker, range(I_MIN, I_MAX+1))
pool.close()

def block_averaging(arr, n_chunks=10):
    chunks = np.array_split(arr, n_chunks)
    mean_arr = np.array([np.mean(chunk) for chunk in chunks])
    ste = np.std(mean_arr) / np.sqrt(n_chunks - 1)
    return ste

total_A_arr = [res[3] for res in results]

# plot overlap evolution
fig, ax = plt.subplots()
for i, res in enumerate(results):
    overlap_arr = res[4]
    ax.plot(overlap_arr, label="I={}".format(i+I_MIN), color=colors[i])

ax.set_xlabel("Frame")
ax.set_ylabel("Overlap")
ax.set_ylim(0, 1)
ax.set_title("Overlap evolution for {}".format(which))
ax.legend()
plt.savefig("{}overlap_evolution_{}.png".format(PLOTS_DIR, which))
plt.close(fig)

# do the same with running averages
WINDOW = 100

fig, ax = plt.subplots()
for i, res in enumerate(results):
    overlap_arr = res[4]
    ax.plot(np.convolve(overlap_arr, np.ones(WINDOW)/WINDOW, mode="valid"), label="I={}".format(i+I_MIN), color=colors[i])

ax.set_xlabel("Frame")
ax.set_ylabel("Overlap")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Running average of overlap evolution for {}".format(which))
plt.savefig("{}overlap_evolution_running_{}.png".format(PLOTS_DIR, which))

# plot the curvature vs I
total_A_arr = np.array(total_A_arr)
fig, ax = plt.subplots()
ax.plot(np.arange(I_MIN, I_MAX+1), total_A_arr)
ax.set_xlabel("I")
ax.set_ylabel("Max C [nm^-1]")
ax.set_title("Midplane mean curvature vs I for {}".format(which))
plt.savefig("{}mean_curvature_{}.png".format(PLOTS_DIR, which))
plt.close()

# plot composition vs I
fig, ax = plt.subplots()
comp_arr = np.array([res[1] for res in results])
comp_evolution_arr = [res[6] for res in results]
comp_ste_arr = np.array([[block_averaging([c[i] for c in comp_evolution[-int(EQUILIBRIUM/SKIP):]], n_chunks=10) for i in range(3)] for comp_evolution in comp_evolution_arr])
# plot with error bar
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 0], yerr=comp_ste_arr[:, 0], label=lipid)
ax.plot(np.arange(I_MIN, I_MAX+1), comp_arr[:, 1] - np.mean(comp_arr[:, 1]), label="DUPC")
ax.plot(np.arange(I_MIN, I_MAX+1), comp_arr[:, 2] - np.mean(comp_arr[:, 2]), label="CHOL")
ax.set_xlabel("I")
ax.set_ylabel("delta Composition [%]")
ax.legend()
ax.set_title("Delta composition of the Lo domain for {}".format(which))
plt.savefig("{}ave_comp_lo_{}.png".format(PLOTS_DIR, which))
plt.close()

# plot composition vs I
fig, ax = plt.subplots()
comp_arr = np.array([res[2] for res in results])
comp_evolution_arr = [res[7] for res in results]
comp_ste_arr = np.array([[block_averaging([c[i] for c in comp_evolution[-int(EQUILIBRIUM/SKIP):]], n_chunks=10) for i in range(3)] for comp_evolution in comp_evolution_arr])
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 0] - np.mean(comp_arr[:, 0]), yerr=comp_ste_arr[:, 0], label=lipid)
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 1] - np.mean(comp_arr[:, 1]), yerr=comp_ste_arr[:, 1], label="DUPC")
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 2] - np.mean(comp_arr[:, 2]), yerr=comp_ste_arr[:, 2], label="CHOL")
ax.set_xlabel("I")
ax.set_ylabel("delta Composition [%]")
ax.legend()
ax.set_title("Delta composition of the Ld domain for {}".format(which))
plt.savefig("{}ave_comp_ld_{}.png".format(PLOTS_DIR, which))
plt.close()

# plot the area of the lo domain vs I
fig, ax = plt.subplots()
lo_area_arr = np.array([res[0] for res in results])
lo_area_evol = [res[5] for res in results]
lo_area_ste = np.array([block_averaging(lo_area[-int(EQUILIBRIUM/SKIP):], n_chunks=10) for lo_area in lo_area_evol])
ax.errorbar(np.arange(I_MIN, I_MAX+1), lo_area_arr, yerr=lo_area_ste)
ax.set_xlabel("I")
ax.set_ylabel("Lo Area [%]")
ax.set_title("Lo Area % vs I for {}".format(which))
plt.savefig("{}lo_area_{}.png".format(PLOTS_DIR, which))
plt.close()

# plot composition vs I
fig, ax = plt.subplots()
comp_arr = np.array([res[1] for res in results])
comp_evolution_arr = [res[6] for res in results]
comp_ste_arr = np.array([[block_averaging([c[i] for c in comp_evolution[-int(EQUILIBRIUM/SKIP):]], n_chunks=10) for i in range(3)] for comp_evolution in comp_evolution_arr])
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 0], yerr=comp_ste_arr[:, 0], label=lipid)
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 1], yerr=comp_ste_arr[:, 1], label="DUPC")
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 2], yerr=comp_ste_arr[:, 2], label="CHOL")
ax.set_xlabel("I")
ax.set_ylabel("Composition [%]")
ax.set_title("Composition of the Lo domain for {}".format(which))
ax.legend()
plt.savefig("{}ave_comp_lo_{}_no_shift.png".format(PLOTS_DIR, which))
plt.close()

# plot composition vs I
fig, ax = plt.subplots()
comp_arr = np.array([res[2] for res in results])
comp_evolution_arr = [res[7] for res in results]
comp_ste_arr = np.array([[block_averaging([c[i] for c in comp_evolution[-int(EQUILIBRIUM/SKIP):]], n_chunks=10) for i in range(3)] for comp_evolution in comp_evolution_arr])
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 0], yerr=comp_ste_arr[:, 0], label=lipid)
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 1], yerr=comp_ste_arr[:, 1], label="DUPC")
ax.errorbar(np.arange(I_MIN, I_MAX+1), comp_arr[:, 2], yerr=comp_ste_arr[:, 2], label="CHOL")
ax.set_xlabel("I")
ax.set_ylabel("Composition [%]")
ax.set_title("Composition of the Ld domain for {}".format(which))
ax.legend()
plt.savefig("{}ave_comp_ld_{}_no_shift.png".format(PLOTS_DIR, which))
plt.close()

# plot Lo composition evolution for each I, use running averages
WINDOW = 100

for i, res in enumerate(results):
    fig, ax = plt.subplots()
    comp_arr = res[6]
    ax.plot(np.convolve(comp_arr[:, 0] - comp_arr[:, 0].mean(), np.ones(WINDOW)/WINDOW, mode="valid"), label=lipid, color="blue")
    ax.plot(np.convolve(comp_arr[:, 1] - comp_arr[:, 1].mean(), np.ones(WINDOW)/WINDOW, mode="valid"), label="DUPC", color="red")
    ax.plot(np.convolve(comp_arr[:, 2] - comp_arr[:, 2].mean(), np.ones(WINDOW)/WINDOW, mode="valid"), label="CHOL", color="green")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Delta composition [%]")
    ax.set_title("Lo composition for I={} for {}".format(i+I_MIN, which))
    ax.legend()
    plt.savefig("{}comp_evolution_lo_{}_{}.png".format(PLOTS_DIR, which, i+I_MIN))
    plt.close()

# plot Ld composition evolution for each I, use running averages
WINDOW = 100

for i, res in enumerate(results):
    fig, ax = plt.subplots()
    comp_arr = res[7]
    ax.plot(np.convolve(comp_arr[:, 0] - comp_arr[:, 0].mean(), np.ones(WINDOW)/WINDOW, mode="valid"), label=lipid, color="blue")
    ax.plot(np.convolve(comp_arr[:, 1] - comp_arr[:, 1].mean(), np.ones(WINDOW)/WINDOW, mode="valid"), label="DUPC", color="red")
    ax.plot(np.convolve(comp_arr[:, 2] - comp_arr[:, 2].mean(), np.ones(WINDOW)/WINDOW, mode="valid"), label="CHOL", color="green")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Delta composition [%]")
    ax.set_title("Ld composition for I={} for {}".format(i+I_MIN, which))
    ax.legend()
    plt.savefig("{}comp_evolution_ld_{}_{}.png".format(PLOTS_DIR, which, i+I_MIN))
    plt.close()

# plot mean overlap vs I
fig, ax = plt.subplots()
overlap_arr = [np.mean(res[4][-int(EQUILIBRIUM/SKIP):]) for res in results]
overlap_evolution_arr = [res[4] for res in results]
overlap_ste_arr = np.array([block_averaging(overlap_evolution[-int(EQUILIBRIUM/SKIP):], n_chunks=10) for overlap_evolution in overlap_evolution_arr])
ax.errorbar(np.arange(I_MIN, I_MAX+1), overlap_arr, yerr=overlap_ste_arr)
ax.set_xlabel("I")
ax.set_ylabel("Mean Overlap")
ax.set_title("Mean Overlap vs I for {}".format(which))
plt.savefig("{}mean_overlap_{}.png".format(PLOTS_DIR, which))
plt.close()
