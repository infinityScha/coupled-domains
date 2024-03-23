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

which = "DAPC"

# Parameters
sim_dir = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/{}/".format(which)

I_MIN = 23
I_MAX = 39

EPS_LO = 14
MIN_SAMPLES_LO = 5
EPS_LD = 14
MIN_SAMPLES_LD = 4

START = 24500
SKIP = 10

lo_arr = []
comp_arr_lo = []
comp_arr_ld = []

# define default colors for the plots to have gradual color
colors = plt.cm.viridis(np.linspace(0, 1, np.abs(I_MAX-I_MIN)+1))


total_A_arr = []

fig, ax = plt.subplots()
for i in range(I_MIN, I_MAX+1):
    grofile = "{}_{}_nw.gro".format(which, i)
    xtcfile = "{}_{}.xtc".format(which, i)

    # Load trajectory
    u = mda.Universe(os.path.join(sim_dir, grofile), os.path.join(sim_dir, xtcfile))

    # add nojump transformation
    nj = nojump.NoJump()

    # define transformation that wraps the trajectory on the xy plane
    #def wrap_xy(ts):
    #    ts = nj(ts)
    #    pos = ts.positions
    #    pos[:, :2] = pos[:, :2] % u.dimensions[:2]
    #    ts.positions = pos
    #    return ts

    # add the transformation to the trajectory
    #u.trajectory.add_transformations(wrap_xy)

    leaflet1, leaflet2 = [], []

    def update_leaflets():
        global leaflet1, leaflet2
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

    def fit_the_curve(which=0):
        leaflet = leaflet1 if which == 0 else leaflet2 
        resnames = [res.resname for res in leaflet]
        pos = []
        for res in leaflet:
            if res.resname == "CHOL":
                pos.append(res.atoms[-1].position)
            else:
                tail2 = res.atoms[-1].position
                tail1 = res.atoms[-6 if which == "DAPC" else -5].position
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
        # plot for visual inspection
        #fig, ax = plt.subplots()
        #ax.plot(x, z, "o", label="Data")
        #ax.scatter(x, sine(x, *popt), label="Fit", color="red")
        #ax.set_xlabel("x [A]")
        #ax.set_ylabel("z [A]")
        #ax.legend()
        #plt.savefig("simulations_analysis/plots/fit_{}.png".format(which))
        #plt.close(fig)
        # arc length function over the sines
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
        is_dapc = np.array([resname == which for resname in resnames])
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
        # do the same with composition
        if np.random.rand() < 0.05:
            which_domain2 = np.zeros(len(pos), dtype=int)
            which_domain2[is_dupc] = 0
            which_domain2[is_dapc] = 1
            which_domain2[is_chol] = 2
            z_grid2 = pg.periodic_griddata(pos[:, :2], which_domain2, (x_grid, y_grid), method="nearest", period=[period, u.dimensions[1]])
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            plt.colorbar(axs[0].imshow(z_grid, cmap="viridis", extent=(0, period, 0, u.dimensions[1]), origin="lower"))
            axs[1].imshow(z_grid2, cmap="viridis", extent=(0, period, 0, u.dimensions[1]), origin="lower", vmin=0, vmax=2)
            axs[0].set_title("Domain")
            axs[1].set_title("Composition")
            axs[1].plot([], [], "s", label="DUPC", color=plt.cm.viridis(0.0))
            axs[1].plot([], [], "s", label= which, color=plt.cm.viridis(0.5))
            axs[1].plot([], [], "s", label="CHOL", color=plt.cm.viridis(1.0))
            # describe the colors in a legend
            axs[1].legend()
            plt.savefig("simulations_analysis/plots/heads_domains.png")
            plt.close(fig)        
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
    for ts in tqdm(u.trajectory[START::SKIP], total=len(u.trajectory[START::SKIP]), desc="{}_{}".format(which, i)):
        update_leaflets()
        overlap_arr, area_arr, A = domains_overlap()
        total_overlap_arr.append(overlap_arr)
        total_area_arr.append(area_arr)
        A_arr.append(A)
    total_overlap_arr = np.concatenate(total_overlap_arr)
    total_area_arr = np.concatenate(total_area_arr)
    total_A_arr.append(np.mean(A_arr))
    # plot the overlap distribution
    #fig, ax = plt.subplots()
    #ax.hist(total_overlap_arr, bins=np.linspace(0.0, 1.0, 10, endpoint=True), density=True, alpha=0.5, label="Overlap")
    #plt.savefig("simulations_analysis/plots/overlap_distribution.png")
    #plt.close(fig)
    # time evolution of the overlap
    ax.plot(total_overlap_arr, label="{}".format(i), color=colors[np.abs(i-I_MIN)])
    # remove all but the last 5 microseconds of the lo_arr and comp_arrs
    lo_arr = lo_arr[-int(5000/SKIP):]
    comp_arr_lo = comp_arr_lo[-int(5000/SKIP):]
    comp_arr_ld = comp_arr_ld[-int(5000/SKIP):]
    print("Mean LO area: ", np.mean(lo_arr))
    print("Mean LO composition: ", (np.mean(comp_arr_lo, axis=0)*1000).astype(int)/10)
    print("Mean LD composition: ", (np.mean(comp_arr_ld, axis=0)*1000).astype(int)/10)
    lo_arr = []
ax.set_xlabel("Frame")
ax.set_ylabel("Overlap")
ax.set_ylim(0, 1)
ax.legend()
plt.savefig("simulations_analysis/plots/overlap_evolution_{}.png".format(which))
plt.close(fig)


total_A_arr = np.array(total_A_arr)

# plot the curvature vs I
fig, ax = plt.subplots()
ax.plot(np.arange(I_MIN, I_MAX+1), total_A_arr)
ax.set_xlabel("I")
ax.set_ylabel("Max C [nm^-1]")
plt.savefig("simulations_analysis/plots/ave_curvature_{}.png".format(which))