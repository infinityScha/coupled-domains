import numpy as np
import matplotlib.pyplot as plt
import os
import MDAnalysis as mda
from MDAnalysis.transformations import nojump
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy import interpolate
from utils import periodic_griddata as pg
from scipy.stats import gaussian_kde
from tqdm import tqdm

# remove warnings
import warnings
warnings.filterwarnings("ignore")

# Parameters
sim_dir = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/CURVATURE_BONUSPACK/"
grofile = "DAPC_50_CURVED_nw.gro"
xtcfile = "DAPC_50_CURVED.xtc"

EPS = 12
MIN_SAMPLES = 6

# Load trajectory
u = mda.Universe(os.path.join(sim_dir, grofile), os.path.join(sim_dir, xtcfile))

# add nojump transformation
nj = nojump.NoJump()

# define transformation that wraps the trajectory on the xy plane
def wrap_xy(ts):
    ts = nj(ts)
    pos = ts.positions
    pos[:, :2] = pos[:, :2] % u.dimensions[:2]
    ts.positions = pos
    return ts

# add the transformation to the trajectory
u.trajectory.add_transformations(wrap_xy)

DAPCs = u.select_atoms("resname DAPC").residues
DUPCs = u.select_atoms("resname DUPC").residues
CHOLs = u.select_atoms("resname CHOL").residues

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

def flatten(heads, arc_length_f):
    # flatten the sines to the arc length
    pos = heads.positions
    x_flat = np.vectorize(arc_length_f)(pos[:, 0])
    pos[:, 0] = x_flat
    pos[:, 2] = 0.0
    return pos

def fit_the_curve(which=0):
    leaflet = leaflet1 if which == 0 else leaflet2 
    heads = mda.AtomGroup([res.atoms[0] for res in leaflet])
    x = heads.positions[:, 0]
    z = heads.positions[:, 2]
    period = u.dimensions[0]
    mid = z.mean()
    # fit position to a sine function, with the period and midline as fixed parameters
    def sine(x, A):
        return A * np.sin(2 * np.pi * x / period) + mid
    # fit the data
    popt, _ = curve_fit(sine, x, z)
    # arc length function over the sines
    def sine_arc_length(x):
        return arc_length(sine, x, popt)
    return heads, sine_arc_length

def dbscan_clus_(heads, pos, period):
    is_lo = [head.resname != "DUPC" for head in heads]
    kdt = cKDTree(pos[is_lo, :2], boxsize=np.array([period, u.dimensions[1]])+0.001)
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="precomputed")
    X = kdt.sparse_distance_matrix(kdt, EPS)
    dbscan.fit(X)
    which_domain = np.zeros(len(pos), dtype=int)-1
    # find the largest cluster
    domains, counts = np.unique(dbscan.labels_[dbscan.labels_ != -1], return_counts=True)
    max_cluster = domains[np.argmax(counts)]
    which_domain[is_lo] = dbscan.labels_
    which_domain[which_domain != max_cluster] = -1
    which_domain[which_domain == max_cluster] = 0
    xx = np.arange(0, period, 1.0)
    yy = np.arange(0, u.dimensions[1], 1.0)
    # extend the grid by +- 20.0 to avoid edge effects
    x_grid, y_grid = np.meshgrid(xx, yy)
    z_grid = pg.periodic_griddata(pos[:, :2], which_domain, (x_grid, y_grid), method="nearest", period=[period, u.dimensions[1]])
    # do the same with composition
    if np.random.rand() < 0.1:
        which_domain2 = np.zeros(len(pos), dtype=int)
        is_dupc = [head.resname == "DUPC" for head in heads]
        is_dapc = [head.resname == "DAPC" for head in heads]
        is_chol = [head.resname == "CHOL" for head in heads]
        which_domain2[is_dupc] = 0
        which_domain2[is_dapc] = 1
        which_domain2[is_chol] = 2
        z_grid2 = pg.periodic_griddata(pos[:, :2], which_domain2, (x_grid, y_grid), method="nearest", period=[period, u.dimensions[1]])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(z_grid, cmap="viridis", extent=(0, period, 0, u.dimensions[1]), origin="lower")
        ax[1].imshow(z_grid2, cmap="viridis", extent=(0, period, 0, u.dimensions[1]), origin="lower", vmin=0, vmax=2)
        ax[0].set_title("Domain")
        ax[1].set_title("Composition")
        ax[1].plot([], [], "s", label="DUPC", color=plt.cm.viridis(0.0))
        ax[1].plot([], [], "s", label="DAPC", color=plt.cm.viridis(0.5))
        ax[1].plot([], [], "s", label="CHOL", color=plt.cm.viridis(1.0))
        # describe the colors in a legend
        ax[1].legend()
        plt.savefig("simulations_analysis/plots/heads_domains.png")
        plt.close(fig)
    return z_grid, 1 #max(dbscan.labels_)

def domains_overlap():
    # flatten the curve
    heads1, arch_length_f1 = fit_the_curve(0)
    heads2, arch_length_f2 = fit_the_curve(1)
    arch_length_f = lambda x: (arch_length_f1(x) + arch_length_f2(x)) / 2
    pos1 = flatten(heads1, arch_length_f)
    pos2 = flatten(heads2, arch_length_f)
    period = arch_length_f(u.dimensions[0])
    grid1, n1 = dbscan_clus_(heads1, pos1, period)
    grid2, n2 = dbscan_clus_(heads2, pos2, period)
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
    return overlap_arr, area_arr



total_overlap_arr = []
total_area_arr = []
for ts in tqdm(u.trajectory[100:-1500:50], total=len(u.trajectory[100:-1500:50])):
    if leaflet1 == [] or leaflet2 == [] or ts.frame % 100 == 0:
        update_leaflets()
    overlap_arr, area_arr = domains_overlap()
    total_overlap_arr.append(overlap_arr)
    total_area_arr.append(area_arr)
total_overlap_arr = np.concatenate(total_overlap_arr)
total_area_arr = np.concatenate(total_area_arr)
# plot the overlap distribution
fig, ax = plt.subplots()
ax.hist(total_overlap_arr, bins=np.linspace(0.0, 1.0, 10, endpoint=True), density=True, alpha=0.5, label="Overlap")
plt.savefig("simulations_analysis/plots/overlap_distribution.png")
plt.close(fig)
# time evolution of the overlap
fig, ax = plt.subplots()
ax.plot(total_overlap_arr)
ax.set_xlabel("Frame")
ax.set_ylabel("Overlap")
plt.savefig("simulations_analysis/plots/overlap_evolution.png")
plt.close(fig)
# correlate the overlap with the area by a 2d kde
x = total_overlap_arr
y = total_area_arr
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
fig, ax = plt.subplots()
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
zi = interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method="cubic") 
ax.contourf(xi, yi, zi, levels=100, cmap="viridis")
ax.set_xlabel("Overlap area")
ax.set_ylabel("Area")
plt.savefig("simulations_analysis/plots/overlap_area.png")
plt.close(fig)
