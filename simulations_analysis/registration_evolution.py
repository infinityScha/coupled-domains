import numpy as np
import matplotlib.pyplot as plt
import os
import MDAnalysis as mda
from MDAnalysis.transformations import nojump
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy import interpolate
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
MIN_SAMPLES = 7

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

def split_residues_by_leaflet(residues):
    leaflet1 = []
    leaflet2 = []
    for res in residues:
        pos1, pos2 = res.atoms.positions[0, 2], res.atoms.positions[-1, 2]
        if pos1 < pos2:
            leaflet1.append(res)
        else:
            leaflet2.append(res) 
    return leaflet1, leaflet2

def arc_length(f, x, args=()):
    xs = np.arange(0, x, 1.0)
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
    leaflet = split_residues_by_leaflet(u.residues)[which] 
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
    is_dupc = [head.resname == "DUPC" for head in heads]
    is_dapc = [head.resname == "DAPC" for head in heads]
    is_chol = [head.resname == "CHOL" for head in heads]
    kdt = cKDTree(pos[is_lo, :2], boxsize=np.array([period, u.dimensions[1]])+0.001)
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="precomputed")
    X = kdt.sparse_distance_matrix(kdt, EPS)
    dbscan.fit(X)
    which_domain = np.zeros(len(pos), dtype=int)-1
    #which_domain[is_lo] = dbscan.labels_
    which_domain[is_dupc] = 0
    which_domain[is_dapc] = 1
    which_domain[is_chol] = 2
    xx = np.arange(0, period, 2.5)
    yy = np.arange(0, u.dimensions[1], 2.5)
    x_grid, y_grid = np.meshgrid(xx, yy)
    z_grid = interpolate.griddata(pos[:, :2], which_domain, (x_grid, y_grid), method="nearest")
    return z_grid, max(dbscan.labels_)

def domains_overlap():
    # flatten the curve
    heads1, arch_length_f1 = fit_the_curve(0)
    heads2, arch_length_f2 = fit_the_curve(1)
    arch_length_f = lambda x: (arch_length_f1(x) + arch_length_f2(x)) / 2
    pos1 = flatten(heads1, arch_length_f)
    pos2 = flatten(heads2, arch_length_f)
    grid1, n1 = dbscan_clus_(heads1, pos1, arch_length_f(u.dimensions[0]))
    grid2, n2 = dbscan_clus_(heads2, pos2, arch_length_f(u.dimensions[0]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(grid1, cmap="viridis")
    ax[1].imshow(grid2, cmap="viridis")
    ax[0].set_title("Leaflet 1")
    ax[1].set_title("Leaflet 2")
    plt.savefig("simulations_analysis/plots/heads_domains.png")
    plt.close()
    # calculate the overlap
    overlap_arr = np.zeros((n1))
    area_arr = np.zeros(n1)
    for i in range(n1):
        area_i = (grid1 == i).sum()
        area_arr[i] = area_i
        for j in range(n2):
            overlap_area = ((grid1 == i) & (grid2 == j)).sum()
            overlap_arr[i] += overlap_area
        overlap_arr[i] /= area_i
    return overlap_arr, area_arr


total_overlap_arr = []
total_area_arr = []
for ts in tqdm(u.trajectory[5000:5150], total=150):
    overlap_arr, area_arr = domains_overlap()
    total_overlap_arr.append(overlap_arr)
    total_area_arr.append(area_arr)
total_overlap_arr = np.concatenate(total_overlap_arr)
total_area_arr = np.concatenate(total_area_arr)
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
plt.close()
