import numpy as np
import matplotlib.pyplot as plt
import os
import MDAnalysis as mda
from MDAnalysis.transformations import nojump
from scipy.optimize import curve_fit
from scipy import interpolate
from utils import periodic_griddata as pg
from scipy.stats import gaussian_kde
from tqdm import tqdm
from freud.density import GaussianDensity

# remove warnings
import warnings
warnings.filterwarnings("ignore")

which = "DAPC"

# Parameters
sim_dir = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/FORCEDFLAT/{}/075/".format(which)


grofile = "{}_FORCEFLAT_075_nw.gro".format(which)
xtcfile = "{}_FORCEFLAT_075.xtc".format(which)

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

EPS_LOS = [7, 8, 9, 10, 11, 12, 13, 14]
RATIOS_LOS = [0.3, 0.4, 0.5]
EPS_LDS = [7, 8, 9, 10, 11, 12, 13, 14]
RATIOS_LDS = [0.3, 0.4, 0.5]

# merge all combinations of parameters
PARAMS = []
for eps_lo in EPS_LOS:
    for ratio_lo in RATIOS_LOS:
        for eps_ld in EPS_LDS:
            for ratio_ld in RATIOS_LDS:
                PARAMS.append((eps_lo, int(eps_lo*ratio_lo), eps_ld, int(eps_ld*ratio_ld)))

for param in PARAMS:
    EPS_LO, MIN_SAMPLES_LO, EPS_LD, MIN_SAMPLES_LD = param

    lo_arr = []
    comp_arr_lo = []
    comp_arr_ld = []

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
    
    def _clus_helper(pos_relevant, period, eps, min_samples):
        kdt = cKDTree(pos_relevant[:, :2], boxsize=np.array([period, u.dimensions[1]])+0.001)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        X = kdt.sparse_distance_matrix(kdt, eps)
        dbscan.fit(X)
        return dbscan.labels_

    def dbscan_clus_(heads, pos, period):
        is_dupc = np.array([head.resname == "DUPC" for head in heads])
        is_dapc = np.array([head.resname == which for head in heads])
        is_chol = np.array([head.resname == "CHOL" for head in heads])
        box = (period, u.dimensions[1])
        rounded_box = (round(period), round(u.dimensions[1]))
        # get gaussian density of each head
        gd = GaussianDensity(width=rounded_box, r_max=50.0, sigma = 12.0)
        # get DUPC density
        gd.compute(system=(box, pos[is_dupc]))
        dupc_density = gd.density.copy().reshape(rounded_box)
        # get DAPC density
        gd.compute(system=(box, pos[np.invert(is_dupc)]))
        rest_density = gd.density.copy().reshape(rounded_box)
        # normalize the densities
        tot_density = dupc_density + rest_density
        dupc_density /= tot_density
        # get the mean density of each head
        z_grid = (dupc_density < 0.5).T
        xx = np.linspace(0, period, tot_density.shape[0])
        yy = np.linspace(0, u.dimensions[1], tot_density.shape[1])
        x_grid, y_grid = np.meshgrid(xx, yy)
        #z_grid = pg.periodic_griddata(pos[:, :2], phase, (x_grid, y_grid), method="nearest", period=[period, u.dimensions[1]])
        pos_int = pos.astype(int)
        pos_int[:, 0] = pos_int[:, 0] % round(u.dimensions[0])
        pos_int[:, 1] = pos_int[:, 1] % round(u.dimensions[1])
        in_lo = z_grid[pos_int[:, 1], pos_int[:, 0]]
        lo_arr.append(z_grid.mean())
        dapc_lo = is_dapc[in_lo]
        dupc_lo = is_dupc[in_lo]
        chol_lo = is_chol[in_lo]
        comp_arr_lo.append([dapc_lo.mean(), dupc_lo.mean(), chol_lo.mean()])
        dapc_ld = is_dapc[np.invert(in_lo)]
        dupc_ld = is_dupc[np.invert(in_lo)]
        chol_ld = is_chol[np.invert(in_lo)]
        comp_arr_ld.append([dapc_ld.mean(), dupc_ld.mean(), chol_ld.mean()])
        # do the same with composition
        if np.random.rand() < 1.0:
            which_domain2 = np.zeros(len(pos), dtype=int)
            which_domain2[is_dupc] = 0
            which_domain2[is_dapc] = 1
            which_domain2[is_chol] = 2
            z_grid2 = pg.periodic_griddata(pos[:, :2], which_domain2, (x_grid, y_grid), method="nearest", period=[period, u.dimensions[1]])
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(z_grid, cmap="viridis", extent=(0, period, 0, u.dimensions[1]), origin="lower")
            axs[0].scatter(pos[is_dupc, 0], pos[is_dupc, 1], color="red", s=1)
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
    
    def fit_the_curve(which=0):
        leaflet = leaflet1 if which == 0 else leaflet2 
        heads = mda.AtomGroup([res.atoms[-1] for res in leaflet])
        x = heads.positions[:, 0]
        z = heads.positions[:, 2] % u.dimensions[2]
        period = u.dimensions[0]
        # arc length function over the sines
        def sine_arc_length(x):
            return x
        return heads, sine_arc_length

    def flatten(heads, arc_length_f):
        # flatten the sines to the arc length
        pos = heads.positions
        x_flat = np.vectorize(arc_length_f)(pos[:, 0])
        pos[:, 0] = x_flat
        pos[:, 2] = 0.0
        return pos

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
    for ts in tqdm(u.trajectory[-500::10]):#[-500::10], total=len(u.trajectory[-500::10])):
        update_leaflets()
        overlap_arr, area_arr = domains_overlap()
        total_overlap_arr.append(overlap_arr)
        total_area_arr.append(area_arr)
    total_overlap_arr = np.concatenate(total_overlap_arr)
    total_area_arr = np.concatenate(total_area_arr)
    print("Parameters: EPS_LO {}, MIN_SAMPLES_LO {}, EPS_LD {}, MIN_SAMPLES_LD {}".format(EPS_LO, MIN_SAMPLES_LO, EPS_LD, MIN_SAMPLES_LD))  
    print("Mean LO area: ", np.mean(lo_arr))
    print("Mean LO composition: ", (np.mean(comp_arr_lo, axis=0)*1000).astype(int)/10)
    print("Mean LD composition: ", (np.mean(comp_arr_ld, axis=0)*1000).astype(int)/10)
    lo_arr = []