import numpy as np
import MDAnalysis as mda
import warnings
warnings.filterwarnings('ignore')

coex = np.array([37.5, 37.5, 25])
COMP_LO = {"DPPC": [62, 2, 36], "DAPC": [61, 4, 35]}
COMP_LD = {"DPPC": [4, 85, 11], "DAPC": [2, 88, 10]}

folder = '/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/SINGLEPHASE'
lipid = 'DPPC'

lo, ld = np.array(COMP_LO[lipid]), np.array(COMP_LD[lipid])

min_x, min_y = 0.0, 1000
for x in np.linspace(0.0, 1.0, 1000):
    y = np.sqrt(np.sum((coex - x*lo - (1 - x)*ld)**2))
    if y < min_y:
        print(y)
        min_y = y
        min_x = x
print(min_x,coex - min_x*lo - (1 - min_x)*ld )

areas = list()
# get area
for phase in ['lo', 'ld']:
    print(lipid.lower()+'_'+phase)
    u = mda.Universe('{}/{}_{}.gro'.format(folder, lipid.lower(), phase), '{}/{}_{}.xtc'.format(folder, lipid.lower(), phase))
    n_lipids = len(u.select_atoms('not name W WF').residues)
    a = [2*u.dimensions[0]*u.dimensions[1]/n_lipids for ts in u.trajectory[2000:]]
    print(np.mean(a))
    areas.append(np.mean(a))
tot_lo_area = areas[0]*min_x
tot_ld_area = areas[1]*(1-min_x)
tot_area = tot_lo_area + tot_ld_area
tot_lo_area /= tot_area
tot_ld_area /= tot_area
print(tot_lo_area, tot_ld_area)