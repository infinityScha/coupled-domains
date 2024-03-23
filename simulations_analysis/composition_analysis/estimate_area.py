import numpy as np
import MDAnalysis as mda

coex = np.array([37.5, 37.5, 25])
lo = np.array([59, 8, 33])
ld = np.array([19, 62, 19])

folder = '/run/media/itay/Elements/curved-domains/SINGLEPHASE/DMPC'
lipid = 'dmpc'

min_x, min_y = 0.0, 1000
for x in np.linspace(0.0, 1.0, 1000):
    y = np.sqrt(np.sum((coex - x*lo - (1 - x)*ld)**2))
    if y < min_y:
        print(y)
        min_y = y
        min_x = x
print(min_x,coex - min_x*lo - (1 - min_x)*ld )

# get area
for phase in ['lo', 'ld']:
    print(lipid+'_'+phase)
    u = mda.Universe('{}/{}_{}.gro'.format(folder, lipid, phase), '{}/{}_{}.xtc'.format(folder, lipid, phase))
    n_lipids = len(u.select_atoms('not name W WF').residues)
    a = [2*u.dimensions[0]*u.dimensions[1]/n_lipids for ts in u.trajectory]
    print(np.mean(a))