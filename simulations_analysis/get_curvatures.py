import numpy as np
import MDAnalysis as mda
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

u = None

def get_mean_x_dim():
    """
    Get the mean x dimension of the simulation box.
    """
    global u
    x_dim_arr = [u.dimensions[0] for ts in u.trajectory[-5000:]]
    return np.mean(x_dim_arr)

def get_mean_y_dim():
    """
    Get the mean y dimension of the simulation box.
    """
    global u
    y_dim_arr = [u.dimensions[1] for ts in u.trajectory[-5000:]]
    return np.mean(y_dim_arr)

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


def get_curvatures(which, I_MIN, I_MAX):
    """
    Get the curvatures of the leaflets of the simulation.
    """
    global u

    # load the forced flat simulation
    folder = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/FORCEDFLAT/{}/075/".format(which)
    grofile = "{}/{}_FORCEFLAT_075_nw.gro".format(folder, which)
    xtcfile = "{}/{}_FORCEFLAT_075.xtc".format(folder, which)
    u = mda.Universe(grofile, xtcfile)
    x_ff = get_mean_x_dim()
    y_ff = get_mean_y_dim()

    folder = "/mnt/635d0d06-878a-4beb-a0b2-6ced0caa25db/SIMULATIONS/curved-domains/trajectories/{}/".format(which)
    res_arr = ["which i Rmax"]
    for i in range(I_MIN, I_MAX+1):
        grofile = "{}_{}_nw.gro".format(which, i)
        xtcfile = "{}_{}.xtc".format(which, i)
        u = mda.Universe("{}/{}".format(folder, grofile), "{}/{}".format(folder, xtcfile))
        x = get_mean_x_dim()
        y = get_mean_y_dim()
        x_ff_fixed =  x_ff * y_ff / y
        print("i = {}, x_ff = {}, x = {}".format(i, x_ff_fixed, x))
        c = find_curvature(x_ff_fixed, x)
        res_arr.append("{} {} {}".format(which, i, c))
    print("\n".join(res_arr))


#get_curvatures("DAPC", 23, 33)
get_curvatures("DPPC", 23, 33)