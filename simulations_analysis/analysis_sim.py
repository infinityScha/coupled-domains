import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import *

#import warnings
#warnings.filterwarnings("ignore")

LIPIDS = ['DAPC', 'DPPC', 'DAPC_LIN'] #['DAPC', 'DAPC_AS', 'DAPC_LIN', 'DPPC', 'DPPC_AS']
CSV_DIR = 'simulations_analysis/csv'

RESULTS_DIR = 'simulations_analysis/results_sim'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def morph_profile(profile):
    """
    Morph the profile to a standard format.

    Args:
        profile (pandas.DataFrame): The profile to morph.

    Returns:
        pandas.DataFrame: The morphed profile.
    """
    # extract only x, DUPC_leaflet1 and DUPC_leaflet2
    profile = profile[['x', 'DUPC_leaflet1', 'DUPC_leaflet2']]
    # rename the columns
    profile.columns = ['x', 'cᵤ', 'cₗ']
    profile.loc[:, 'cᵤ'] *= 2.0
    profile.loc[:, 'cₗ'] *= 2.0
    profile.loc[:, 'cᵤ'] -= 1.0
    profile.loc[:, 'cₗ'] -= 1.0
    profile.loc[:, 'x'] /= 10.0
    # strect the data so that the maximal c is at 1.0 and the minimal c is at -1.0
    cu_max = profile['cᵤ'].max()
    cl_max = profile['cₗ'].max()
    cu_min = profile['cᵤ'].min()
    cl_min = profile['cₗ'].min()
    cu_mid = (cu_max + cu_min) / 2
    cl_mid = (cl_max + cl_min) / 2
    cu_diff = cu_max - cu_min
    cl_diff = cl_max - cl_min
    profile.loc[:, 'cᵤ'] = 2.0*(profile['cᵤ'] - cu_mid) / cu_diff
    profile.loc[:, 'cₗ'] = 2.0*(profile['cₗ'] - cl_mid) / cl_diff
    return profile

def run_analyses(filenames, analyses):
    """
    Run multiple analyses on the given data.

    Args:
        filenames (list): List of filenames.
        analyses (list): List of analysis objects.

    Returns:
        None
    """
    # run the analyses on the profiles
    for filename in filenames:
        profile = pd.read_csv(filename)
        profile = morph_profile(profile)
        for analysis in analyses:
            analysis.run(profile)

for lipid in LIPIDS:
    print(lipid)
    # find all csv files in the test_params directory
    csv_files = [f'{CSV_DIR}/{lipid}/{f}' for f in os.listdir(f'{CSV_DIR}/{lipid}')]
    # make pandas array with filename and the params stated in the filenames
    y_dims = np.array([f[-6:-4] for f in csv_files], dtype=int)
    argsort_y_dims = np.argsort(y_dims)
    
    # make a list of the analyses to run
    reg = Registration()
    width_u = InterfaceWidth(True)
    width_l = InterfaceWidth(False)

    # run the analyses on the profiles
    run_analyses(csv_files, [reg, width_u, width_l])

    reg, width_u, width_l = reg.results(), width_u.results(), width_l.results()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(lipid)
    for j, (prop, prop_label, ylim) in enumerate([(reg, 'Registration', (0.0, 1.0)), (width_u, 'Interface Width (upper)', (1.0, 2.0)), (width_l, 'Interface Width (lower)', (1.0, 2.0))]):
        prop = np.array(prop)
        ax[j].set_title(prop_label)
        ax[j].set_xlabel('y_dim')
        ax[j].set_ylabel(prop_label)
        ax[j].set_ylim(ylim)
        ax[j].plot(y_dims[argsort_y_dims], prop[argsort_y_dims])
        # save the data
        folder = f'{RESULTS_DIR}/{lipid}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savetxt(f'{folder}/{prop_label}.csv', np.array([y_dims, prop]).T, delimiter=',', header=f'y_dim,{prop_label}')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{lipid}.png')
    plt.close()