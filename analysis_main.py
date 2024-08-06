import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from analysis import *
import re

RESULTS_DIR = 'analysis_results'


if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# find all csv files in the test_params directory
csv_files = ['test_params/'+f for f in os.listdir('test_params') if f.endswith('.csv')]
# make pandas array with filename and the params stated in the filenames
# split by '_' and '/'
params = [re.split('_|/', f)[2:5] for f in csv_files]
# make a dataframe with the params
df = pd.DataFrame(params, columns=['gamma', 'lambda', 'gamma_l'], dtype=float)
df = pd.DataFrame(params, columns=['gamma', 'lambda', 'gamma_l'], dtype=float)

# make a list of the analyses to run
reg = Registration()
width_u = InterfaceWidth(True)
width_l = InterfaceWidth(False)

# run the analyses on the profiles
run_analyses(csv_files, [reg, width_u, width_l])

reg, width_u, width_l = reg.results(), width_u.results(), width_l.results()

gammas = np.sort(df['gamma'].unique())
lambdas = np.sort(df['lambda'].unique())
gamma_ls = np.sort(df['gamma_l'].unique())


# make a plot of the results with variation in one parameter and the other two fixed
gammas_lambdas = [(g, l) for g in gammas for l in lambdas]
gammas_gamma_ls = [(g, gl) for g in gammas for gl in gamma_ls]
lambdas_gamma_ls = [(l, gl) for l in lambdas for gl in gamma_ls]

for (ab, a_label, b_label, free_param, fp_label) in [(gammas_lambdas, 'gamma', 'lambda', gamma_ls, 'gamma_l'), (gammas_gamma_ls, 'gamma', 'gamma_l', lambdas, 'lambda'), (lambdas_gamma_ls, 'lambda', 'gamma_l', gammas, 'gamma')]:
    fig, ax = plt.subplots(len(ab), 3, figsize=(15, 5*len(ab)))
    fig.suptitle(f'Variation in {fp_label}')
    for i, (a, b) in enumerate(ab):
        for j, (prop, prop_label, ylim) in enumerate([(reg, 'Registration', [0, 1]), (width_u, 'Upper interface width', [1, 2.5]), (width_l, 'Lower interface width', [1, 2])]):
            mask = (df[a_label] == a) & (df[b_label] == b)
            masked_prop = prop[mask]
            masked_df = df[mask][fp_label]
            sorted_indices = np.argsort(masked_df)
            prop_sorted = masked_prop[sorted_indices]
            free_param_sorted = masked_df.values[sorted_indices]
            ax[i, j].set_title(f'{a_label} = {a}, {b_label} = {b}')
            ax[i, j].set_xlabel(fp_label)
            ax[i, j].set_ylabel(prop_label)
            ax[i, j].set_ylim(ylim)
            ax[i, j].plot(free_param_sorted, prop_sorted, 'o')
            # save the data
            folder = f'{RESULTS_DIR}/{fp_label}_variation'
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.savetxt(f'{folder}/{prop_label}_{a}_{b}.csv', np.array([free_param_sorted, prop_sorted]).T, delimiter=',', header=f'{fp_label},{prop_label}')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{fp_label}_variation.png')
    plt.close()



