import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

FILENAME = "new_model/analysis_df.csv"

WHICH_LIPIDS = ["DAPC", "DPPC"]
GAMMAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0]
RS = [100.0, 75.0, 50.0, 40.0, 30.0, 27.5, 25.0, 22.5, 20.0, 17.5, 15.0, 12.5, 10.0, 7.5, 5.0]
C0S = [0.0]
IS_MIDS = ["true", "false"]

# Load the data
df = pd.read_csv(FILENAME)
# turn the columns of the data into the correct types
df['which_lipid'] = df['which_lipid'].astype(str)
df['gamma'] = df['gamma'].astype(float)
df['lambda'] = df['lambda'].astype(float)
df['R'] = df['R'].astype(float)
df['c0'] = df['c0'].astype(float)
df['is_mid'] = df['is_mid'].astype(bool)
df['coupling_ratio'] = df['coupling_ratio'].astype(float)
df['domain_fit_ᵤ'] = df['domain_fit_ᵤ'].astype(float)
df['domain_fit_ₗ'] = df['domain_fit_ₗ'].astype(float)

# change column name is_mid to uniform
# NOTE: there was a bug such that is_mid == true was actually is_mid == false and vice versa!!!
df = df.rename(columns={"is_mid": "uniform"})

def plot_violin(ax, dataset, property, hue):
    sns.violinplot(data=dataset, x="R", y=property, hue=hue, split=True, inner="stick", palette="summer", ax=ax, linewidth=0.3, density_norm='count', scale='width')

def plot_coupling_ratio_vs_R(which_lipid, lambda_, gamma):
    # filter out the data by which_lipid == DAPC
    dapc_df = df[df['which_lipid'] == which_lipid]
    # fliter lambda == 0.3
    dapc_df = dapc_df[dapc_df['lambda'] == lambda_]
    # filter gamma == 2.0
    dapc_df = dapc_df[dapc_df['gamma'] == gamma]
    fig, ax = plt.subplots()
    plot_violin(ax, dapc_df, 'coupling_ratio', 'uniform')
    ax.set_xlabel("R")
    ax.set_ylabel("Coupling Ratio")
    plt.legend(title="Uniform", loc='upper left')
    #plt.xlim(0, 100)
    ax.set_ylim(0, 1)
    plt.title(r"{}, $\lambda = {}$, $\gamma = {}$".format(which_lipid, lambda_, gamma))
    plt.savefig("plots/coupling_ratio_vs_R_{}_lambda_{}_gamma_{}.png".format(which_lipid, lambda_, gamma), dpi=300)
    plt.close()

plot_coupling_ratio_vs_R("DAPC", 0.3, 2.0)
plot_coupling_ratio_vs_R("DPPC", 0.3, 2.0)
plot_coupling_ratio_vs_R("DAPC", 0.3, 1.0)
plot_coupling_ratio_vs_R("DPPC", 0.3, 1.0)
plot_coupling_ratio_vs_R("DAPC", 0.3, 0.5)
plot_coupling_ratio_vs_R("DPPC", 0.3, 0.5)
plot_coupling_ratio_vs_R("DAPC", 0.3, 4.0)
plot_coupling_ratio_vs_R("DPPC", 0.3, 4.0)
plot_coupling_ratio_vs_R("DAPC", 0.5, 2.0)
plot_coupling_ratio_vs_R("DPPC", 0.5, 2.0)
plot_coupling_ratio_vs_R("DAPC", 0.5, 1.0)
plot_coupling_ratio_vs_R("DPPC", 0.5, 1.0)
plot_coupling_ratio_vs_R("DAPC", 0.5, 0.5)
plot_coupling_ratio_vs_R("DPPC", 0.5, 0.5)
plot_coupling_ratio_vs_R("DAPC", 0.5, 4.0)
plot_coupling_ratio_vs_R("DPPC", 0.5, 4.0)
plot_coupling_ratio_vs_R("DAPC", 0.1, 2.0)
plot_coupling_ratio_vs_R("DPPC", 0.1, 2.0)
plot_coupling_ratio_vs_R("DAPC", 0.1, 1.0)
plot_coupling_ratio_vs_R("DPPC", 0.1, 1.0)
plot_coupling_ratio_vs_R("DAPC", 0.1, 0.5)
plot_coupling_ratio_vs_R("DPPC", 0.1, 0.5)
plot_coupling_ratio_vs_R("DAPC", 0.1, 4.0)
plot_coupling_ratio_vs_R("DPPC", 0.1, 4.0)
