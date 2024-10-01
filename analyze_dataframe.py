import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import beta
from scipy.optimize import minimize

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


def _beta_mixture_log_likelihood(params, data):
    w1, alpha1, beta1, alpha2, beta2 = params
    w2 = 1 - w1  # Since weights must sum to 1
    
    # Calculate the log-likelihood of the mixture
    likelihood1 = w1 * beta.pdf(data, alpha1, beta1)
    likelihood2 = w2 * beta.pdf(data, alpha2, beta2)
    
    # Total likelihood is the sum of the two
    total_likelihood = likelihood1 + likelihood2
    log_likelihood = np.sum(np.log(total_likelihood))
    
    # Return negative log-likelihood because we want to minimize
    return -log_likelihood


def _fit_bi_beta_distribution(data):
    # Step 1: Calculate rough statistics for the two groups
    peak1_data = data[data <= 0.25]  # Data below 0.25 (first peak)
    peak2_data = data[data > 0.25]   # Data above 0.25 (second peak)

    mean1 = np.mean(peak1_data)  # Rough mean for the first peak (around 0.1)
    mean2 = np.mean(peak2_data)  # Rough mean for the second peak

    # Estimate variance for initial guesses (optional)
    var1 = np.var(peak1_data)
    var2 = np.var(peak2_data)
    # Step 2: Define a good initial guess for the Beta distribution parameters
    # Use mean = alpha / (alpha + beta) and variance formulas to estimate alpha and beta
    def estimate_beta_parameters(mean, var):
        # Solve for alpha and beta using method of moments
        common_term = (mean * (1 - mean) / var) - 1
        alpha = mean * common_term
        beta_param = (1 - mean) * common_term
        return alpha, beta_param

    # Initial guess for first Beta distribution (mean ~ 0.1, variance based on data)
    alpha1, beta1 = estimate_beta_parameters(mean1, var1)

    # Initial guess for second Beta distribution (mean ~ second peak, variance based on data)
    alpha2, beta2 = estimate_beta_parameters(mean2, var2)

    # Initial weight guess (can be roughly 0.5 if populations are similar in size)
    w1_initial = len(peak1_data) / len(data)  # Weight based on relative sizes

    # Step 3: Use these estimates as an initial guess in the optimization
    initial_params = [w1_initial, alpha1, beta1, alpha2, beta2]

    # Define reasonable bounds (weights between 0 and 1, alphas and betas positive)
    bounds = [(0, 1), (0.1, 10), (0.1, 10), (0.1, 10), (0.1, 10)]

    # Step 4: Optimize the negative log-likelihood function
    result = minimize(_beta_mixture_log_likelihood, initial_params, args=(data,), bounds=bounds)

    # Step 5: Extract the optimized parameters
    w1, alpha1, beta1, alpha2, beta2 = result.x

    registration_ratio = w1
    coupling_ratio_reg = alpha1 / (alpha1 + beta1)
    coupling_ratio_anti_reg = alpha2 / (alpha2 + beta2)

    return registration_ratio, coupling_ratio_reg, coupling_ratio_anti_reg


def plot_heatmaps_lambda_vs_R(ax, which_lipid, gamma, property="registration_percentage", is_uniform=True):
    # filter out the data by which_lipid == DAPC
    dapc_df = df[df['which_lipid'] == which_lipid]
    # filter gamma == 2.0
    dapc_df = dapc_df[dapc_df['gamma'] == gamma]
    # filter is_uniform
    dapc_df = dapc_df[dapc_df['uniform'] == is_uniform]
    
    Rs, lambdas = np.meshgrid(RS, LAMBDAS)
    
    property_values = []
    count_reg = []
    count_anti_reg = []
    for (R, lambda_) in zip(Rs.flatten(), lambdas.flatten()):
        temp = dapc_df[(dapc_df['R'] == R) & (dapc_df['lambda'] == lambda_)]['coupling_ratio'].values
        reg = temp[temp >= 0.25]
        anti_reg = temp[temp < 0.25]
        count_reg.append(str(len(reg))+"/50")
        #count_anti_reg.append(str(len(anti_reg))+"/50")
        if property == "registration_percentage":
            property_values.append(len(reg) / (len(reg) + len(anti_reg)))
        elif property == "coupling_ratio_reg":
            property_values.append(np.mean(reg))
        elif property == "coupling_ratio_anti_reg":
            property_values.append(np.mean(anti_reg))
        else:
            raise ValueError("Invalid property")
    
    count_reg = np.array(count_reg).reshape(Rs.shape)
    #count_anti_reg = np.array(count_anti_reg).reshape(Rs.shape)

    property_values = np.array(property_values).reshape(Rs.shape)
    property_values_as_strings = property_values.astype(str)
    # change nan to "N/A"
    property_values_as_strings[np.isnan(property_values)] = "N/A"
    # change the rest to 2 decimal places
    property_values_as_strings[np.logical_not(np.isnan(property_values))] = np.round(property_values[np.logical_not(np.isnan(property_values))], 2).astype(str)
    # if property != "coupling_ratio_anti_reg" add the reg counts to the string
    #if property != "registration_percentage":
    #    for (i, j), value in np.ndenumerate(property_values_as_strings):
    #        if property != "coupling_ratio_anti_reg":
    #            property_values_as_strings[i, j] += count_reg[i, j]
    #        else:
    #            property_values_as_strings[i, j] += count_anti_reg[i, j]
    if property == "registration_percentage":
        property_values_as_strings = count_reg

    vmin = 0.0 if property == "registration_percentage" else (0.25 if property == "coupling_ratio_reg" else 0.0)
    vmax = 1.0 if property == "registration_percentage" else (1.0 if property == "coupling_ratio_reg" else 0.25)
    fontsize = 3.4 if property != "registration_percentage" else 2.8

    sns.heatmap(data=property_values, xticklabels=RS, yticklabels=LAMBDAS, ax=ax, cmap='spring', cbar_kws={'label': property}, vmin=vmin, vmax=vmax, linewidths=0.3, linecolor='black', cbar=False, alpha=0.7)
    # and the number of data points below, use a transparent heatmap and add a line for the number of data points
    sns.heatmap(data=np.zeros_like(property_values), xticklabels=RS, yticklabels=LAMBDAS, ax=ax, cmap='viridis', cbar=False, vmin=0, vmax=1, annot=property_values_as_strings, fmt="", annot_kws={"size": fontsize, "color": "black"}, alpha=0, linewidths=0.3, linecolor='black')
    ax.set_xlabel("R")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"{}".format(property), fontsize=5)

def plot_heatmaps_lambda_vs_R_all_properties(gamma, is_uniform=True):
    fig, axs = plt.subplots(2, 3, figsize=(7,2.33), gridspec_kw = {'wspace':0.05, 'hspace':0.25})
    for i, which_lipid in enumerate(WHICH_LIPIDS):
        for j, property in enumerate(["registration_percentage", "coupling_ratio_reg", "coupling_ratio_anti_reg"]):
            plot_heatmaps_lambda_vs_R(axs[i, j], which_lipid, gamma, property, is_uniform)
    # remove the yticks and ylabels for the second and third plot
    axs[0, 1].set_yticks([])
    axs[0, 2].set_yticks([])
    axs[1, 1].set_yticks([])
    axs[1, 2].set_yticks([])
    axs[0, 1].set_ylabel("")
    axs[0, 2].set_ylabel("")
    axs[1, 1].set_ylabel("")
    axs[1, 2].set_ylabel("")
    # remove the xlabel for the first and third plot
    axs[0, 0].set_xlabel("")
    axs[1, 0].set_xlabel("")
    axs[0, 2].set_xlabel("")
    axs[1, 2].set_xlabel("")
    axs[0, 1].set_xlabel("")   
    # remove the ticks from the first row
    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])
    axs[0, 2].set_xticks([])
    # set the y ticks and labels for the first plot to be larger
    axs[0, 0].tick_params(axis='y', labelsize=4)
    axs[0, 0].set_ylabel(r"$\lambda$", fontsize=5)
    axs[1, 0].tick_params(axis='y', labelsize=4)
    axs[1, 0].set_ylabel(r"$\lambda$", fontsize=5)
    axs[1, 1].set_xlabel("R", fontsize=5)
    axs[1,0].tick_params(axis='x', labelsize=4)
    axs[1,1].tick_params(axis='x', labelsize=4)
    axs[1,2].tick_params(axis='x', labelsize=4)
    # remove titles from the second row
    axs[1, 0].set_title("")
    axs[1, 1].set_title("")
    axs[1, 2].set_title("")
    
    # add title above the plots for the lipid and gamma
    fig.suptitle(r" DAPC ($\gamma = {}$)".format(gamma), fontsize=5)
    # make a title to each row with the lipid
    axs[0, 0].set_title("Registration Ratio", fontsize=5)
    axs[0, 1].set_title("Coupling Ratio in Registered Membranes", fontsize=5)
    axs[0, 2].set_title("Coupling Ratio in Anti-Registered Membranes", fontsize=5)
    axs[1, 1].set_title(r"DPPC ($\gamma = {}$)".format(gamma), fontsize=5)
    plt.tight_layout()
    plt.savefig("plots/heatmaps_lambda_vs_R_gamma_{}.png".format(gamma), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# plot_coupling_ratio_vs_R("DAPC", 0.3, 2.0)
# plot_coupling_ratio_vs_R("DPPC", 0.3, 2.0)
# plot_coupling_ratio_vs_R("DAPC", 0.3, 1.0)
# plot_coupling_ratio_vs_R("DPPC", 0.3, 1.0)
# plot_coupling_ratio_vs_R("DAPC", 0.3, 0.5)
# plot_coupling_ratio_vs_R("DPPC", 0.3, 0.5)
# plot_coupling_ratio_vs_R("DAPC", 0.3, 4.0)
# plot_coupling_ratio_vs_R("DPPC", 0.3, 4.0)
# plot_coupling_ratio_vs_R("DAPC", 0.5, 2.0)
# plot_coupling_ratio_vs_R("DPPC", 0.5, 2.0)
# plot_coupling_ratio_vs_R("DAPC", 0.5, 1.0)
# plot_coupling_ratio_vs_R("DPPC", 0.5, 1.0)
# plot_coupling_ratio_vs_R("DAPC", 0.5, 0.5)
# plot_coupling_ratio_vs_R("DPPC", 0.5, 0.5)
# plot_coupling_ratio_vs_R("DAPC", 0.5, 4.0)
# plot_coupling_ratio_vs_R("DPPC", 0.5, 4.0)
# plot_coupling_ratio_vs_R("DAPC", 0.1, 2.0)
# plot_coupling_ratio_vs_R("DPPC", 0.1, 2.0)
# plot_coupling_ratio_vs_R("DAPC", 0.1, 1.0)
# plot_coupling_ratio_vs_R("DPPC", 0.1, 1.0)
# plot_coupling_ratio_vs_R("DAPC", 0.1, 0.5)
# plot_coupling_ratio_vs_R("DPPC", 0.1, 0.5)
# plot_coupling_ratio_vs_R("DAPC", 0.1, 4.0)
# plot_coupling_ratio_vs_R("DPPC", 0.1, 4.0)

plot_heatmaps_lambda_vs_R_all_properties(4.0)
plot_heatmaps_lambda_vs_R_all_properties(2.0)
plot_heatmaps_lambda_vs_R_all_properties(1.0)
plot_heatmaps_lambda_vs_R_all_properties(0.5)