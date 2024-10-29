import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import beta, chi2
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
    # print a message if the likelihood is zero or negative or nan
    if np.any(total_likelihood <= 0) or np.any(np.isnan(total_likelihood)):
        print("Params: ", params)
        print("Likelihood1: ", likelihood1)
        print("Likelihood2: ", likelihood2)
        print("Total Likelihood: ", total_likelihood)
    log_likelihood = np.sum(np.log(total_likelihood))
    
    # Return negative log-likelihood because we want to minimize
    return -log_likelihood


def _beta_non_mixture_log_likelihood(params, data):
    alpha, beta_ = params
    likelihood = beta.pdf(data, alpha, beta_)
    log_likelihood = np.sum(np.log(likelihood))
    return -log_likelihood


def _fit_bi_beta_distribution(data):
    # Step 1: Calculate rough statistics for the two groups
    peak1_data = data[data <= 0.25]  # Data below 0.25 (first peak)
    peak2_data = data[data > 0.25]   # Data above 0.25 (second peak)

    if len(peak1_data) == 0:
        # there is no anti-registration
        registration_ratio = 1.0
        coupling_ratio_reg = np.mean(data)
        coupling_ratio_anti_reg = np.nan
        return registration_ratio, coupling_ratio_reg, coupling_ratio_anti_reg
    elif len(peak2_data) == 0:
        # there is no registration
        registration_ratio = 0.0
        coupling_ratio_reg = np.nan
        coupling_ratio_anti_reg = np.mean(data)
        return registration_ratio, coupling_ratio_reg, coupling_ratio_anti_reg

    mean1 = np.mean(peak1_data)  # Rough mean for the first peak (around 0.1)
    mean2 = np.mean(peak2_data)  # Rough mean for the second peak

    # Estimate variance for initial guesses (optional)
    var1 = np.var(peak1_data)
    var2 = np.var(peak2_data)
    # Step 2: Define a good initial guess for the Beta distribution parameters
    # Use mean = alpha / (alpha + beta) and variance formulas to estimate alpha and beta
    def estimate_beta_parameters(mean, var):
        if var == 0:
            # If variance is zero, return a default value
            return 1.0, 1.0
        # Solve for alpha and beta using method of moments
        common_term = (mean * (1 - mean) / var) - 1
        alpha = mean * common_term
        beta_param = (1 - mean) * common_term
        alpha = max(1.0, alpha)  # Ensure alpha is at least 1
        beta = max(1.0, beta_param)  # Ensure beta is at least 1
        alpha = min(1000.0, alpha)  # Ensure alpha is at most 1000
        beta_param = min(1000.0, beta_param)  # Ensure beta is at most 1000
        return alpha, beta_param

    # Initial guess for first Beta distribution (mean ~ 0.1, variance based on data)
    alpha1, beta1 = estimate_beta_parameters(mean1, var1)

    # Initial guess for second Beta distribution (mean ~ second peak, variance based on data)
    alpha2, beta2 = estimate_beta_parameters(mean2, var2)

    # Initial weight guess based on relative sizes of the two peaks
    w1_initial = len(peak1_data) / len(data)  # Weight based on relative sizes

    # Initial guess for alpha and beta of the non-mixture
    alpha1_non_mix, beta1_non_mix = estimate_beta_parameters(np.mean(data), np.var(data))

    # Step 3: Use these estimates as an initial guess in the optimization
    initial_params = [w1_initial, alpha1, beta1, alpha2, beta2]

    # Define reasonable bounds (weights between 0 and 1, alphas and betas positive)
    bounds = [(0.03, 0.97), (1.0, 1000.0), (1.0, 1000.0), (1.0, 1000.0), (1.0, 1000.0)]

    # Step 4: Optimize the negative log-likelihood function
    result = minimize(_beta_mixture_log_likelihood, initial_params, args=(data,), bounds=bounds)
    result_non_mix = minimize(_beta_non_mixture_log_likelihood, [alpha1_non_mix, beta1_non_mix], args=(data,), bounds=[(1.0, 1000.0), (1.0, 1000.0)])

    # compare log likelihoods, use likelihood ratio test to see if the mixture is neeeded   
    log_likelihood_mixture = -result.fun
    log_likelihood_non_mix = -result_non_mix.fun

    # do p-value test (?)
    l_LR = 2 * (log_likelihood_mixture - log_likelihood_non_mix)
    p_value = chi2.sf(l_LR, 3)

    print("P-value: ", p_value)

    if p_value > 0.05:
        # use the non-mixture because the mixture is not needed
        registration_ratio = 1.0
        coupling_ratio_reg = np.mean(data)
        coupling_ratio_anti_reg = np.nan
        if coupling_ratio_reg < 0.25:
            # make reg
            registration_ratio = 0.0
            coupling_ratio_reg = np.nan
            coupling_ratio_anti_reg = np.mean(data)
        # plot the beta distribution and the data's histogram
        x = np.linspace(0, 1, 1000)
        y = beta.pdf(x, alpha1_non_mix, beta1_non_mix)
        fig, ax = plt.subplots()
        ax.hist(data, bins=50, density=True, alpha=0.5)
        ax.plot(x, y, label="Beta")
        ax.legend()
        plt.savefig("plots/fit_example.png", dpi=300)
        plt.close()
        return registration_ratio, coupling_ratio_reg, coupling_ratio_anti_reg


    # Step 5: Extract the optimized parameters
    w1, alpha1, beta1, alpha2, beta2 = result.x

    registration_ratio = w1
    coupling_ratio_reg = alpha1 / (alpha1 + beta1)
    coupling_ratio_anti_reg = alpha2 / (alpha2 + beta2)

    # the coupling registartion should be larger in the registered phase
    if coupling_ratio_reg < coupling_ratio_anti_reg:
        coupling_ratio_reg, coupling_ratio_anti_reg = coupling_ratio_anti_reg, coupling_ratio_reg
        registration_ratio = 1 - registration_ratio

    # plot the two beta distributions and the data's histogram
    x = np.linspace(0, 1, 1000)
    y1 = beta.pdf(x, alpha1, beta1)
    y2 = beta.pdf(x, alpha2, beta2)
    y = w1 * y1 + (1 - w1) * y2
    fig, ax = plt.subplots()
    ax.hist(data, bins=50, density=True, alpha=0.5)
    ax.plot(x, w1 * y1, label="Beta 1")
    ax.plot(x, (1 - w1) * y2, label="Beta 2")
    ax.plot(x, y, label="Mixture")
    ax.legend()
    plt.savefig("plots/fit_example.png", dpi=300)
    plt.close()
    
    if coupling_ratio_reg < 0.25:
        # there is no registration
        registration_ratio = 0.0
        coupling_ratio_reg = np.nan
        coupling_ratio_anti_reg = np.mean(data)

    if coupling_ratio_anti_reg > 0.25:
        # there is no anti-registration
        registration_ratio = 1.0
        coupling_ratio_reg = np.mean(data)
        coupling_ratio_anti_reg = np.nan

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
    for (R, lambda_) in zip(Rs.flatten(), lambdas.flatten()):
        temp = dapc_df[(dapc_df['R'] == R) & (dapc_df['lambda'] == lambda_)]['coupling_ratio'].values
        registration_ratio, coupling_ratio_reg, coupling_ratio_anti_reg = _fit_bi_beta_distribution(temp)
        if property == "registration_percentage":
            property_values.append(registration_ratio)
        elif property == "coupling_ratio_reg":
            property_values.append(coupling_ratio_reg)
        elif property == "coupling_ratio_anti_reg":
            property_values.append(coupling_ratio_anti_reg)

    property_values = np.array(property_values).reshape(Rs.shape)
    property_values_as_strings = property_values.astype(str)
    # change nan to "N/A"
    property_values_as_strings[np.isnan(property_values)] = "N/A"
    # change the rest to 2 decimal places
    property_values_as_strings[np.logical_not(np.isnan(property_values))] = np.round(property_values[np.logical_not(np.isnan(property_values))], 2).astype(str)

    vmin = 0.0 if property == "registration_percentage" else (0.25 if property == "coupling_ratio_reg" else 0.0)
    vmax = 1.0 if property == "registration_percentage" else (1.0 if property == "coupling_ratio_reg" else 0.25)
    fontsize = 3.4

    sns.heatmap(data=property_values, xticklabels=RS, yticklabels=LAMBDAS, ax=ax, cmap='spring', cbar_kws={'label': property}, vmin=vmin, vmax=vmax, linewidths=0.3, linecolor='black', cbar=False, alpha=0.7)
    # and the number of data points below, use a transparent heatmap and add a line for the number of data points
    sns.heatmap(data=np.zeros_like(property_values), xticklabels=RS, yticklabels=LAMBDAS, ax=ax, cmap='spring', cbar=False, vmin=0, vmax=1, annot=property_values_as_strings, fmt="", annot_kws={"size": fontsize, "color": "black"}, alpha=0, linewidths=0.3, linecolor='black')
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
    
    axs[0, 0].set_title("Registration Ratio", fontsize=5)
    axs[0, 1].set_title("Coupling Ratio in Registered Membranes", fontsize=5)
    axs[0, 2].set_title("Coupling Ratio in Anti-Registered Membranes", fontsize=5)
    # make a title to each row with the lipid by adding text to the lelf
    axs[0, 0].text(-0.5, 0.5, "DAPC", fontsize=5, ha='center', va='center', rotation='vertical')
    axs[1, 0].text(-0.5, 0.5, "DPPC", fontsize=5, ha='center', va='center', rotation='vertical')

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