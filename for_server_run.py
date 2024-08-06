import itertools

# Define your parameter lists
which_lipid = ["DAPC", "DPPC"]
gammas = [0.5, 1.0, 2.0, 4.0]
lambdas = [0.0, 0.1, 0.3, 0.5, 1.0]
Rs = [100.0, 75.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0]
c0s = [0.0]
is_mids = ["true", "false"]

# Generate all combinations of parameters
parameter_lists = list(itertools.product(which_lipid, gammas, lambdas, Rs, c0s, is_mids))

# Save to a file
with open("parameters.txt", "w") as f:
    for params in parameter_lists:
        f.write(" ".join(map(str, params)) + "\n")