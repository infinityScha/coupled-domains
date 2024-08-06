import subprocess
from tqdm.contrib.concurrent import process_map

def run_script_with_params(params):
    command = ['julia', '-O3', 'CHSolver_fixed_gl.jl', *map(str, params)]
    subprocess.run(command)

if __name__ == '__main__':
    # Define your parameter lists
    which_lipid = ["DAPC", "DPPC"]
    gammas = [0.5, 1.0, 2.0, 4.0]
    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0]
    Rs = [100.0, 75.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0]
    c0s = [0.0]
    is_mids = ["true", "false"]
    run_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Generate all combinations of parameters
    from itertools import product
    parameter_lists = list(product(which_lipid, gammas, lambdas, Rs, c0s, is_mids, run_count))                    

    # Run the script with the parameters with 13 workers
    r = process_map(run_script_with_params, parameter_lists, max_workers=13)    