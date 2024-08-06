import subprocess
from tqdm.contrib.concurrent import process_map

def run_script_with_params(params):
    command = ['julia', 'CHSolver.jl', *params]
    subprocess.run(command)

if __name__ == '__main__':
    # Define your parameter lists
    gamma = [1.0] #[2.0, 4.0, 6.0]
    lambda_ = [0.05, 0.15, 0.3, 0.4]
    gamma_l = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

    parameter_lists = []
    for g in gamma:
        for l in lambda_:
            for gl in gamma_l:
                parameter_lists.append([str(g), str(l), str(gl)])

    # Run the script with the parameters with 13 workers
    r = process_map(run_script_with_params, parameter_lists, max_workers=13)    