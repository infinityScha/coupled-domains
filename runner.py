import subprocess
from multiprocessing import Pool

def run_script_with_params(params):
    print(f'Running with parameters: {params}')
    script_path = 'CHSolver.jl'
    command = ['julia', script_path] + params
    subprocess.run(command)
    print(f'Finished running with parameters: {params}')

if __name__ == '__main__':
    # Define your parameter lists
    gamma = [2.0, 4.0, 6.0, 8.0]
    lambda_ = [0.0, 0.03, 0.1, 0.25, 0.4]
    gamma_l = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

    parameter_lists = []
    for g in gamma:
        for l in lambda_:
            for gl in gamma_l:
                parameter_lists.append([str(g), str(l), str(gl)])

    # Create a pool of 4 worker processes
    with Pool(6) as pool:
        # Map the parameter lists to the function to run in parallel
        pool.map(run_script_with_params, parameter_lists)
