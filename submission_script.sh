#!/bin/bash
#SBATCH --job-name=c_d
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --array=1-640%72  # Run up to 72 jobs simultaneously
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -n 1 
#SBATCH --time=02:00:00  # Adjust based on your script duration
#SBATCH --mem=1G         # Adjust based on your memory needs
#SBATCH --constraint=part-a
#SBATCH -p a36_any

# Read the parameter from the file corresponding to this job array index
param=$(sed -n "${SLURM_ARRAY_TASK_ID}p" parameters.txt)

# Run your Julia script with the current parameter
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 1
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 2
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 3
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 4
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 5
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 6
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 7
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 8
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 9
~/soft/julia-1.9.0/bin/julia CHSolver_fixed_gl.jl $param 10
