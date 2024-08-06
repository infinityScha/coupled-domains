#!/bin/bash

# Define the command-line arguments
# WHICH_LIST = ["DPPC", "DAPC", "DAPC_LIN", "DAPC_AS", "DPPC_AS"]
# FOLDER_LIST = ["DPPC", "DAPC", "DAPC_LIN", "DAPC_POPC", "DPPC_POPC"]

# Define the arguments for WHICH and FOLDER
WHICH_LIST=("DAPC_LIN") # ("DPPC" "DAPC" "DAPC_LIN" "DAPC_AS" "DPPC_AS")
FOLDER_LIST=("CURVED_LINACTANT") # ("DPPC" "DAPC" "DAPC_LIN" "DAPC_POPC" "DPPC_POPC")


# Loop through the which and folder lists at the same time
for i in "${!WHICH_LIST[@]}"; do
    which="${WHICH_LIST[$i]}"
    folder="${FOLDER_LIST[$i]}"
    taskset -c 0-8 python simulations_analysis/densities.py $which $folder
done
