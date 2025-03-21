#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mem=32G                                 # memory pool for all cores
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output=/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/log/%j.err  # where to store error messages

# Exit on errors
set -o errexit

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Binary or script to execute
#OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 /itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/benchmark_clinical.py
python /itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/benchmark_bci.py

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0