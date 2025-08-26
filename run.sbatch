#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --job-name=pytorch
#SBATCH --account=bdym-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1

export HF_HOME=<YOUR_HF_HOME>
export HF_TOKEN= <YOUR_HF_TOKEN>
echo "job is starting on `hostname`"
time srun \
  numactl \
  stdbuf -oL -eL python3 -u script.py

exit