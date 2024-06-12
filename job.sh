#!/bin/bash
#SBATCH -J DongDong # Job name
#SBATCH -o para.log # SLURM standard output to file
#SBATCH -e para.log # SLURM standard error to file
#SBATCH --nodes=1 # Require 4 nodes to be used
#SBATCH --ntasks-per-node=1 # 64 tasks/processes to be executed in each node
#SBATCH --cpus-per-task=8 # 1 CPU cores to be used for each task/process
#SBATCH --gres=gpu:4 # Require 8 GPUs to be used
#SBATCH --mem=128G # Specify the 100G real memory required per node
#SBATCH --time=13:00:00 # Set three and a half days as waltime of the job
#SBATCH -p defq # Partition/Queue name

#==========================
# Load modules
#==========================
module purge
module load slurm/slurm/23.02.4
module load nvidia-hpc/2024_241
module load nvhpc/24.1
module list
#==========================
# Execute My Program
#==========================
srun conda init
srun conda activate RLR
srun python3 PrefixPretrain.py