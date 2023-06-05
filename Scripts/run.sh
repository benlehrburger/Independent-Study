#!/bin/bash -l
# Name of the job
#SBATCH --job-name=imageGen

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=4

# Request memory
#SBATCH --mem=300G

# Request the GPU partition
#SBATCH --partition=v100_12

# Request the GPU resources
#SBATCH --gres=gpu:1

# Name of the output files to be created. If not specified the outputs will be joined
#SBATCH --output=logs/OUT.out
#SBATCH --error=logs/ERR.err

# Walltime (job duration)
#SBATCH --time=02:00:00

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate ImageGen
cd /dartfs-hpc/rc/home/5/f003xf5/ImageGenPy
python forwardDreamBooth.py # "$1" "$2" "$3"
