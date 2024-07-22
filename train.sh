#!/bin/bash

#SBATCH --job-name ghana_imerg

#SBATCH --partition=gpu1
#SBATCH --nodes 2

#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=#) in the code

#SBATCH --mem=200GB

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
# export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

# nvidia-smi
source activate servir

srun python /home1/zhang2012/nowcasting/examples/ghana_imerg_dgmr_train.py
