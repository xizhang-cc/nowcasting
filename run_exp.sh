#!/bin/bash

#SBATCH --job-name convLSTM_l1_loss

#SBATCH --partition=gpu2
#SBATCH --nodes 4

#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=#) in the code

#SBATCH --mem=200GB

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

# nvidia-smi
source activate servir

srun python /home1/zhang2012/nowcasting/examples/wa_imerg_train.py
