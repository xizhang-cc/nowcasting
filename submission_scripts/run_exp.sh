#!/bin/bash

#SBATCH --job-name convLSTM_l1_loss

#SBATCH --partition=gpu1
#SBATCH --nodes 2

#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=#) in the code

#SBATCH --mem=200GB

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

source activate servir

srun python /home1/zhang2012/nowcasting/examples/wa_imerg_train.py
