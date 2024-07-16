#!/bin/bash

#SBATCH --job-name convLSTM_l1_loss

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=200GB
#SBATCH --partition=gpu1


#SBATCH --mem=200GB

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

# nvidia-smi
source activate servir

srun python /home1/zhang2012/nowcasting/examples/wa_imerg_convLSTM_test.py
