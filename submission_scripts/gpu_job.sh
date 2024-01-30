#!/bin/bash

#SBATCH --job-name 1gpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=256GB
#SBATCH --partition=gpu1

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out


python /home1/zhang2012/nowcasting/examples/wa_imerg_convLSTM.py
