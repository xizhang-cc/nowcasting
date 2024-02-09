#!/bin/bash

#SBATCH --job-name imerg_fsss


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=100GB
#SBATCH --partition=gpu2

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

export CUDA_VISIBLE_DEVICES=3
python /home1/zhang2012/nowcasting/examples/wa_imerg_convLSTM_pred.py
