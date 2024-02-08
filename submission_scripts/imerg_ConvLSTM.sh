#!/bin/bash

#SBATCH --job-name imerg_only


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=400GB
#SBATCH --partition=gpu1

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

export CUDA_VISIBLE_DEVICES=3
python /home1/zhang2012/nowcasting/examples/wa_imerg_convLSTM.py
