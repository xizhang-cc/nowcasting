#!/bin/bash

#SBATCH --job-name ij3


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=200GB
#SBATCH --partition=gpu2

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

export CUDA_VISIBLE_DEVICES=1
python /home1/zhang2012/nowcasting/examples/exp_1dataset.py
