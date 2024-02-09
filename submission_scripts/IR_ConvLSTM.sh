#!/bin/bash

#SBATCH --job-name ir_only


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=256GB
#SBATCH --partition=gpu1

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out


# python /home1/zhang2012/nowcasting/servir/datasets/dataLoader_wa_IR.py

export CUDA_VISIBLE_DEVICES=1
python /home1/zhang2012/nowcasting/examples/wa_IR_convLSTM.py
