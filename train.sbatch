#!/bin/bash
#SBATCH --job-name=PROMOTE_Test_Training
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
# 
module load jaspy
python state_mo_3_2D.py