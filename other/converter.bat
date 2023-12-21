#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 40G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00


echo "STARTING"
python model_converter.py