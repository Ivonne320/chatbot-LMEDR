#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 40G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00


echo "STARTING"
python train_PersonaChat.py --lr 8e-6 --epochs 20 --train_batch_size 2 --valid_batch_size 2 --infer_batch_size 64 --load_from /home/yiwang/chatbot-LMEDR/persona_original/checkpoint/