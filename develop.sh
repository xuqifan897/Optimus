#!/bin/bash
#SBATCH -J emb
#SBATCH -o embo.txt
#SBATCH -e embe.txt
#SBATCH -p rtx
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 00:30:00

module load cuda/10.1
source $HOME/programs/anaconda3/bin/activate
conda activate SUMMA

srun python pretrain_develop.py \
    --checkpoint-activations \
    --distribute-checkpointed-activations \
    --master-port 2048 \
    --batch-size 96 \
    --hidden-size 2048 \
    --rank-rearrange
