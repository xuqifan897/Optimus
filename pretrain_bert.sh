#!/bin/bash
#SBATCH -J train_bert
#SBATCH -o berto.txt
#SBATCH -e berte.txt
#SBATCH -p rtx
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 02:00:00
module load cuda/10.1
source $HOME/programs/anaconda3/bin/activate
conda activate SUMMA

export NCCL_SOCKET_IFNAME=ib0

srun python pretrain_bert.py \
    --checkpoint-activations \
    --distribute-checkpointed-activations \
    --DDP-impl "torch" \
    --model-parallel-size 4 \
    --summa-dim 2 \
    --batch-size 76 \
    --hidden-size 2048 \
    --log-interval 1 \
    --train-iters 5 \
    --rank-rearrange \
    --master-port 6007


#srun python pretrain_bert.py \
#    --checkpoint-activations \
#    --distribute-checkpointed-activations \
#    --DDP-impl "torch" \
#    --model-parallel-size 4 \
#    --summa-dim 2 \
#    --batch-size 64 \
#    --hidden-size 2048 \
#    --log-interval 1 \
#    --train-iters 1 \
#    --master-port 6008
