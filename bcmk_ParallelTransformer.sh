#!/bin/bash
#SBATCH -J Optimus_bcmk
#SBATCH -o Optimus_bcmko.txt
#SBATCH -e Optimus_bcmke.txt
#SBATCH -p rtx
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 02:00:00
module load cuda/10.1
source $HOME/programs/anaconda3/bin/activate
conda activate SUMMA

srun python bcmk_ParallelTransformer.py \
    --checkpoint-activations \
    --distribute-checkpointed-activations \
    --model-parallel-size 4 \
    --summa-dim 2 \
    --num-attention-heads 32 \
    --batch-size 96 \
    --hidden-size 2048 \
    --eval-iters 10 \
    --rank-rearrange \
    --master-port 6006
