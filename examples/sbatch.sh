#!/bin/sh
#SBATCH -p big
#SBATCH -t 48:0:00
#SBATCH --cpus-per-task 128

python run_ga.py  --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1  --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay  --eval-interval 50 --randseed 5
