#!/bin/sh
#SBATCH -p big
#SBATCH -t 48:0:00
#SBATCH --cpus-per-task 64

python run_ga.py  --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1  --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay  --eval-interval 64 --randseed $1 --is-pruning $2 --env-name-for-ist $3
#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $1 --is-pruning $2
#python run_fit_corr.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64