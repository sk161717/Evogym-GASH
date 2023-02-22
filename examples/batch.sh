#!/bin/bash

python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $2 --is-pruning $1
python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $3 --is-pruning $1
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $4 --is-pruning $1
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $5 --is-pruning $1
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $6 --is-pruning $1
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $7 --is-pruning $1
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $8 --is-pruning $1

#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 104 --is-pruning


#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $2 --is-pruning $1
#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $3 --is-pruning $1
#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $4 --is-pruning $1
#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $5 --is-pruning $1
#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $6 --is-pruning $1

#python run_fit_corr.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64