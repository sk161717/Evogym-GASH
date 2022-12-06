#!/bin/bash
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 101 --is-pruning
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 102 --is-pruning
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 103 --is-pruning
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 104 --is-pruning
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 105 --is-pruning
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 12
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 13
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 14
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 15

python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $2 --is-pruning $1
python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $3 --is-pruning $1
#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $4 --is-pruning $1
#python run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $5 --is-pruning $1
#ython run_cppn_neat.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed $6 --is-pruning $1
