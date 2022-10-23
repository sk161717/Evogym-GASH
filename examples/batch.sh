#!/bin/bash
python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 104
python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 105
python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 106
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 512 --use-linear-lr-decay --eval-interval 64 --randseed 105
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 10
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 12
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 13
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 14
#python run_ga.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 500 --use-linear-lr-decay --eval-interval 50 --randseed 15
