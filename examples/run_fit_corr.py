from ast import arg
from enum import Flag
import random
import numpy as np
import torch
import math

from ga.fit_corr import calc_fit_corr

from ppo.arguments import get_args
from utils.algo_utils import calc_GAEval_from_SHEvaluations

if __name__ == "__main__":
    args=get_args()
    seed = args.randseed
    random.seed(seed)
    np.random.seed(seed)

    last_gen_dict=\
    {
        'Walker-v0':10,
        "UpStepper-v0":33,
        "PlatformJumper-v0":67,
        'ObstacleTraverser-v0':33,
        'ObstacleTraverser-v1':67,
        'Hurdler-v0':67,
        'GapJumper-v0':67,
        'WingspanMazimizer-v0':21,
        'Lifter-v0':67,
        'CaveCrawler-v0':33
    }

    env_name="PlatformJumper-v0"

    train_iters=1024
    #train_iters=64

    assert  train_iters%args.eval_interval==0

    calc_fit_corr(
        experiment_name=env_name+'_corr_seed:'+str(seed),
        load_expr_name=env_name+'_GA',
        last_gen=last_gen_dict[env_name],
        train_iters=train_iters, 
        num_cores=8,
        env_name=env_name,
        num_per_k=20,
        max_k=15,
        p=3,
        batch_size=32)