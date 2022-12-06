from ast import arg
from enum import Flag
import random
import numpy as np
import torch
import math

from ga.run import run_ga
from ga.run_tournament import run_ga_tournament
from ppo.arguments import get_args
from utils.algo_utils import calc_GAEval_from_SHEvaluations

if __name__ == "__main__":
    args=get_args()
    seed = args.randseed
    random.seed(seed)
    np.random.seed(seed)

    env_max_eval=\
    {
        'Walker-v0':1000,
        "UpStepper-v0":3000,
        "PlatformJumper-v0":6000,
        'ObstacleTraverser-v0':3000,
        'ObstacleTraverser-v1':6000,
        'Hurdler-v0':6000,
        'GapJumper-v0':6000,
        'WingspanMazimizer-v0':2000,
    }

    #env_name='Walker-v0'
    #env_name="UpStepper-v0"
    #env_name="PlatformJumper-v0"
    #env_name="ObstacleTraverser-v1"
    #env_name='ObstacleTraverser-v0'
    #env_name='Hurdler-v0'
    #env_name='GapJumper-v0'
    env_name='WingspanMazimizer-v0'

    is_pruning=True if args.is_pruning==1 else False
    is_ist=False
    resume_gen=204
    robot_size=5
    scale=1

    is_tournament=False
    is_transfer=False
    transfer_gen=100
    suffix="_transfer:gen="+str(transfer_gen) if is_transfer else ''
    suffix="scale"+str(scale)+"_" if scale > 1 else ""
    suffix+="SuHa" if is_pruning else ""
    suffix+=str(robot_size)+'*'+str(robot_size) if robot_size!=5 else ''

    train_iters=1024
    eval_timing_arr=[64,128,256,512]
    #train_iters=128
    #eval_timing_arr=[8,16,32,64]
    max_evaluations=env_max_eval[env_name]

    assert  train_iters%args.eval_interval==0
    for eval_timing in eval_timing_arr:
        assert eval_timing%args.eval_interval==0
    assert is_ist==False or resume_gen!=None

    



    if is_tournament:
        run_ga_tournament(
            experiment_name = env_name+"_Tournament"+suffix+"GA_seed:"+str(seed),
            structure_shape = (5,5),
            pop_size = 32,
            train_iters = train_iters,
            num_cores = 8,
            env_name=env_name,
            max_evaluations = 1024,
            eval_timing_arr=eval_timing_arr,
            is_pruning=is_pruning,
            scale=scale,
            is_ist=is_ist,
            resume_gen=resume_gen,
        )
    else:
        run_ga(
            experiment_name = env_name+"_"+suffix+"GA"+'_seed:'+str(seed),
            structure_shape = (robot_size,robot_size),
            pop_size = 32*scale,
            train_iters = train_iters,
            num_cores = 32 if is_pruning or is_ist else 8,
            env_name=env_name,
            max_evaluations = max_evaluations if is_pruning else math.ceil(calc_GAEval_from_SHEvaluations(max_evaluations)),
            eval_timing_arr=eval_timing_arr,
            is_pruning=is_pruning,
            scale=scale,
            is_ist=is_ist,
            resume_gen=resume_gen,
        )