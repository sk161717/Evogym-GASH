import random
import numpy as np
import math

from cppn_neat.run import run_cppn_neat
from ppo.arguments import get_args

if __name__ == '__main__':
    args=get_args()
    seed = args.randseed
    random.seed(seed)
    np.random.seed(seed)

    env_max_eval=\
    {
        'Walker-v0':1000,
        'BridgeWalker-v0':1000,
        'Climber-v0':3000,
        'Traverser-v0':3000,
    }

    #env_name='Walker-v0'
    env_name='BridgeWalker-v0'
    #env_name='Climber-v0'
    #env_name='Traverser-v0'

    is_pruning= True if args.is_pruning==1 else False

    suffix="SuHa" if is_pruning else ""

    train_iters=1024
    eval_timing_arr=[64,128,256,512]
    #train_iters=128
    #eval_timing_arr=[8,16,32,64]
    max_evaluations=env_max_eval[env_name]

    assert  train_iters%args.eval_interval==0
    for eval_timing in eval_timing_arr:
        assert eval_timing%args.eval_interval==0

    best_robot, best_fitness = run_cppn_neat(
        experiment_name=env_name+"_"+suffix+"cppn"+'_seed:'+str(seed),
        structure_shape=(5, 5),
        pop_size=32,
        train_iters=train_iters,
        num_cores=32 if is_pruning else 8,
        max_evaluations=max_evaluations if is_pruning else math.ceil(max_evaluations*3/16),
        env_name=env_name,
        eval_timing_arr=eval_timing_arr,
        is_pruning=is_pruning,
    )
    
    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)