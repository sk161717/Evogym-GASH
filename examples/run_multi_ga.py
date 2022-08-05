import random
import numpy as np
import torch

from ga.run_multi import run_multi_ga
from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    env_name1="UpStepper-v0"
    env_name2="NSGC"
    is_ist=False
    is_two_env_parallel=False
    is_specialJump=False
    is_NSGC=True
    suffix='_forwardJump' if is_specialJump else ''
    
    run_multi_ga(
        pop_size = 32,
        structure_shape = (5,5),
        experiment_name = env_name1+"_"+env_name2+suffix+'_seed:'+str(seed),
        total_generation = 30,
        train_iters =1000,
        num_cores =4,
        env_name1=env_name1,
        env_name2=env_name2,
        seed=seed,
        is_two_env_parallel=is_two_env_parallel,
        is_ist=is_ist,
        is_NSGC=is_NSGC
    )