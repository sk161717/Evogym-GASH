import random
import numpy as np
import torch

from ga.run_multi import run_multi_ga
from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 40
    random.seed(seed)
    np.random.seed(seed)
    env_name1="Jumper-v0"
    env_name2="PlatformJumper-v0"
    is_ist=False
    is_two_env_parallel=False
    is_specialJump=True
    suffix='_forwardJump' if is_specialJump else ''
    
    run_multi_ga(
        pop_size = 32,
        structure_shape = (5,5),
        experiment_name = env_name1+"_"+env_name2+suffix+'_seed:'+str(seed),
        total_generation = 70,
        train_iters =1000,
        num_cores =6,
        env_name1=env_name1,
        env_name2=env_name2,
        seed=seed,
        is_two_env_parallel=is_two_env_parallel,
        is_ist=is_ist,
    )