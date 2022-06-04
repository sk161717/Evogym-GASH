import random
import numpy as np
import torch

from ga.run_multi import run_multi_ga
from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env_name1="Climber-v0"
    env_name2="Climber-v1"
    
    run_multi_ga(
        pop_size = 40,
        structure_shape = (5,5),
        experiment_name = env_name1+"_"+env_name2+"_"+'debug',
        total_generation = 100,
        train_iters =50,
        num_cores =8,
        env_name1=env_name1,
        env_name2=env_name2
    )