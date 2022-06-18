import random
import numpy as np
import torch

from ga.run import run_ga
from ppo.arguments import get_args

import pyME.map_elites.common as cm_map_elites

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env_name="Thrower-v0"
    
    run_ga(
        experiment_name = env_name+"_GA",
        structure_shape = (5,5),
        pop_size = 40,
        train_iters = 1000,
        num_cores = 8,
        env_name=env_name,
        dim_map=2,
        n_niches=128,
        max_evaluations = 10000,
        
    )