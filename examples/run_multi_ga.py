import random
import numpy as np
import torch

from ga.run_multi import run_multi_ga
from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_multi_ga(
        pop_size = 10,
        structure_shape = (5,5),
        experiment_name = "test_ga",
        max_evaluations = 100,
        train_iters = 50,
        num_cores = 3,
        env_name1="BridgeWalker-v0",
        env_name2="Walker-v0"
    )