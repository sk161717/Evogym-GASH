import random
import numpy as np
import torch

from ga.run import run_ga
from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_ga(
        pop_size = 40,
        structure_shape = (5,5),
        experiment_name = "Climber-v0_ga",
        max_evaluations = 10000,
        train_iters = 50,
        num_cores = 8,
    )