import random
import numpy as np
import torch

from pymap_elites.map_elites.single_cvt import run_single_ME
import pymap_elites.map_elites.common as cm_map_elites

from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env_name="Thrower-v0"

    px = cm_map_elites.default_params.copy()

    run_single_ME(
        experiment_name = env_name+"_"+"ME",  
        structure_shape = (5,5),
        total_generation = 100,
        train_iters =50,
        num_cores = 8,
        env_name=env_name,
        n_samples=8,
        batch_size=16,
        p_mut=1.0,
        dim_map=2,
        n_niches=128,
        max_eval=10000)

