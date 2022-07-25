from os import environ
import random
import numpy as np
import torch

from pyME.map_elites.single_cvt import run_single_ME
import pyME.map_elites.common as cm_map_elites

from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    env_name="Walker-v0"
    batch_size=32

    run_single_ME(
        experiment_name = env_name+"_ME"+'_seed:'+str(seed),  
        structure_shape = (5,5),
        train_iters =1000,
        num_cores = 8,
        env_name=env_name,
        n_samples=batch_size,
        batch_size=batch_size,
        dim_map=2,
        n_niches=32,
        max_eval=3000,
        produce_gif=False)

