import random
import numpy as np
import torch

from pyME.map_elites.multi_cvt import run_MOME
import pyME.map_elites.common as cm_map_elites

from ppo.arguments import get_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env_name1="Climber-v0"
    env_name2="Climber-v1"

    px = cm_map_elites.default_params.copy()

    run_MOME(
        pop_size = 40,
        structure_shape = (5,5),
        experiment_name = env_name1+"_"+env_name2+"_"+"MOME",
        total_generation = 100,
        train_iters =50,
        num_cores = 8,
        env_name1=env_name1,
        env_name2=env_name2,
        n_samples=8,
        batch_size=32,
        p_mut=1.0,
        pareto_max=20,
        dim_map=2,
        n_niches=128)

