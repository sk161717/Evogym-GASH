import random
import numpy as np
import torch

from ga.run import run_ga
from ppo.arguments import get_args

import pyME.map_elites.common as cm_map_elites

if __name__ == "__main__":
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    env_name="PlatformJumper-v0"
    is_transfer=False
    transfer_gen=100
    suffix="_transfer:gen="+str(transfer_gen) if is_transfer else ''
    
    run_ga(
        experiment_name = env_name+"_GA"+suffix+'_seed:'+str(seed),
        structure_shape = (5,5),
        pop_size = 40,
        train_iters = 1000,
        num_cores = 4,
        env_name=env_name,
        dim_map=2,
        n_niches=128,
        max_evaluations = 3000,
        is_transfer=is_transfer,
        transfer_expr_name='Jumper-v0_GA',
        transfer_gen=transfer_gen,
    )