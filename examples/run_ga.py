import random
import numpy as np
import torch

from ga.run import run_ga
from ga.run_tournament import run_ga_tournament
from ppo.arguments import get_args

if __name__ == "__main__":
    args=get_args()
    seed = args.randseed
    random.seed(seed)
    np.random.seed(seed)
    env_name="UpStepper-v0"

    is_tournament=True
    is_transfer=False
    is_pruning=False
    transfer_gen=100
    suffix="_transfer:gen="+str(transfer_gen) if is_transfer else ''


    if is_tournament:
        run_ga_tournament(
            experiment_name = env_name+"_TournamentGA"+suffix+'_seed:'+str(seed),
            structure_shape = (5,5),
            pop_size = 32,
            train_iters = 1000,
            num_cores = 4,
            env_name=env_name,
            max_evaluations = 1024,
            is_pruning=is_pruning,
        )
    else:
        run_ga(
            experiment_name = env_name+"_GA"+suffix+'_seed:'+str(seed),
            structure_shape = (5,5),
            pop_size = 40,
            train_iters = 1000,
            num_cores = 32,
            env_name=env_name,
            dim_map=2,
            n_niches=128,
            max_evaluations = 3000,
            is_transfer=is_transfer,
            transfer_expr_name='Jumper-v0_GA',
            transfer_gen=transfer_gen,
        )