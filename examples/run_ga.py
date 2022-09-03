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

    is_tournament=False
    is_transfer=False
    is_pruning=False
    transfer_gen=100
    suffix="_transfer:gen="+str(transfer_gen) if is_transfer else ''
    suffix="SuHa" if is_pruning else ""


    if is_tournament:
        run_ga_tournament(
            experiment_name = env_name+"_Tournament"+suffix+"GA_seed:"+str(seed),
            structure_shape = (5,5),
            pop_size = 4,
            train_iters = 200,
            num_cores = 4,
            env_name=env_name,
            max_evaluations = 1000,
            is_pruning=is_pruning,
        )
    else:
        run_ga(
            experiment_name = env_name+"_"+suffix+"GA"+'_seed:'+str(seed),
            structure_shape = (5,5),
            pop_size = 32,
            train_iters = 1000,
            num_cores = 4,
            env_name=env_name,
            max_evaluations = 600,
            is_pruning=is_pruning,
        )