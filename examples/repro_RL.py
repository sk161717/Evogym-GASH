from json import load
import os
import numpy as np
import shutil
import random
import math

import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '.')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import  TerminationCondition, Structure
from ppo.arguments import get_args

def run_repro_RL(
    experiment_name,
    pop_size,
    train_iters, 
    num_cores,
    env_name,
    load_expr_name,
    load_gen,
    load_index 
    ):
    
     ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)
    try:
        #shutil.rmtree(home_path)
        os.makedirs(home_path)
    except:
        print("making home_path directory failed")
        exit(1)

    structures=[]
    
    save_path_structure = os.path.join(root_dir, str(load_index) + ".npz")
    np_data = np.load(save_path_structure)
    structure_data = []
    for key, value in np_data.items():
        structure_data.append(value)
    structure_data = tuple(structure_data)

    for i in range(pop_size):
        structures.append(Structure(*structure_data, i))
    
    save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "controller")
    try:
        os.makedirs(save_path_controller)
    except:
        pass
    
    group = mp.Group()
    for structure in structures:
        ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),env_name)
        group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
    group.run_jobs(num_cores)

    for structure in structures:
        structure.compute_fitness()
    structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)


    temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output.txt")
    f = open(temp_path, "w")
    out = ""
    for structure in structures:
        out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
    f.write(out)
    f.close()

if __name__== '__main__':
    args = get_args()
    seed=args.seed
    env_name1="Jumper-v0"
    env_name2="PlatformJumper-v0"
    is_specialJump=False
    suffix='_forwardJump' if is_specialJump else ''
    load_seed=20
    score=8.52
    expr_name=env_name1+"_"+env_name2+suffix+'_seed:'+str(load_seed)
    
    run_repro_RL(
    experiment_name='Repro_'+expr_name+'_score:'+str(score)+'_seed:'+str(seed),
    pop_size=4,
    train_iters=10000, 
    num_cores=4,
    env_name=env_name2,
    load_expr_name=expr_name,
    load_gen=48,
    load_index=1956, 
    )