from json import load
import os
import numpy as np
import shutil
import random
import math
import multiprocessing

import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '.')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import *
from ppo.arguments import get_args

def run_repro_RL(
    experiment_name,
    num_cores,
    env_name,
    ):

    print("Enter gen to start training on (0-indexed): ", end="")
    generation = int(input())
    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name,'generation_'+str(generation),"controller64")
    
    try:
        #shutil.rmtree(home_path)
        os.makedirs(home_path)
    except:
        print("making home_path directory failed")
        exit(1)
    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(64)
    

    structures=load_archive(generation,experiment_name,filename='structures')
    
    group = mp.Group()
    queue=multiprocessing.Queue()
    for structure in structures:
        ppo_args = ((structure.body, structure.connections), tc, (home_path, structure.label),env_name,None,None,False,None,queue)
        group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
    group.run_jobs(num_cores,queue)

    for structure in structures:
        structure.compute_fitness()
    structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)


    temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output64.txt")
    f = open(temp_path, "w")
    out = ""
    for structure in structures:
        out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
    f.write(out)
    f.close()

if __name__== '__main__':
    args = get_args()
    seed=104
    env_name='Lifter-v0'
   
    
    run_repro_RL(
    experiment_name=env_name+'_SuHaGA_seed:'+str(seed),
    num_cores=32,
    env_name=env_name
    )