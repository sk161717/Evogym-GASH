from json import load
import os
from tracemalloc import start
import numpy as np
import shutil
import random
import math
import multiprocessing
import json

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from pyME.map_elites.single_cvt import __add_to_archive
from pyME.map_elites import common as cm
from sklearn.neighbors import KDTree
from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import *
from utils.pruning_params import Params

def run_ga(experiment_name, structure_shape, pop_size,train_iters, num_cores,
        env_name,
        max_evaluations,
        eval_timing_arr, 
        is_pruning=False,
        scale=1,
        is_ist=False,
        resume_gen=None,
        dim_map=2,
        n_niches=128, 
        target_score=None,
        survival_rate_from_score=False,
        cm_params=cm.default_params,
        is_transfer=False,
        transfer_expr_name=None,
        transfer_gen=None,):

    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0
    unique_label = UniqueLabel()

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)

    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        if is_ist:
            print('this experiment is launched in ist, continue from gen:'+str(resume_gen))
            start_gen=resume_gen
            is_continuing=True
        else:
            print("Override? (y/n/c): ", end="")
            ans = input()
            if ans.lower() == "y":
                shutil.rmtree(home_path)
                print()
            elif ans.lower() == "c":
                print("Enter gen to start training on (0-indexed): ", end="")
                start_gen = int(input())
                is_continuing = True
                print()
            else:
                return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = start_gen
    archive = {}  # init archive (empty)
    curr_max=0
    params=Params(pop_size,eval_timing_arr,scale,num_cores)
    div_log=[]

    c = cm.cvt(n_niches, dim_map,
               cm_params['cvt_samples'], cm_params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    
    #generate a population
    if not is_continuing:
        if is_transfer==False: 
            for i in range (pop_size):
                
                temp_structure = sample_robot(structure_shape)
                while (hashable(temp_structure[0]) in population_structure_hashes):
                    temp_structure = sample_robot(structure_shape)

                structures.append(Structure(*temp_structure, unique_label.give_label(),-1))
                population_structure_hashes[hashable(temp_structure[0])] = True
                num_evaluations += 1
        else:
            for i in range(pop_size):
                save_path_structure = os.path.join(root_dir, "saved_data", transfer_expr_name, "generation_" + str(transfer_gen), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                structures.append(Structure(*structure_data, i))
            num_evaluations=load_evaluation(transfer_expr_name,transfer_gen)

    #read status from file
    else:
        structures=load_archive(generation,experiment_name,filename='structures')
        archive=load_archive(generation,experiment_name)
        population_structure_hashes=load_population_hashes(generation,experiment_name)
        num_evaluations = len(list(population_structure_hashes.keys()))
        unique_label.set_label_start_for_resuming(num_evaluations)
        div_log=load_single_array_val(experiment_name,generation,'div')
        remove_only_files(experiment_name,generation)
        

    while True:

        ### UPDATE NUM SURVIORS ###	
        if survival_rate_from_score:
            percent_survival= get_percent_survival_from_score(curr_max,target_score)
        else:		
            percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        num_evaluated=sum([0 if structure.is_survivor else 1 for structure in structures])
        params.calc_params_interactivly(num_evaluated,scale)
        queue=multiprocessing.Queue()
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                
                
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error copying controller for {save_path_controller_part}.\n')
                
            else:        
                ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),env_name,experiment_name,generation,is_pruning,params,queue)
                group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
        initialize_start_log(experiment_name,generation,params)
        group.add_args(experiment_name,generation,params)
        group.run_jobs(num_cores,queue)

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()
            structure.desc=cm.calc_desc(structure.body)
            __add_to_archive(structure, structure.desc, archive, kdt)
        
        save_archive(archive,generation,experiment_name)

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)
        
        write_output(structures,experiment_name,generation,num_evaluations)
        div_log.append(compute_diversity(structures))
        fitness_list,evaluation_list=max_fit_list_single(experiment_name,generation)
        plot_one_graph(experiment_name,generation,fitness_list,evaluation_list)
        plot_one_graph(experiment_name,generation,div_log,evaluation_list,target='div')
        save_single_array_val(div_log,experiment_name,generation,'div')

        cm.save_centroid_and_map(root_dir,experiment_name,generation,archive,n_niches)

        add_lineage(structures,experiment_name,generation)

        curr_max=structures[0].fitness
        

         ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        print(f'FINISHED GENERATION {generation} \n')

        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
    
        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, unique_label.give_label(),survivors[parent_index[0]].label)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children+num_survivors]

        save_polulation_hashes(population_structure_hashes,generation,experiment_name)
        save_archive(structures,generation,experiment_name,filename='structures')

        generation += 1