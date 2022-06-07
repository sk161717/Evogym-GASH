from random import random
from typing import Tuple
import numpy as np
import shutil
import sys, os,itertools

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '../..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
from sklearn.neighbors import KDTree

from pymap_elites.map_elites import common as cm
from utils.algo_utils import mutate, TerminationCondition, Structure, UniqueLabel
from evogym import sample_robot, hashable
from ppo import run_ppo
import utils.mp_group as mp

population_structure_hashes = {}
unique_label = UniqueLabel()


def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1


class MyProblem():

    def __init__(self,tc,num_cores,env_name) -> None:
        self.tc = tc
        self.num_cores=num_cores
        self.env_name=env_name

    def _evaluate(self,X,save_path_controller):
        self.RunOneEnv(X,self.env_name,save_path_controller)

    def RunOneEnv(self,X,env_name,save_path_controller):
        group = mp.Group()
        for x in X:
            ppo_args = ((x.body, x.connections), self.tc, (save_path_controller, x.label),env_name)
            group.add_job(run_ppo, ppo_args, callback=x.set_reward)
        group.run_jobs(self.num_cores)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for x in X:
            x.compute_fitness()
            x.desc=cm.calc_desc(x.body)

def make_save_path(expr_name, generation):
    save_path_controller = os.path.join(root_dir, "saved_data", expr_name, "generation_" + str(generation),
                                         "controller")
    try:
        os.makedirs(save_path_controller)
    except:
        pass

    return save_path_controller


# map-elites algorithm (CVT variant)
def run_single_ME(experiment_name, structure_shape, total_generation,
            train_iters, num_cores, env_name, n_samples, batch_size, p_mut,
            dim_map,
            n_niches,
            params=cm.default_params,
            max_eval=10000):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)

    is_continuing = False
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
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
        f.write(f'NICHE SIZE: {n_niches}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'TOTAL_GENERATION: {total_generation}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.write(f'ENV_NAMES: {env_name}\n')
        f.close()

    # setup the parallel processing pool
    myProblem = MyProblem(tc, num_cores, env_name)

    # create the CVT
    c = cm.cvt(n_niches, dim_map,
               params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')

    archive = {}  # init archive (empty)
    generation = 0  # number of evaluations since the beginning
    curr_evaluation=0
    b_generation = 0  # number evaluation since the last dump

    # main loop
    while (curr_evaluation < max_eval):
        save_path_controller = make_save_path(experiment_name, generation)
        # random initialization
        if generation == 0:
            # initialize robot
            X = np.full(n_samples, None, dtype=object)
            curr_evaluation+=n_samples

            for i in range(n_samples):
                temp_structure = sample_robot(structure_shape)
                while (hashable(temp_structure[0]) in population_structure_hashes):
                    temp_structure = sample_robot(structure_shape)

                X[i] = Structure(*temp_structure, unique_label.give_label())
                population_structure_hashes[hashable(temp_structure[0])] = True

        else:  # variation/selection loop
            keys = list(archive.keys())
            n_mut = int(batch_size * p_mut)
            n_cross = int(batch_size * (1 - p_mut))
            X = np.full(n_mut + n_cross, None, dtype=object)
            curr_evaluation+=batch_size
            
            # mutation
            rand = np.random.randint(len(keys), size=n_mut)
            for i in range(0, n_mut):
                # parent selection
                x = archive[keys[rand[i]]]
                # copy & add variation
                child = None

                while child == None or hashable(child[0]) in population_structure_hashes:
                    child = mutate(x.body.copy(), mutation_rate=0.1, num_attempts=50)

                # overwrite structures array w new child
                X[i] = Structure(*child, unique_label.give_label())
                population_structure_hashes[hashable(child[0])] = True

            # crossover
            rand1 = np.random.randint(len(keys), size=batch_size)
            rand2 = np.random.randint(len(keys), size=batch_size)
            for i in range(0, n_cross):
                # parent selection
                x = archive[keys[rand1[i]]]
                y = archive[keys[rand2[i]]]
                # copy & add variation
                child = None

                # cross overは後で書く
                while child == None or hashable(child[0]) in population_structure_hashes:
                    child = mutate(x.body.copy(), mutation_rate=0.1, num_attempts=50)

                # overwrite structures array w new child
                X[i] = Structure(*child, unique_label.give_label())
                population_structure_hashes[hashable(child[0])] = True

            print("n_mut",n_mut,"\n n_cross",n_cross,"\n X",X)
        # evaluation of the fitness for to_evaluate
        myProblem._evaluate(X, save_path_controller)
        # natural selection
        for x in X:
            __add_to_archive(x, x.desc, archive, kdt)

        ######################save data per generation##########################

        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation),
                                           "structure")
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        structures = [structure for structure in archive.values()]
        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        ### SAVE POPULATION DATA ###
        for i in range(len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        # SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        out+="current evaluation: "+str(curr_evaluation)+"\n"
        f.write(out)
        f.close()

        if b_generation%1==0:
            temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "centroid_score.txt")
            f = open(temp_path, "w")

            out = ""
            for key in archive:
                out += str(key[0]) + "\t\t" + str(key[1]) + "\t\t" + str(archive[key].fitness) + "\n"
            f.write(out)
            f.close()
            cm.draw_voronoi_map(n_niches,experiment_name,generation)

        print(f'FINISHED GENERATION {generation}\n')

        ### update generation ###
        generation += 1
        b_generation += 1
