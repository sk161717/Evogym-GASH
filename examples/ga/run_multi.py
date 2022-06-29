import os
import numpy as np
import shutil
import random
import math

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
population_structure_hashes = {}
generation=0
save_path_controller1=None
save_path_controller2=None
curr_evaluation=0

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import mutate, TerminationCondition, Structure,UniqueLabel,save_polulation_hashes
from make_gifs_multi import Job
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.factory import get_visualization
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter

unique_label=UniqueLabel()


class MyProblem(Problem):

    def __init__(self,algorithm,tc,num_cores,env_name1,env_name2) -> None:
        super().__init__(n_var=1, n_obj=2, n_constr=0)
        self.algorithm=algorithm
        self.tc = tc
        self.num_cores=num_cores
        self.env_name1=env_name1
        self.env_name2=env_name2

    def _evaluate(self, X, out, *args, **kwargs):
        
        fitness1=self.RunOneEnv(X,self.env_name1,save_path_controller1,1)
        fitness2=self.RunOneEnv(X,self.env_name2,save_path_controller2,2)
        out["F"] = np.column_stack([fitness1, fitness2])
    
    def RunOneEnv(self,X,env_name,save_path_controller,env_index):
        global curr_evaluation

        group = mp.Group()
        for x in X:
            ppo_args = ((x[0].body, x[0].connections), self.tc, (save_path_controller, x[0].label),env_name)
            if env_index==1:
                group.add_job(run_ppo, ppo_args, callback=x[0].set_reward)
            elif env_index==2:
                group.add_job(run_ppo, ppo_args, callback=x[0].set_reward2)
            curr_evaluation+=1
        group.run_jobs(self.num_cores)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        f=np.full((X.shape[0],1),None,dtype=object)
        for i,x in enumerate(X):
            if env_index==1:
                x[0].compute_fitness()
                f[i][0]=-x[0].fitness
            elif env_index==2:
                x[0].compute_fitness2()
                f[i][0]=-x[0].fitness2
        return f


class MySampling(Sampling):
    def __init__(self,structure_shape,experiment_name,is_continuing=False):
        super().__init__()
        self.structure_shape=structure_shape
        self.experiment_name=experiment_name
        self.is_continuing=is_continuing

    def _do(self, problem, n_samples, **kwargs):
        global save_path_controller1
        global save_path_controller2

        save_path_controller1 = os.path.join(root_dir, "saved_data", self.experiment_name, "generation_" + str(generation), "controller1")
        try:
            os.makedirs(save_path_controller1)
        except:
            pass
            
        save_path_controller2 = os.path.join(root_dir, "saved_data", self.experiment_name, "generation_" + str(generation), "controller2")
        try:
            os.makedirs(save_path_controller2)
        except:
            pass

        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            temp_structure = sample_robot(self.structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(self.structure_shape)

            X[i, 0] = Structure(*temp_structure, unique_label.give_label())
            population_structure_hashes[hashable(temp_structure[0])] = True

        return X

class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):

        return X

class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):

        # for each individual
        for i in range(len(X)):
            child=None

            while child ==None or hashable(child[0]) in population_structure_hashes:
                child = mutate(X[i,0].body.copy(), mutation_rate = 0.1, num_attempts=50)

            # overwrite structures array w new child
            X[i,0] = Structure(*child, unique_label.give_label())
            population_structure_hashes[hashable(child[0])] = True

        return X

class MyCallback(Callback):
        def __init__(self,experiment_name) -> None:
            super().__init__()
            self.experiment_name = experiment_name

        def notify(self, algorithm):
            global generation
            global save_path_controller1
            global save_path_controller2
            global curr_evaluation
            
            ### MAKE GENERATION DIRECTORIES ###
            save_path_structure = os.path.join(root_dir, "saved_data", self.experiment_name, "generation_" + str(generation), "structure")
            try:
                os.makedirs(save_path_structure)
            except:
                pass

            save_polulation_hashes(population_structure_hashes,generation,self.experiment_name)
            
            structures=algorithm.pop.get("X")
            structures = sorted(structures, key=lambda structure: structure[0].fitness, reverse=True)

            ### SAVE POPULATION DATA ###
            for i in range (len(structures)):
                temp_path = os.path.join(save_path_structure, str(structures[i][0].label))
                np.savez(temp_path, structures[i][0].body, structures[i][0].connections)
            
            #SAVE RANKING TO FILE
            temp_path = os.path.join(root_dir, "saved_data", self.experiment_name, "generation_" + str(generation), "output.txt")
            f = open(temp_path, "w")

            out = ""
            for structure in structures:
                out += str(structure[0].label) + "\t\t" + str(structure[0].fitness) + "\t\t" + str(structure[0].fitness2) + "\n"
            out+="current evaluation: "+str(curr_evaluation)+"\n"
            f.write(out)
            f.close()

            print(f'FINISHED GENERATION {generation}\n')
           
            # plot pareto front
            plot = Scatter(title = "Objective Space", labels="f")
            front=algorithm.pop.get("F")
            plot.add(-front)
            temp_path = os.path.join(root_dir, "saved_data", self.experiment_name,"generation_" + str(generation), "pareto_front.pdf")
            plot.save(temp_path)

            #make directory for next generation
            generation+=1
            save_path_controller1 = os.path.join(root_dir, "saved_data", self.experiment_name, "generation_" + str(generation), "controller1")
            try:
                os.makedirs(save_path_controller1)
            except:
                pass
                
            save_path_controller2 = os.path.join(root_dir, "saved_data", self.experiment_name, "generation_" + str(generation), "controller2")
            try:
                os.makedirs(save_path_controller2)
            except:
                pass
            unique_label.update_last_label()




def run_multi_ga(experiment_name, structure_shape, pop_size, total_generation, train_iters, num_cores,env_name1,env_name2,seed=0):
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
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'TOTAL_GENERATION: {total_generation}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.write(f'ENV_NAMES1: {env_name1}\n')
        f.write(f'ENV_NAMES2: {env_name2}\n')
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
                total_generation = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max generation: {total_generation}, train iters {train_iters}.')
        
        f.close()

    #Multiobjective Initialization
    algorithm=NSGA2(
        pop_size=pop_size,
        sampling=MySampling(structure_shape,experiment_name),
        mutation=MyMutation(),
        crossover=MyCrossover(),
        eliminate_duplicates=False
    )

    res = minimize(MyProblem(algorithm,tc,num_cores,env_name1,env_name2),
                algorithm,
                ('n_gen',total_generation),
                seed=seed,
                callback=MyCallback(experiment_name),
                verbose=False)

    plot = Scatter(title = "Objective Space", labels="f")
    plot.add(-res.F)
    temp_path = os.path.join(root_dir, "saved_data", experiment_name, "pareto_front_final.pdf")
    plot.save(temp_path)

        