import math
import sys,os
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform
import pickle
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')

class Structure():

    def __init__(self, body, connections, label):
        self.body = body
        self.connections = connections

        self.reward = 0
        self.fitness = self.compute_fitness()
        self.fitness2=self.compute_fitness2()

        #descriptor spaceの値を格納するタプル(p1,p2)
        self.desc=None
        self.centroid=None
        
        self.is_survivor = False
        self.prev_gen_label = 0

        self.label = label

    def compute_fitness(self):

        self.fitness = self.reward
        return self.fitness
    
    def compute_fitness2(self):

        self.fitness2 = self.reward
        return self.fitness2

    def set_reward(self, reward):

        self.reward = reward
        self.compute_fitness()
        
    
    def set_reward2(self,reward):
        self.reward = reward
        self.compute_fitness2()

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nF: {self.fitness}\tR: {self.fitness2}\tR: {self.reward}\tID: {self.label}'

    def __repr__(self):
        return self.__str__()

class TerminationCondition():

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def __call__(self, iters):
        return iters >= self.max_iters

    def change_target(self, max_iters):
        self.max_iters = max_iters

def mutate(child, mutation_rate=0.1, num_attempts=10):
    
    pd = get_uniform(5)  
    pd[0] = 0.6 #it is 3X more likely for a cell to become empty

    # iterate until valid robot found
    for n in range(num_attempts):
        # for every cell there is mutation_rate% chance of mutation
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                if draw(mutation) == 0: # mutation
                    child[i][j] = draw(pd)
        
        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))

    # no valid robot found after num_attempts
    return None

def get_percent_survival(gen, max_gen):
    low = 0.0
    high = 0.8
    return ((max_gen-gen-1)/(max_gen-1))**1.5 * (high-low) + low

def total_robots_explored(pop_size, max_gen):
    total = pop_size
    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
    return total

def total_robots_explored_breakpoints(pop_size, max_gen, max_evaluations):
    
    total = pop_size
    out = []
    out.append(total)

    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
        if total > max_evaluations:
            total = max_evaluations
        out.append(total)

    return out

def search_max_gen_target(pop_size, evaluations):
    target = 0
    while total_robots_explored(pop_size, target) < evaluations:
        target += 1
    return target
    


def parse_range(str_inp, rbt_max):
    
    inp_with_spaces = ""
    out = []
    
    for token in str_inp:
        if token == "-":
            inp_with_spaces += " " + token + " "
        else:
            inp_with_spaces += token
    
    tokens = inp_with_spaces.split()

    count = 0
    while count < len(tokens):
        if (count+1) < len(tokens) and tokens[count].isnumeric() and tokens[count+1] == "-":
            curr = tokens[count]
            last = rbt_max
            if (count+2) < len(tokens) and tokens[count+2].isnumeric():
                last = tokens[count+2]
            for i in range (int(curr), int(last)+1):
                out.append(i)
            count += 3
        else:
            if tokens[count].isnumeric():
                out.append(int(tokens[count]))
            count += 1
    return out

def pretty_print(list_org, max_name_length=30):

    list_formatted = []
    for i in range(len(list_org)//4 +1):
        list_formatted.append([])

    for i in range(len(list_org)):
        row = i%(len(list_org)//4 +1)
        list_formatted[row].append(list_org[i])

    print()
    for row in list_formatted:
        out = ""
        for el in row:
            out += str(el) + " "*(max_name_length - len(str(el)))
        print(out)

def get_percent_survival_evals(curr_eval, max_evals):
    low = 0.0
    high = 0.6
    return ((max_evals-curr_eval-1)/(max_evals-1)) * (high-low) + low

def total_robots_explored_breakpoints_evals(pop_size, max_evals):
    
    num_evals = pop_size
    out = []
    out.append(num_evals)
    while num_evals < max_evals:
        num_survivors = max(2,  math.ceil(pop_size*get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        if num_evals > max_evals:
            num_evals = max_evals
        out.append(num_evals)

    return out

class UniqueLabel():
    def __init__(self) -> None:
        self.label=-1
        self.prev_gen_last_label=-1

    def give_label(self):
        self.label+=1
        return self.label
    
    def update_last_label(self):
        self.prev_gen_last_label=self.label
    
    def set_label_start_for_resuming(self,start_label):
        self.label=start_label-1

def save_polulation_hashes(hash_dict,gen,experiment_name):
    print('population structure hash length : '+str(len(hash_dict)))
    save_path=os.path.join(root_dir,'saved_data', experiment_name, "generation_" + str(gen),'population_structure_hashes.pkl')
    with open(save_path,'wb') as f:
        pickle.dump(hash_dict,f)

def load_population_hashes(gen,experiment_name):
    open_path=os.path.join(root_dir,'saved_data',experiment_name,"generation_" + str(gen-1),'population_structure_hashes.pkl')
    with open(open_path,'rb') as f:
        hash_dict=pickle.load(f)
    return hash_dict

def save_archive(archive,gen,experiment_name):
    save_path=os.path.join(root_dir,'saved_data', experiment_name, "generation_" + str(gen),'archive.pkl')
    with open(save_path,'wb') as f:
        pickle.dump(archive,f)

def load_archive(gen,experiment_name):
    open_path=os.path.join(root_dir,'saved_data',experiment_name,"generation_" + str(gen-1),'archive.pkl')
    with open(open_path,'rb') as f:
        archive=pickle.load(f)
    return archive

def calc_curr_evaluation(gen,n_samples,batch_size):
    curr_eval=0
    for i in range(gen):
        if i==0:
            curr_eval+=n_samples
        else:
            curr_eval+=batch_size
    return curr_eval
def load_evaluation(expr_name,gen):
    outputTXT_path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen),'output.txt')
    with open(outputTXT_path,'r') as f:
        for line in f:
            pass
        last_line = line
    curr_evaluation=int(line.split()[2])
    return curr_evaluation


if __name__ == "__main__":

    pop_size = 25
    num_evals = pop_size
    max_evals = 750

    count = 1
    print(num_evals, num_evals, count)
    while num_evals < max_evals:
        num_survivors = max(2,  math.ceil(pop_size*get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        count += 1
        print(new_robots, num_evals, count)

    print(total_robots_explored_breakpoints_evals(pop_size, max_evals))
        
    # target = search_max_gen_target(25, 500)
    # print(target)
    # print(total_robots_explored(25, target-1))
    # print(total_robots_explored(25, target))

    # print(total_robots_explored_breakpoints(25, target, 500))