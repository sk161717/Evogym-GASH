import math
import sys,os,random
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import utils.pruning_params as pp
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')

class Structure():

    def __init__(self, body, connections, label,parent_label):
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
        self.parent_label=parent_label

        self.eval_history=[]

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

    def give_label(self):
        self.label+=1
        return self.label
    
    def set_label_start_for_resuming(self,start_label):
        self.label=start_label-1

def save_polulation_hashes(hash_dict,gen,experiment_name):
    save_path=os.path.join(root_dir,'saved_data', experiment_name, "generation_" + str(gen),'population_structure_hashes.pkl')
    with open(save_path,'wb') as f:
        pickle.dump(hash_dict,f)

def load_population_hashes(gen,experiment_name):
    open_path=os.path.join(root_dir,'saved_data',experiment_name,"generation_" + str(gen-1),'population_structure_hashes.pkl')
    with open(open_path,'rb') as f:
        hash_dict=pickle.load(f)
    return hash_dict

def save_archive(archive,gen,experiment_name,filename='archive.pkl'):
    save_path=os.path.join(root_dir,'saved_data', experiment_name, "generation_" + str(gen),filename)
    with open(save_path,'wb') as f:
        pickle.dump(archive,f)

def load_archive(gen,experiment_name,filename='archive.pkl'):
    open_path=os.path.join(root_dir,'saved_data',experiment_name,"generation_" + str(gen-1),filename)
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

def tournament_selection(structures,pop_size):
    selected_indices = random.sample(range(pop_size), 2)
    a=structures[selected_indices[0]]
    b=structures[selected_indices[1]]
    if a.fitness>=b.fitness:
        return selected_indices[0]
    else:
        return selected_indices[1]

def write_output(structures,expr_name,gen,num_evaluations):
    temp_path = os.path.join(root_dir, "saved_data", expr_name, "generation_" + str(gen), "output.txt")
    f = open(temp_path, "w")

    out = ""
    for structure in structures:
        out += str(structure.label) + "\t\t" + str(structure.fitness) + "\t\t"+str(structure.fitness2) + "\n"
    out+="current evaluation: "+str(num_evaluations)+"\n"
    f.write(out)
    f.close()
    
def add_lineage(structures,expr_name,gen):
    path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen),'lineage.txt')
    child_parent_dict={}
    with open(path,'w') as f:
        out=''
        if gen!=0:
            past_path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen-1),'lineage.txt')
            with open(past_path,'r') as f_past:
                for line in f_past:
                    child_parent_dict[int(line.split(' ')[0])]=line.split(' ')[1]
                    out+= str(line.split(' ')[0])+  ' ' + str(line.split(' ')[1])
        for structure in structures:
            if structure.label not in child_parent_dict:
                out+= str(structure.label)+  ' ' + str(structure.parent_label) +'\n'
        f.write(out)

def max_fit_list_single(expr_name,generation):
    fitness_gen=[]
    evaluation_list=[]
    for i in range(generation+1):
        log_dir = os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(i),'output.txt')
        with open(log_dir) as f:
            fitnesses=[]
            for line in f:
                try:
                    fitnesses.append(float(line.split()[1]))
                except:
                    evaluation_list.append(int(line.split()[2]))
            fitness_gen.append(max(fitnesses))
    return fitness_gen,evaluation_list

def max_fit_list_multi(expr_name,generation):
    fitness1_gen=[]
    fitness2_gen=[]
    evaluation_list=[]
    for i in range(generation+1):
        log_dir = os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(i),'output.txt')
        with open(log_dir) as f:
            fitnesses1=[]
            fitnesses2=[]
            for line in f:
                try:
                    fitnesses1.append(float(line.split()[1]))
                    fitnesses2.append(float(line.split()[2]))
                except:
                    evaluation_list.append(int(line.split()[2]))
            fitness1_gen.append(max(fitnesses1))
            fitness2_gen.append(max(fitnesses2))
    return fitness1_gen,fitness2_gen,evaluation_list

def plot_graph(expr_name,gen,is_multi,is_eval_base=True):
    if is_multi:
        fitness1_list,fitness2_list,evaluation_list=max_fit_list_multi(expr_name,gen)
        plot_one_graph(expr_name,gen,fitness1_list,evaluation_list)
        plot_one_graph(expr_name,gen,fitness2_list,evaluation_list,index=2)
    else:    
        fitness1_list,evaluation_list=max_fit_list_single(expr_name,gen)
        plot_one_graph(expr_name,gen,fitness1_list,evaluation_list)
    

def plot_one_graph(expr_name,gen,fitness_list,evaluation_list,index=1,is_eval_base=True):
    fig = plt.figure(figsize=(12, 8)) #...1
    
    # Figure内にAxesを追加()
    ax = fig.add_subplot(111) #...2
    if is_eval_base==False:
        evaluation_list=np.array(evaluation_list)/2
    ax.plot(evaluation_list, fitness_list) 
    plt.xlabel('evaluations' if is_eval_base else 'evaluated design')
    plt.ylabel('score of '+str(index))

    path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen),'score'+str(index)+'.pdf')
    # プロット表示(設定の反映)
    plt.savefig(path)


def calc_edit_distance(structure1,structure2):
    dist=0
    for i in range(len(structure1)):
        for j in range(len(structure2)):
            if structure1[i][j]!=structure2[i][j]:
                dist+=1
    return dist


def compute_novelty(X,novelty_archive):
    k=3
    novelty_threshold=10
    pop_size=X.shape[0]
    f2=np.full((pop_size,1),None,dtype=object)

    if len(novelty_archive)==0:
        for i in range(pop_size):
            novelty_archive.append(X[i][0])
            f2[i][0]=novelty_threshold
        return f2

    for i in range(pop_size):
        dist_arr=[]
        for j in range(len(novelty_archive)):
                dist_arr.append(calc_edit_distance(X[i][0].body,novelty_archive[j].body))

        latter_index = k+1 if len(dist_arr)>k+1 else len(dist_arr)
        topk=sorted(dist_arr)[:latter_index]
        f2[i][0]=statistics.mean(topk)
        # save archive
        if f2[i][0]>novelty_threshold:
            novelty_archive.append(X[i][0])
    print('archive size:',len(novelty_archive))
    return f2

def compute_novelty_for_list(structures,novelty_archive,pop_size):
    X = np.full((pop_size, 1), None, dtype=object)
    for i in range(pop_size):
        X[i][0]=structures[i]
    f2=compute_novelty(X,novelty_archive)
    for i in range(pop_size):
        structures[i].fitness2=f2[i][0]

def save_eval_history(expr_name,gen,eval_history,label,curr_evals=None):
    if curr_evals==None:
        path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen),'eval_history.txt')
    else:
        path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen),'eval_history_'+str(curr_evals)+'.txt')
    out = ""
    str_eval_hist=list(map(str,eval_history))
    out+= str(label) +'\t\t'+ '\t\t'.join(str_eval_hist) + '\n'
    with open(path,'a') as f:
        f.write(out)

def save_evaluated_structures(expr_name,generation,structures):
    save_path_structure = os.path.join(root_dir, "saved_data", expr_name, "generation_" + str(generation), "structure")
    try:
        os.makedirs(save_path_structure)
    except:
        pass

    ### SAVE POPULATION DATA ###
    for i in range(len(structures)):
        temp_path = os.path.join(save_path_structure, str(structures[i].label))
        np.savez(temp_path, structures[i].body, structures[i].connections)

def save_evaluated_score(expr_name,gen,structures):
    temp_path = os.path.join(root_dir, "saved_data", expr_name, "generation_" + str(gen), "evaluated_score.txt")
    f = open(temp_path, "w")

    out = ""
    for structure in structures:
        out += str(structure.label) + "\t\t" + str(structure.fitness) + "\t\t"+str(structure.fitness2) + "\n"
    f.write(out)
    f.close()

def is_pruned(label,curr_evals,expr_name,gen,eval_interval):
    data_arr=[]
    path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen),'eval_history_'+str(curr_evals)+'.txt')
    
    with open(path) as f:
        for line in f:
            items=list(map(float,line.split('\t\t')))
            if len(items)!=(curr_evals//eval_interval+1):
                items=items[0:1]+items[2:]
            data_arr.append(items)
            
    df=pd.DataFrame(data_arr)
    start_index=curr_evals//eval_interval-4
    end_index=curr_evals//eval_interval+1
    
    
    df.iloc[:,0]=df.iloc[:,0].apply(int)
    #df.loc[:,'var']=df.iloc[:,start_index:end_index].var(axis=1)
    df.loc[:,'max']=df.iloc[:,1:end_index].max(axis=1)
    #df.loc[:,'var_rank']=df['var'].rank(ascending=False)
    df.loc[:,'max_rank']=df['max'].rank(ascending=False)
    #df.loc[:,'var_max']=df.loc[:,'var_rank']*df.loc[:,'max_rank']
    df.loc[:,'var_max']=df.loc[:,'max_rank']
    df.loc[:,'vm_rank']=df['var_max'].rank(method='first',ascending=True)
    
    rank=df[df.iloc[:,0]==label].loc[:,'vm_rank'].iloc[0]
    if rank > pp.eval_border_dict[curr_evals]:
        return True
    else:
        return False


def is_stop(curr_evals,expr_name,gen,pop_size=pp.params['pop_size']):
    path=os.path.join(root_dir,'saved_data',expr_name,'generation_'+str(gen),'eval_history_'+str(curr_evals)+'.txt')
    
    with open(path) as f:
        file_length=len(f.readlines())
    
    

    required=pp.eval_require_dict[curr_evals] 

    if file_length<required:
        return True
    elif file_length==required:
        return False
    elif file_length>required:
        print('Error : file length exceeds required {}'.format(required))
        exit(1)
        


    
            

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