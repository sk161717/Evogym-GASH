pruning_num=4
params = \
    {
        "pop_size": 32,
        "pruning_timing1": 50,
        "pruning_timing2": 100,
        'pruning_timing3': 250,
        'pruning_timing4': 500,
        "timing1_border": 16,
        "timing2_border": 8,
        "timing3_border": 4,
        "timing4_border": 2,
    }

eval_require_dict={}
eval_border_dict={}

eval_require_dict[params["pruning_timing1"]]=params["pop_size"]
eval_require_dict[params["pruning_timing2"]]=params["timing1_border"]
eval_require_dict[params["pruning_timing3"]]=params["timing2_border"]
eval_require_dict[params["pruning_timing4"]]=params["timing3_border"]

eval_border_dict[params["pruning_timing1"]]=params["timing1_border"]
eval_border_dict[params["pruning_timing2"]]=params["timing2_border"]
eval_border_dict[params["pruning_timing3"]]=params["timing3_border"]
eval_border_dict[params["pruning_timing4"]]=params["timing4_border"]
'''
pruning_num=2

params = \
    {
        "pop_size": 4,
        "pruning_timing1": 50,
        "pruning_timing2": 100,
        "timing1_border": 2,
        "timing2_border": 1,
    }

eval_require_dict={}
eval_border_dict={}

eval_require_dict[params["pruning_timing1"]]=params["pop_size"]
eval_require_dict[params["pruning_timing2"]]=params["timing1_border"]


eval_border_dict[params["pruning_timing1"]]=params["timing1_border"]
eval_border_dict[params["pruning_timing2"]]=params["timing2_border"]
'''

def judge_timing(j):
    for i in range(1,pruning_num+1):
        index="pruning_timing"+str(i)
        if j==params[index]:
            return True
    return False
