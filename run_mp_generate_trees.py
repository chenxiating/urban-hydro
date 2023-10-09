"""
run_mp_generate_trees.py 
@author: Xiating Chen
Last Edited: 2023/10/08


This code is to run Gibbs to generate many trees. It should be run before running 
'run_mp_SWMM_coverage.py' and 'run_mp_SWMM_plcmt.py'. 

"""

import Gibbs
import pandas as pd
import pickle
import multiprocessing as mp
import time
import datetime as date
import os

def generate_one_tree(size,beta,deltaH = None,tree_count=10):
    my_trees = []
    tree_pool = mp.Pool(processes=mp.cpu_count())
    print('Tree pool:', tree_pool)
    for _ in range(tree_count):
        my_trees.append(tree_pool.apply_async(Gibbs.Uniform_network,args=(size,size,beta,deltaH,)).get())
    return my_trees

def generate_forest(size, beta, deltaH=None, tree_count = 2):
    my_trees = generate_one_tree(size=size, beta=beta, deltaH=deltaH, tree_count=tree_count)
    return my_trees

if __name__ == '__main__':
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    size = 10
    beta_list = [0.05, 0.8] # example
    dir_name =  f'gibbs{size}_{dt_str}'

    try: 
        os.chdir(dir_name)
    except FileNotFoundError:
        os.makedirs(dir_name)
        os.chdir(dir_name)

    start0 = time.perf_counter()

    deltaH = None
    for beta in beta_list:
        start1 = time.perf_counter()
        my_trees = generate_forest(size=size,beta=beta,deltaH=deltaH, tree_count=10)
        pool = mp.Pool(processes=mp.cpu_count())
        all_deltaH_list = pd.DataFrame(data=[[tree.beta, tree.path_diff_prime, tree.path_diff] for tree in my_trees], 
        columns=['beta', 'Hp', 'H'])
        Gibbs.gibbs_pdf(beta,all_deltaH_list) # put the PDF of generated graphs
        name = f'deltaH_beta{beta}_size{size}.pickle'
        f = open(name,'wb')
        pickle.dump(all_deltaH_list,f)
        f.close()
        finish1 = time.perf_counter()
        print(f'Finished beta = {beta} in {round(finish1-start1,2)} seconds(s)')
    finish0 = time.perf_counter()
    print(f'Finished STARTING TREE in {round(finish0-start0,2)} seconds(s), In folder {dir_name}')
    