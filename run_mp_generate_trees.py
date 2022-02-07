from sklearn import tree
import Gibbs
import pandas as pd
import pickle
import multiprocessing as mp
import time
import datetime as date
import numpy as np
import os
import sys
from scipy.special import factorial

def simulation(uni):
    # Gibbs.main(size = size,beta = beta)
    deltaH_list = uni.generate_tree(k=4000)
    return deltaH_list

def generate_starting_tree(size,beta,deltaH,tree_count=10):
    my_trees = []
    tree_pool = mp.Pool(processes=mp.cpu_count())
    print('Tree pool:', tree_pool)
    for _ in range(tree_count):
        my_trees.append(tree_pool.apply_async(Gibbs.Uniform_network,args=(size,size,beta,deltaH,)).get())
    return my_trees

def mp_loop(my_trees):
    pool = mp.Pool(processes=mp.cpu_count())
    all_deltaH_list = []
    group_deltaH = pool.map_async(simulation,my_trees).get()
    ungroup_deltaH = [elem for elem in group_deltaH]
    [all_deltaH_list.append(i) for i in ungroup_deltaH]
    pool.close()
    pool.join()
    return(all_deltaH_list)

def test(size, beta, deltaH=None, tree_count = 10):
    my_trees = generate_starting_tree(size=size, beta=beta, deltaH=deltaH, tree_count=tree_count)

if __name__ == '__main__':
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    size = 10
    beta_list = np.array([0.5])
    deltaH = 30
    dir_name =  f'gibbs{size}_{dt_str}'

    try: 
        os.chdir(dir_name)
    except FileNotFoundError:
        os.makedirs(dir_name)
        os.chdir(dir_name)

    start0 = time.perf_counter()
    test(size=size,beta=1.2,deltaH=25,tree_count=100)
    # # mp_loop(size)
    # for beta in beta_list:
    #     start1 = time.perf_counter()
    #     my_trees = generate_starting_tree(size,beta)
    #     all_deltaH_list = mp_loop(my_trees)
    #     Gibbs.gibbs_pdf(beta,all_deltaH_list)
    #     name = f'deltaH_beta{beta}_size{size}.pickle'
    #     f = open(name,'wb')
    #     pickle.dump(all_deltaH_list,f)
    #     f.close()
    #     finish1 = time.perf_counter()
    #     print(f'Finished beta = {beta} in {round(finish1-start1,2)} seconds(s)')
    # finish0 = time.perf_counter()
    # print(f'Finished STARTING TREE in {round(finish0-start0,2)} seconds(s)')

    