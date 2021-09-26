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
    uni.generate_tree(mode='Gibbs',k=4000)

def generate_starting_tree(size,beta,tree_count=1000):
    my_trees = []
    tree_pool = mp.Pool(processes=mp.cpu_count())
    for _ in range(tree_count):
        my_trees.append(tree_pool.apply_async(Gibbs.Uniform_network,args=(size,size,beta,)).get())
    return my_trees

def mp_loop(my_trees):
    pool = mp.Pool(processes=mp.cpu_count())
    all_deltaH_list = []
    for _ in range(1000):
        all_deltaH_list.append(pool.map_async(simulation,my_trees))
    pool.close()
    pool.join()

if __name__ == '__main__':
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    size = 10
    beta_list = np.array([0.2, 0.4, 0.6, 0.8])
    dir_name =  f'gibbs{size}_{dt_str}'

    try: 
        os.chdir(dir_name)
    except FileNotFoundError:
        os.makedirs(dir_name)
        os.chdir(dir_name)

    start0 = time.perf_counter()
    # mp_loop(size)
    for beta in beta_list:
        start1 = time.perf_counter()
        my_trees = generate_starting_tree(size,beta)
        all_deltaH_list = mp_loop(my_trees)
        Gibbs.gibbs_pdf(beta,all_deltaH_list)
        name = f'deltaH_beta{beta}.pickle'
        f = open(name,'wb')
        pickle.dump(all_deltaH_list,f)
        f.close()
        finish1 = time.perf_counter()
        print(f'Finished ONE BETA in {round(finish1-start1,2)} seconds(s)')
    finish0 = time.perf_counter()
    print(f'Finished STARTING TREE in {round(finish0-start0,2)} seconds(s)')

    