import Gibbs
import pandas as pd
import pickle
import multiprocessing as mp
import time
import datetime as date
import numpy as np
import os

# def simulation(uni):
#     # Gibbs.main(size = size,beta = beta)
#     deltaH = uni.generate_tree()
#     return deltaH

def generate_one_tree(size,beta,deltaH = None,tree_count=10):
    my_trees = []
    tree_pool = mp.Pool(processes=mp.cpu_count())
    print('Tree pool:', tree_pool)
    for _ in range(tree_count):
        my_trees.append(tree_pool.apply_async(Gibbs.Uniform_network,args=(size,size,beta,deltaH,)).get())
    return my_trees

# def mp_loop(my_trees):
#     pool = mp.Pool(processes=mp.cpu_count())
#     all_deltaH_list = []
#     # group_deltaH = pool.map_async(simulation,my_trees).get()
#     # ungroup_deltaH = [elem for elem in group_deltaH]
#     # [all_deltaH_list.append(i) for i in ungroup_deltaH]
#     print(pool.map_async(simulation,my_trees))
#     print(pool.map_async(simulation,my_trees).get())
#     all_deltaH_list.append(pool.map_async(simulation,my_trees).get())
#     pool.close()
#     pool.join()
#     return(all_deltaH_list)

def generate_forest(size, beta, deltaH=None, tree_count = 2):
    my_trees = generate_one_tree(size=size, beta=beta, deltaH=deltaH, tree_count=tree_count)
    return my_trees

if __name__ == '__main__':
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    size = 10
    beta_list = [0.05, 0.8]
    # beta_list = [0.01]
    # # deltaH = 28
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
        # my_trees = generate_starting_tree(size,beta)
        my_trees = generate_forest(size=size,beta=beta,deltaH=deltaH, tree_count=500)
        print([tree.path_diff_prime for tree in my_trees])
        pool = mp.Pool(processes=mp.cpu_count())
        all_deltaH_list = pd.DataFrame(data=[[tree.beta, tree.path_diff_prime, tree.path_diff] for tree in my_trees], 
        columns=['beta', 'Hp', 'H'])
        # Gibbs.gibbs_pdf(beta,all_deltaH_list)
        name = f'deltaH_beta{beta}_size{size}.pickle'
        f = open(name,'wb')
        pickle.dump(all_deltaH_list,f)
        f.close()
        finish1 = time.perf_counter()
        print(f'Finished beta = {beta} in {round(finish1-start1,2)} seconds(s)')
    finish0 = time.perf_counter()
    print(f'Finished STARTING TREE in {round(finish0-start0,2)} seconds(s)')

    