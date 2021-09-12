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

def simulation(size,beta):
    Gibbs.main(size = size,beta = beta)
def mp_loop(size):
    beta_list = np.array([0.2, 0.4, 0.6, 0.8])
    pool = mp.Pool(processes=mp.cpu_count())
    for beta in beta_list:
        pool.apply(simulation, args=(size,beta,))
    pool.close()
    pool.join()

if __name__ == '__main__':
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    size = 10
    dir_name =  f'gibbs{size}_{dt_str}'

    try: 
        os.chdir(dir_name)
    except FileNotFoundError:
        os.makedirs(dir_name)
        os.chdir(dir_name)
    mp_loop(size) 
    finish = time.perf_counter()