import multiprocessing as mp
from multiprocessing import pool
#import simulation_hydro_network
import numpy as np
import datetime as date
import datetime
import time
import Gibbs

def main(x,y=0):
    z = x**2*np.ones(x)
    z = z+y
    return z
    # print(f'x is {x}, and y is {y}')

def mp_pool(max_x,iter_times=1):
    pool = mp.Pool(processes=mp.cpu_count())
    results = []
    for _ in range(iter_times):
        for x in range(max_x):
            result = pool.apply_async(main, args=(x,))
            results.append(result.get())
    pool.close()
    pool.join()
    return results

def mp_map(max_x,iter_times=1):
    pool = mp.Pool(processes=mp.cpu_count())
    results = []
    for _ in range(iter_times):
        results.append(pool.map(main, list(range(max_x))))
    pool.close()
    pool.join()
    return results

def mp_map_async(max_x,iter_times=1):
    pool = mp.Pool(processes=mp.cpu_count())
    results = []
    for _ in range(iter_times):
        results.append(pool.map_async(main, list(range(max_x))))
    pool.close()
    pool.join()
    return results

# def mp_queue(max_x,iter_times):
#     q = mp.Queue()
#     for _ in range(iter_times):
#         for x in range(max_x):
#             p = mp.Process(target=main,args=(x,q,))
#             p.start()

# def mp_process(max_x,iter_times):
#     processes = []
#     for _ in range(iter_times):
#         for x in range(max_x):
#             p = mp.Process(target = main, args = [x])
#             # print('x: ', x)
#             p.start()
#             processes.append(p)
#     for process in processes:
#         # print('Running process: ', process)
#         process.join()
    
if __name__ == '__main__': 
    my_trees = []
    for _ in range(10):
        uni = Gibbs.Uniform_network(10,10,0.5)
        my_trees.append(uni)
    # print(f'Number of CPU availabe: {mp.cpu_count()}')
    # max_x = 10
    
    # start = time.perf_counter()
    # map_results = mp_map(10,10000)
    # finish = time.perf_counter()
    # print(f'Finished POOL MAP in {round(finish-start,2)} seconds(s)')

    # start = time.perf_counter()
    # map_async_results = mp_map_async(10,10000)
    # finish = time.perf_counter()
    # print(f'Finished POOL MAP ASYNC in {round(finish-start,2)} seconds(s)')

    # start = time.perf_counter()
    # pool_results = mp_pool(10,10)
    # finish = time.perf_counter()
    # print(f'Finished POOL APPLY ASYNC in {round(finish-start,2)} seconds(s)')


    # print(map_results is None)
    # print(map_async_results is None)
    # print(pool_results is None)