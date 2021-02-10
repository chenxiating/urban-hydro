from multiprocessing import Pool
import simulation_hydro_network
import time
import numpy as np
import os

def main(nodes_num, process_core_name, soil_moisture, mean_rainfall):
    simulation_hydro_network.main(nodes_num, process_core_name, soil_moisture, mean_rainfall)
    print('MP script:', nodes_num, process_core_name, soil_moisture, mean_rainfall)

start = time.perf_counter()

if __name__ == '__main__':
    pool = Pool()
    for k in range(10):
        pool.apply_async(func=main, args = (int(100), k, 0.1, 1))
    pool.close()
    pool.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} seconds(s)')