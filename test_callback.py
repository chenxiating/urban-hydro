from multiprocessing import Pool, cpu_count
import simulation_hydro_network
import time
import datetime as date
import numpy as np
import os
import sys

try: 
    dt_str = sys.argv[1]
except IndexError:
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")

def main(nodes_num, process_core_name, soil_moisture, mean_rainfall, days, dt_str):
    # simulation_hydro_network.main(nodes_num, process_core_name, soil_moisture, mean_rainfall, days, dt_str)
    print('MP script:', nodes_num, process_core_name, soil_moisture, mean_rainfall, days, dt_str)

start = time.perf_counter()
soil_moisture_list = np.linspace(0, 1, 3)
mean_rainfall_set = np.linspace(5, 0, 3, endpoint=False)
days = 10

if __name__ == '__main__':
    pool = Pool()
    for k in range(1):
        for soil_moisture in soil_moisture_list:
            for mean_rainfall in mean_rainfall_set:
                pool.apply_async(func=main, args = (int(100), k, soil_moisture, mean_rainfall, days, dt_str))
    pool.close()
    pool.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} seconds(s)')
