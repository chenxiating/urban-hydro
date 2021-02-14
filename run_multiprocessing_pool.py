from multiprocessing import Pool
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
print('Date string:', dt_str)

def main(nodes_num, process_core_name, soil_moisture, mean_rainfall, days, dt_str):
    simulation_hydro_network.main(nodes_num, process_core_name, soil_moisture, mean_rainfall, days, dt_str)
    print('MP script:', nodes_num, process_core_name, soil_moisture, mean_rainfall, days, dt_str)

start = time.perf_counter()
soil_moisture_list = np.linspace(0, 1, 20)
mean_rainfall_set = np.linspace(5, 0, 10, endpoint=False)
days = 10

if __name__ == '__main__':
    pool = Pool()
    for soil_moisture in soil_moisture_list:
        for mean_rainfall in mean_rainfall_set:
            for k in range(10):
                pool.apply_async(func=main, args = (int(100), k, soil_moisture, mean_rainfall, days, dt_str),
                callback=print('Loading script: ', k, round(soil_moisture,1), round(mean_rainfall,1)))
    pool.close()
    pool.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} seconds(s)')