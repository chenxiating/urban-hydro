import simulation_hydro_network
import multiprocessing as mp
import time
import datetime as date
import numpy as np
import os
import sys
from scipy.special import factorial

def dt_str_gen():
    try: 
        dt_str = sys.argv[1]
    except IndexError:
        today = date.datetime.today()
        dt_str = today.strftime("%Y%m%d-%H%M")
    return dt_str

def simulation(soil_moisture, mean_rainfall, dt_str, soil_nodes_range, mu):
    kernel = lambda x: np.exp(-mu)*mu**x/factorial(x)
    simulation_hydro_network.main(days=10,antecedent_soil_moisture=soil_moisture, mean_rainfall_inch=mean_rainfall, process_core_name=os.getpid(),dt_str=dt_str,soil_nodes_range=soil_nodes_range, kernel=kernel)
    return os.getpid()


def apply_async(dt_str):
    soil_nodes_range1=[0, 49,50]
    soil_moisture_list = np.linspace(0, 1, 5)
    mean_rainfall_set = np.linspace(13, 3, 10, endpoint=False)
    pool = mp.Pool()
    for _ in range(10):
        for soil_moisture in soil_moisture_list:
            for mean_rainfall in mean_rainfall_set:
                mu = np.random.uniform(low=1.6, high=2.2)
                pool.apply_async(simulation, args = (soil_moisture, mean_rainfall, dt_str, soil_nodes_range, mu, ))
                # print(soil_moisture)
                # pool.apply_async(simulation, args = (soil_moisture, mean_rainfall, dt_str, soil_nodes_range2, mu, ))
                # print(soil_moisture)
                # print(mean_rainfall)
    pool.close()
    pool.join()

if __name__ == '__main__':
    start = time.perf_counter()
    dt_str = dt_str_gen()
    apply_async(dt_str)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')
