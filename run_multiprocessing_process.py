import multiprocessing
import simulation_hydro_network
import time
import numpy as np

def main(nodes_num, process_core_name, soil_moisture, mean_rainfall):
    simulation_hydro_network.main(nodes_num, process_core_name, soil_moisture, mean_rainfall)

start = time.perf_counter()
soil_moisture_list = np.linspace(0, 1, 20)
mean_rainfall_set = np.linspace(0, 5, 10)

if __name__ == '__main__':
    processes = []
    for k in range(10):
        for soil_moisture in soil_moisture_list:
            for mean_rainfall in mean_rainfall_set:
                p = multiprocessing.Process(target = main, args = (int(100), k, soil_moisture, mean_rainfall))
                p.start()
                processes.append(p)

    for process in processes:
        # print('Running process: ', process)
        process.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} seconds(s)')
