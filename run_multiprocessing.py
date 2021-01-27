import multiprocessing
import simulation_hydro_network
import time

def main(nodes_num, process_core_name):
    simulation_hydro_network.main(nodes_num, process_core_name)

start = time.perf_counter()

if __name__ == '__main__':
    processes = []
    for k in range(9):
        p = multiprocessing.Process(target = main, args = (int(100), k))
        print(k)
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} seconds(s)')
