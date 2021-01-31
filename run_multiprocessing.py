import multiprocessing
import simulation_hydro_network
import datetime as date
import datetime
import time
import os

def setup():
	today = date.datetime.today()
	dt_str = today.strftime("%Y%m%d-%H%M")
	time_openf = time.time()
	file_directory = os.path.dirname(os.path.abspath(__file__))
	datafile_directory=file_directory +'/datafiles_'+dt_str
	print('os.path.exists(datafile_directory)', os.path.exists(datafile_directory))
	if not os.path.exists(datafile_directory):
		os.makedirs(datafile_directory)
	os.chdir(datafile_directory)
	print(datafile_directory)
	return dt_str
	
def main(nodes_num, process_core_name):
	simulation_hydro_network.main(nodes_num, process_core_name)

start = time.perf_counter()
#dt_str0 = setup()
if __name__ == '__main__': 
	processes = []
	for k in range(5):
		print('in Multiprocessing run script before calling process.start()')
		p = multiprocessing.Process(target = main, args = (int(100), k))
		print('Process core number: ', k)
		p.start()	
		print('in Multiprocessing run script after calling process.start()')
		processes.append(p)

	for process in processes:
		print('Running process: ', process)
		process.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} seconds(s)')
