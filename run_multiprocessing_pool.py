import make_SWMM_inp
import pandas as pd
import pickle
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

def simulation(main_df, antecedent_soil_moisture, mean_rainfall_inch,nodes_num,beta,i):
    make_SWMM_inp.main(main_df = main_df, antecedent_soil_moisture=antecedent_soil_moisture, mean_rainfall_inch=mean_rainfall_inch,nodes_num=nodes_num,beta=beta,i=i)

def mp_loop(dt_str,nodes_num, beta):
    soil_moisture_list = np.linspace(0, 1, 10,endpoint=False)
    mean_rainfall_set = np.linspace(13, 3, 10, endpoint=False)
    pool = mp.Pool(processes=mp.cpu_count())
    datafile_name = dt_str + '_full_dataset_'+str(nodes_num)+'-nodes'+'.pickle'
    main_df = None
    for i in range(10):
        for antecedent_soil_moisture in soil_moisture_list:
            for mean_rainfall_inch in mean_rainfall_set:
                pool.apply(simulation, args=(main_df,antecedent_soil_moisture,mean_rainfall_inch,nodes_num,beta,i))
                # pool.apply_async(simulation)
                print(os.getpid())
                # print(soil_moisture)
                # pool.apply_async(simulation, args = (soil_moisture, mean_rainfall, dt_str, soil_nodes_range2, mu, ))
                # print(soil_moisture)
                # print(mean_rainfall)
    pool.close()
    pool.join()

def read_pickle_files(datafile_name):
    all_files = os.listdir()
    main_df = pd.DataFrame()
    for one_file in all_files:
        df = pickle.load(open(one_file, 'rb'))
        main_df = pd.concat([main_df,df],ignore_index=True)
        os.remove(one_file)
    print(main_df.shape)
    f = open(datafile_name,'wb') 
    pickle.dump(main_df,f)
    f.close()

if __name__ == '__main__':
    start = time.perf_counter()
    dt_str = dt_str_gen()
    nodes_num = 100
    beta = 0.5
    datafile_name = dt_str + '_full_dataset_'+str(nodes_num)+'-nodes'+'.pickle'
    folder_name='./SWMM_'+dt_str
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass    
    os.chdir(folder_name)
    print(dt_str)
    mp_loop(dt_str, nodes_num, beta) 
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')
    read_pickle_files(datafile_name)
