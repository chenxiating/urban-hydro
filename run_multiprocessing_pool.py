import make_SWMM_inp
import pandas as pd
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

def simulation(main_df, antecedent_soil_moisture, mean_rainfall_inch,i):
    make_SWMM_inp.main(main_df = main_df, antecedent_soil_moisture=antecedent_soil_moisture, mean_rainfall_inch=mean_rainfall_inch,i=i)
    print(antecedent_soil_moisture)
    print(mean_rainfall_inch)

def apply_async(dt_str):
    # soil_nodes_range=[0, 49,50]
    soil_moisture_list = np.linspace(0, 1, 1)
    mean_rainfall_set = np.linspace(13, 3, 5, endpoint=False)
    pool = mp.Pool()
    datafile_name = dt_str + '_full_dataset_'+str(100)+'-nodes'+'.pickle'
    main_df = pd.DataFrame()

    for i in range(4):
        for antecedent_soil_moisture in soil_moisture_list:
            for mean_rainfall_inch in mean_rainfall_set:
        #         mu = np.random.uniform(low=1.6, high=2.2)
                pool.apply_async(simulation, args=(main_df,antecedent_soil_moisture,mean_rainfall_inch,i))
                # pool.apply_async(simulation)
                print(os.getpid())
                # print(soil_moisture)
                # pool.apply_async(simulation, args = (soil_moisture, mean_rainfall, dt_str, soil_nodes_range2, mu, ))
                # print(soil_moisture)
                # print(mean_rainfall)
    pool.close()
    pool.join()

if __name__ == '__main__':
    start = time.perf_counter()
    dt_str = dt_str_gen()
    datafile_name = dt_str + '_full_dataset_'+str(100)+'-nodes'+'.pickle'
    folder_name='./SWMM_'+dt_str
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass    
    os.chdir(folder_name)
    print(dt_str)
    datafile_name = dt_str + '_full_dataset_'+str(100)+'-nodes'+'.pickle'

    apply_async(dt_str)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')
