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
    make_SWMM_inp.main(main_df = main_df, antecedent_soil_moisture=antecedent_soil_moisture, mean_rainfall_inch=mean_rainfall_inch,nodes_num=nodes_num,beta=beta,i=i,count=0)

def mp_loop(dt_str,nodes_num, beta_list):
    #soil_moisture_list = np.linspace(0, 1, 5,endpoint=False)
    soil_moisture_list = [0.5]
    mean_rainfall_set = np.array([1.44, 1.69, 2.15, 2.59, 3.29, 3.89, 4.55, 5.27, 6.32, 7.19])
    pool = mp.Pool(processes=mp.cpu_count())
    datafile_name = dt_str + '_full_dataset_'+str(nodes_num)+'-nodes'+'.pickle'
    main_df = None
    for i in range(100):
        for antecedent_soil_moisture in soil_moisture_list:
            for mean_rainfall_inch in mean_rainfall_set:
                for beta in beta_list:
                #for count in [0,10,20,30,40,50]:
                    pool.apply_async(simulation, args=(main_df,antecedent_soil_moisture,mean_rainfall_inch,nodes_num,beta,i))
                    # pool.apply_async(simulation)
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
    beta = 0.4
    datafile_name = dt_str + '_full_dataset_'+str(nodes_num)+'-nodes'+'.pickle'
    folder_name='./SWMM_'+dt_str
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass    
    os.chdir(folder_name)
    print(dt_str)
    beta_list= [0.2, 0.5, 0.8]
    mp_loop(dt_str, nodes_num, beta_list) 
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')
    read_pickle_files(datafile_name)
