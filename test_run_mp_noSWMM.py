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

results = []

def dt_str_gen():
    try: 
        dt_str = sys.argv[1]
    except IndexError:
        today = date.datetime.today()
        dt_str = today.strftime("%Y%m%d-%H%M")
    return dt_str

def simulation(main_df, antecedent_soil_moisture, mean_rainfall_inch,nodes_num,i):
    # main_df  = make_SWMM_inp.main(main_df = main_df, antecedent_soil_moisture=antecedent_soil_moisture, mean_rainfall_inch=mean_rainfall_inch,nodes_num=nodes_num,i=i)
    out_df = [antecedent_soil_moisture, mean_rainfall_inch, nodes_num]
    print('out_df to list:',out_df.values.tolist())
    # return out_df.values.tolist()
    return antecedent_soil_moisture

def concat_df(df):
   # main_df.append(df)
   # main_df.concat(df)
    # print(df)
    # main_df = pd.concat([main_df, df]) 
    results.extend(df) 
    # results.append(df) 

def mp_loop(main_df,dt_str,nodes_num):
    soil_moisture_list = np.linspace(0, 1, 2,endpoint=False)
    mean_rainfall_set = np.linspace(13, 3, 1, endpoint=False)
    pool = mp.Pool(processes=mp.cpu_count())
    datafile_name = dt_str + '_full_dataset_'+str(nodes_num)+'-nodes'+'.pickle'
    main_df = pd.DataFrame()
    results = []

    for i in range(3):
        print('i',i)
        for antecedent_soil_moisture in soil_moisture_list:
            for mean_rainfall_inch in mean_rainfall_set:
                # results = pool.apply(simulation, args=(main_df,antecedent_soil_moisture,mean_rainfall_inch,nodes_num,i))
                # pool.apply(simulation, args=(main_df,antecedent_soil_moisture,mean_rainfall_inch,nodes_num,i,),callback=concat_df)
                result = simulation(main_df, antecedent_soil_moisture, mean_rainfall_inch,nodes_num,i)
                results.append(result)
                print(os.getpid())
    pool.close()
    pool.join()
    # main_df = pd.concat(results)
    # p.start()
    # p.join()
    return results

if __name__ == '__main__':
    start = time.perf_counter()
    dt_str = dt_str_gen()
    nodes_num = 20
    datafile_name = dt_str + '_full_dataset_'+str(nodes_num)+'-nodes'+'.pickle'
    folder_name='./SWMM_'+dt_str
    # results = []
    # main_df = pd.DataFrame()
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass    
    os.chdir(folder_name)
    print(dt_str)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')
    main_df = pd.DataFrame(results)
    print('results:',results)
    print('main df:',main_df)
    print(main_df.shape)
    f = open(datafile_name,'wb')
    pickle.dump(main_df,f)
    f.close()
