import make_SWMM_inp
import os
import numpy as np
import pickle
import multiprocessing as mp
import pandas as pd

def read_files(a):
    files = os.listdir(a)
    file_names = list()
    for file in files:
        b = a+file
        file_names.append(b)
    return file_names

def simulation(main_df, mean_rainfall_inch,i,file_name,start):
    main_df = make_SWMM_inp.main(main_df = main_df, antecedent_soil_moisture=0.5, mean_rainfall_inch=mean_rainfall_inch,
    nodes_num=100,count=10,i=i,fixing_graph=True,changing_diam=True,file_name=file_name,make_cluster=start)

def test_simulation(mean_rainfall_inch, file_name):
    main_df = pd.DataFrame()
    print(file_name)
    simulation(main_df=main_df, mean_rainfall_inch=mean_rainfall_inch, i=5, file_name=file_name)

def mp_loop(file_names,mean_rainfall_set):
    pool = mp.Pool(processes=mp.cpu_count())
    file_count = len(file_names)
    main_df = pd.DataFrame()
    print(f'There are {file_count} networks in this folder.')
    for i in range(file_count):
        print(i)
        file_name = file_names[i]
        for mean_rainfall_inch in mean_rainfall_set:
            for start in np.arange(1,15,1,dtype=int):
                print('starting distance:',start)
                pool.apply_async(simulation,args=(main_df,mean_rainfall_inch,i,file_name,start,))
            print(file_name, f'rainfall is {mean_rainfall_inch}"', mp.current_process())
    pool.close()
    pool.join()

def test_read_pickle_file(a):
    df = pickle.load(open(a,'rb'))
    print(df)

def read_pickle_files(datafile_name):
    all_files = os.listdir()
    main_df = pd.DataFrame()
    all_pickle_files = [one_file for one_file in all_files if 'graph' not in one_file]
    for one_file in all_pickle_files:
        df = pickle.load(open(one_file, 'rb'))
        main_df = pd.concat([main_df,df],ignore_index=True)
        os.remove(one_file)
    print(main_df.shape)
    f = open(datafile_name,'wb') 
    pickle.dump(main_df,f)
    f.close()

if __name__ == '__main__':
    ## Initialize folder and workspace
    folder_name='./SWMM_placement_128_20220517/'
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass    
    os.chdir(folder_name)

    ## Environmental attributes
    mean_rainfall_set = [1.69, 2.59, 3.29, 4.55]

    ## Read networks and names
    # a = '../gibbs10_20220120-1544/'
    a = '../gibbs10_dist128/'
    file_names = read_files(a)
    mp_loop(file_names,mean_rainfall_set)
    read_pickle_files('GI_distance_summary.pickle')
    # test_simulation(1.44,file_name=file_names[0])
    # test_read_pickle_file('/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_placement/0.5_1.44-inch_beta-0_0_start-cluster-11.pickle')