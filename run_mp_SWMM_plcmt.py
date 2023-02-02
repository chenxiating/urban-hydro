import make_SWMM_inp
import os
import numpy as np
import pickle
import multiprocessing as mp
import pandas as pd
import datetime as date

def read_files(paths):
    file_names = []
    for path in paths: 
        os.chdir(path)
        file_names_path = [one_file for one_file in os.listdir() if (one_file[:7] =='10-grid')]
        all_files = [one_file for one_file in file_names_path]# if (int(one_file[one_file.find('dist-')+5:one_file.find('_ID-')])<300 or int(one_file[one_file.find('dist-')+5:one_file.find('_ID-')])>900)]
        file_names.extend(all_files)
    return file_names

def simulation(main_df, mean_rainfall_set,i,file_name,start):
    main_df = make_SWMM_inp.main(main_df = main_df, antecedent_soil_moisture=0.5, mean_rainfall_set=mean_rainfall_set,
    nodes_num=100,count=10,i=i,mp=True,file_name=file_name,make_cluster=start)
    print(main_df)

def mp_loop(file_names,mean_rainfall_set):
    pool = mp.Pool(processes=mp.cpu_count())
    file_count = len(file_names)
    main_df = pd.DataFrame()
    print(f'There are {file_count} networks in this folder.')
    for i in range(file_count):
        file_name = file_names[i]
        for start in np.arange(1,25,1,dtype=int):
            # test_mp_simulation(mean_rainfall_inch=mean_rainfall_inch, file_name=file_name, start = start)
            print('starting distance:',start)
            pool.apply_async(simulation,args=(main_df,mean_rainfall_set,i,file_name,start,))
        print(file_name,  mp.current_process())
    pool.close()
    pool.join()

def test_read_pickle_file(a):
    df = pickle.load(open(a,'rb'))
    print(df)

def read_pickle_files(datafile_name):
    all_files = os.listdir()
    main_df = pd.DataFrame()
    all_pickle_files = [one_file for one_file in all_files if (one_file[:3] == '0.5') and one_file[-7:] == '.pickle']
    print(all_pickle_files)
    for one_file in all_pickle_files:
        df = pickle.load(open(one_file, 'rb'))
        print(df)
        main_df = pd.concat([main_df,df],ignore_index=True)
        os.remove(one_file)
    print(main_df.shape)
    f = open(datafile_name,'wb') 
    pickle.dump(main_df,f)

    csv_name = datafile_name.replace('pickle', 'csv',1)
    print(csv_name)
    main_df.to_csv(csv_name)
    f.close()

if __name__ == '__main__':
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    # ## Initialize folder and workspace
    # folder_name='./SWMM_placement_20221207_largeH/'
    # try:
    #     os.mkdir(folder_name)
    # except FileExistsError:
    #     pass    

    ## Environmental attributes
    mean_rainfall_set = [1.69, 2.59, 3.29, 4.55]
    # mean_rainfall_set = [1.69]

    ## Read networks and names
    # a = '../gibbs10_20220120-1544/'
    # paths = ['/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20221219-1304_H200+800/']
    # paths = [r'./10-grid_20221207_lt400+gt700']
    path = ['/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20221227-Hp=0.02+0.2/']
    # paths = ['/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20221028-1327',
    # '/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20221028-1032',
    # '/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20220929-2303-100nodes',
    # '/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20221027-1146']
    file_names = read_files(path)
    mp_loop(file_names,mean_rainfall_set)
    read_pickle_files(rf'{path[0]}/{dt_str}_GI_distance_summary_highly_impervious.pickle')