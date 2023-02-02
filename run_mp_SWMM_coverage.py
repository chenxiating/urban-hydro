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

def simulation(main_df, mean_rainfall_set,i,file_name,count):
    main_df = make_SWMM_inp.main(main_df = main_df, antecedent_soil_moisture=0.5, mean_rainfall_set=mean_rainfall_set,
    nodes_num=100,count=count,i=i,mp=True,file_name=file_name,make_cluster=False)
    print(main_df)

def mp_loop(file_names,mean_rainfall_set):
    pool = mp.Pool(processes=mp.cpu_count())
    file_count = len(file_names)
    main_df = pd.DataFrame()
    print(f'There are {file_count} networks in this folder.')
    for i in range(file_count):
        file_name = file_names[i]
        print(file_name)
        for count in np.arange(0,60,10,dtype=int):
            pool.apply_async(simulation,args=(main_df,mean_rainfall_set,i,file_name,count,))
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
    path = ['/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20221227-Hp=0.02+0.2/']
    # paths = [r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_test/']
    # paths = ['/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20221028-1327',
    # '/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20221028-1032',
    # '/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20220929-2303-100nodes',
    # '/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20221027-1146']
    file_names = read_files(path)
#     print(len(file_names), '0.1: ', len([k for k in file_names if k[13]=='0']),
# '1: ', len([k for k in file_names if k[13]=='1']))
    mp_loop(file_names,mean_rainfall_set)
    read_pickle_files(f'{path[0]}/{dt_str}_GI_coverage_summary_highly_imp.pickle')
    print(f'Total networks: {len(file_names)} * 4 * 6 = {24*len(file_names)}')
    