import os
import pandas as pd
import pickle
import sys

def compile_datasets(folder_name,overwrite=False):
    os.chdir('/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/'+folder_name)
    all_files = os.listdir()
    some_datafile_name = all_files[0]
    pos0 = some_datafile_name.find('-inch_') + len('-inch_')
    pos1 = some_datafile_name.find('soil')
    filename = some_datafile_name[pos0:pos1]+'.pickle'

    if os.path.exists(filename): 
        return filename
    else: 
        main_df = pd.DataFrame()
        for one_file in all_files:
            df = pickle.load(open(one_file, 'rb'))
            main_df = pd.concat([main_df,df], ignore_index=True)

    datafile_directory='/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/datasets-compiled'
    try: 
        os.chdir(datafile_directory)
    except FileNotFoundError:
        os.makedirs(datafile_directory)
    
    compiled_df = open(filename,'wb')
    pickle.dump(main_df,compiled_df)
    compiled_df.close()
    print(filename)
    return(filename)

if __name__ == '__main__':
    compile_datasets()