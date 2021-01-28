import os
import pandas as pd
import pickle

def compile_datasets(folder_name):
    os.chdir('/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/'+folder_name)
    all_files = os.listdir()
    some_datafile_name = all_files[0]
    pos0 = some_datafile_name.find('network_count')
    filename = some_datafile_name[0:pos0]+'.pickle'

    datafile_directory='/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro_datasets-compiled'
    try: 
        os.chdir(datafile_directory)
    except FileNotFoundError:
        os.makedirs(datafile_directory)

    if os.path.exists(filename): 
        return filename
    else: 
        main_df = pd.DataFrame()
        for one_file in all_files:
            df = pickle.load(open(one_file, 'rb'))
            main_df = pd.concat([main_df,df], ignore_index=True)

    compiled_df = open(filename,'wb')
    pickle.dump(main_df,compiled_df)
    compiled_df.close()
    return(filename)

if __name__ == '__main__':
    compile_datasets()