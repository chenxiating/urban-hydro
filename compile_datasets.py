import os
import pandas as pd
import pickle

folder_name = 'datafiles_20210124-2339'
os.chdir('/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/'+folder_name)
all_files = os.listdir()
main_df = pd.DataFrame()

for one_file in all_files:
    df = pickle.load(open(one_file, 'rb'))
    main_df = pd.concat([main_df,df], ignore_index=True)

datafile_directory='/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/datasets_compiled'
if not os.path.exists(datafile_directory):
    os.makedirs(datafile_directory)
os.chdir(datafile_directory)

datafile_name = all_files[0]
pos0 = datafile_name.find('network_count')
filename = datafile_name[0:pos0]+'.pickle'
print(filename)
compiled_df = open(filename,'wb')
pickle.dump(main_df,compiled_df)
compiled_df.close()