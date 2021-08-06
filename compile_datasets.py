import os
import glob
from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import pickle
import datetime as date
import datetime
import matplotlib.pyplot as plt
import networkx as nx
from math import floor
from networkx.drawing.nx_agraph import graphviz_layout

def compile_datasets(folder_name, overwrite=False):
    os.chdir(folder_name)
     
    main_df = pd.DataFrame(columns=['matrix','count'])
    k = 0
    for one_file in glob.glob('*.pickle'):
        df = pickle.load(open(one_file, 'rb'))
        if k == 0:
            check = [False]
        else: 
            check = [np.all(df == k) for k in (main_df.matrix)]
        if any(check):
            index = check.index(True)
            main_df.loc[index,'count'] = main_df.loc[index,'count'] + 1
        else: 
            main_df.loc[k, 'matrix'] = df#.tolist()
            main_df.loc[k,'count'] = 1
            k = k + 1
            name = one_file.replace('.pickle','.png')
            # draw_tree(title= f'Matrix {k}', matrix=df, name=name)
    return main_df

def convert_index(k, n):
    # convert matrix index to (i,j)  coordinates
    return (floor(k/n), k%n)

def draw_tree(matrix, name, title=None):
        _ = plt.figure(figsize=(10,4.5))
        G = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph)
        plt.subplot(121)
        pos_grid = {k: convert_index(k, n) for k in G.nodes()}
        nx.draw(G, pos=pos_grid, node_size=2, edge_color='grey') #, with_labels = True)
        plt.subplot(122)
        pos_gv = graphviz_layout(G, prog = 'dot')
        nx.draw(G, pos=pos_gv, node_size=2, edge_color='lightgrey') #, with_labels=True)
        plt.title(title)
        plt.savefig(f'./{name}')
        plt.close()

if __name__ == '__main__':
    subfolder = 'gibbs4_20210803-1821'
    pos0 = subfolder.find('s')+1
    pos1 = subfolder.find('_')
    n = int(subfolder[pos0:pos1])
    folder_name='/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/'+ subfolder
    main_df = compile_datasets(folder_name = folder_name)

    norm_coef = 0.004273514057548294

#%%
    ax = main_df.plot.bar(y='count')
    yticks = np.linspace(0,max(main_df['count']),num = max(main_df['count']) + 1,dtype=int)
    print(yticks)
    ax.set_yticks(yticks)
    ax.set_xlabel('Index for each unique matrix')
    ax.set_ylabel('Count of matrices')
    plt.show()