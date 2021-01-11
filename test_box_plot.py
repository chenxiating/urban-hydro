import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import pickle


# Fixing random state for reproducibility
np.random.seed(19680801)
datafile_name = 'dataset_1-inch_20-nodes_30-day_20210103-1347.pickle'
df = pickle.load(open(datafile_name, 'rb'))

df['soil_nodes_count'] = [(len(k)) for k in df.soil_nodes_list]

def per_node_count_boxplot(df = df, soil_nodes_count = 0, yaxis_attribute = 'flood_duration_list', color = 'red', datafile_name = datafile_name):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis) + 0.1
    c = yaxis

    if yaxis_attribute == 'flood_duration_list':
        ylabel = 'Days with Flooding'
    elif yaxis_attribute == 'flood_duration_total_list':
        ylabel = 'Average Days of Flooding per Node'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(211)
    ax_NI = fig.add_subplot(212)

    label_TI = []
    soil_node_elev_list = list(set(df.soil_node_elev_list))
    soil_node_elev_list.sort()
    A = [None] * len(soil_node_elev_list)
    k = 0
    for x in soil_node_elev_list: 
        is_set0 = df.soil_node_elev_list == x
        df_plt0 = np.array(df[is_set0][yaxis_attribute])
        A[k] = df_plt0
        label_TI.append(x)
        k +=1

    bp_TI = ax_TI.boxplot(A)
    ax_TI.set_xticklabels(label_TI,rotation=45, fontsize=6)
    ax_TI.set_xlabel("Topography Index")
    ax_TI.xaxis.set_label_position('top') 
    plt.setp(bp_TI['whiskers'], color='k', linestyle='-')
    plt.setp(bp_TI['fliers'], markersize=3.0)
    plt.setp(bp_TI['medians'], color = color)

    label_NI = []
    soil_node_degree_list = list(set(df.soil_node_degree_list))
    soil_node_degree_list.sort()
    A = [None] * len(soil_node_degree_list)
    k = 0
    for x in soil_node_degree_list: 
        is_set0 = df.soil_node_degree_list == x
        df_plt0 = np.array(df[is_set0][yaxis_attribute])
        #df_plt.shape = (-1, 1)
        A[k] = df_plt0
        label_NI.append(x)
        k +=1

    bp_NI = ax_NI.boxplot(A)
    ax_NI.set_xticklabels(label_NI,rotation=45, fontsize=6)
    ax_NI.set_xlabel("Neighbor Index")
    plt.setp(bp_NI['whiskers'], color='k', linestyle='-')
    plt.setp(bp_NI['fliers'], markersize=3.0)
    plt.setp(bp_NI['medians'], color = color)
    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel(ylabel)

#### 
soil_nodes_count_set = set(df.soil_nodes_count)
for k in soil_nodes_count_set:
    is_set = df['soil_nodes_count'] == k
    df_plt = df[is_set]
    per_node_count_boxplot(df = df_plt, soil_nodes_count = k, yaxis_attribute='flood_duration_list', color = 'red')
plt.show()