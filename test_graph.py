import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import pickle
import os
import sys

# dataset_4-inch20201223-2214.pickle - this one the degree & elev indices were divided by # of soil nodes
# dataset_4-inch20201223-2238.pickle - divided by total number of nodes
datafile_name = sys.argv[1]
# datafile_name = 'dataset_1-inch_20-nodes_30-day_20210103-1347.pickle'
print("sys.argv is", sys.argv)
print("datafile_name is", datafile_name)
# datafile_name = 'dataset_3.5-inch_20-nodes_20201231-1320'
pos0 = datafile_name.find('dataset_') + len('dataset_')
pos1 = datafile_name.find('-inch') + len('-inch')
pos2 = pos1 + len('_')
pos3 = datafile_name.find('-nodes')
graph_nodes_count_string = datafile_name[pos2:pos3] + '-Nodes Graph with Mean Rainfall Intensity of ' + datafile_name[pos0:pos1] + '\n'

path=("/Users/xchen/Documents/UMN_PhD/urban_stormwater_analysis/figures/models/")
if not os.path.exists(path):
    os.makedirs(path)
#[flood_duration_list, soil_node_degree_list, soil_node_elev_list] = pickle.load(open('dataset_8-inch.pickle', 'rb'))
df = pickle.load(open(datafile_name, 'rb'))
#df = pd.DataFrame([soil_nodes_list, flood_duration_list, soil_node_degree_list, soil_node_elev_list, outlet_max_list]).transpose()
#df.columns =['soil_nodes_list', "flood_duration_list", "soil_node_degree_list", "soil_node_elev_list", "outlet_max_list"]
df['soil_nodes_count'] = [(len(k)) for k in df.soil_nodes_list]
#df['soil_node_degree_list'] = df['soil_node_degree_list']*10/(df['soil_nodes_count']+1)
#df['soil_node_elev_list'] = df['soil_node_elev_list']*10/(df['soil_nodes_count']+1)

def three_figure_plot(df = df, yaxis_attribute = 'flood_duration_list', cmap = plt.cm.Reds, datafile_name = datafile_name):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis) + 1
    if yaxis_attribute == 'flood_duration_list':
        ylabel = 'Days with Flooding'
    elif yaxis_attribute == 'flood_duration_total_list':
        ylabel = 'Average Days of Flooding per Node'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(311)
    ax_NI = fig.add_subplot(312)
    ax_nodes = fig.add_subplot(313)

    c = yaxis
    TI = ax_TI.scatter(df.soil_node_elev_list, yaxis, c = c,# s = s,
    cmap = cmap, alpha = 0.7, edgecolor = 'k')
    ax_TI.xaxis.tick_top()
    ax_TI.set_xlabel("Topography Index")
    ax_TI.xaxis.set_label_position('top')
    ax_TI.set_ylim(bottom = -0.02, top = ymax)
    
    NI = ax_NI.scatter(df.soil_node_degree_list, yaxis, c = c, #s = s, 
    cmap = cmap, alpha = 0.7, edgecolor = 'k')
    ax_NI.set_xlabel("Neighbor Index")
    ax_NI.set_ylim(bottom = -0.02, top = ymax)

    nodes_fig = ax_nodes.scatter(df.soil_nodes_count, yaxis, c = c, #s = s, 
    cmap = cmap, alpha = 0.7, edgecolor = 'k')
    ax_nodes.set_xlabel("Node Count")
    ax_nodes.set_ylim(bottom = -0.02, top = ymax)

    fig.subplots_adjust(top = 0.8, left =0.3)
    cbar_ax = fig.add_axes([0.1, 0.25, 0.05, 0.35])
    cbar = fig.colorbar(TI, cax=cbar_ax)
    fig.suptitle(graph_nodes_count_string + ylabel)
    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel(ylabel)
    fig_name = datafile_name + yaxis_attribute
    plt.savefig(path + fig_name +'.png')

def per_node_count_plot(df = df, soil_nodes_count = 0, yaxis_attribute = 'flood_duration_list', cmap = plt.cm.Reds, datafile_name = datafile_name):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis) + 0.2
    c = yaxis

    if yaxis_attribute == 'flood_duration_list':
        ylabel = 'Days with Flooding'
    elif yaxis_attribute == 'flood_duration_total_list':
        ylabel = 'Average Days of Flooding per Node'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(211)
    ax_NI = fig.add_subplot(212)
    #fig, [ax_TI, ax_NI] = plt.subplots(2,1)
    TI = ax_TI.scatter(df.soil_node_elev_list, yaxis, c = c,# s = s,
    cmap = cmap, alpha = 0.7, edgecolor = 'k')
    ax_TI.xaxis.tick_top()
    ax_TI.set_xlabel("Topography Index")
    ax_TI.xaxis.set_label_position('top') 
    ax_TI.set_ylim(bottom = -0.02, top = ymax)
    
    NI = ax_NI.scatter(df.soil_node_degree_list, yaxis, c = c, #s = s, 
    cmap = cmap, alpha = 0.7, edgecolor = 'k')
    ax_NI.set_xlabel("Neighbor Index")
    ax_NI.set_ylim(bottom = -0.02, top = ymax)
    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel(ylabel)

    fig.subplots_adjust(top = 0.8, left =0.3)
    cbar_ax = fig.add_axes([0.1, 0.25, 0.05, 0.35])
    cbar = fig.colorbar(TI, cax=cbar_ax)

    fig.suptitle(graph_nodes_count_string + ylabel + ' | m = '+ str(soil_nodes_count))
    fig_name = datafile_name + yaxis_attribute + '_m='+ str(soil_nodes_count)
    plt.savefig(path + fig_name +'.png')

three_figure_plot(df = df, yaxis_attribute='flood_duration_list')
three_figure_plot(df = df, yaxis_attribute='flood_duration_total_list',cmap = plt.cm.Blues)

soil_nodes_count_set = set(df.soil_nodes_count)
for k in soil_nodes_count_set:
    is_set = df['soil_nodes_count'] == k
    df_plt = df[is_set]
    per_node_count_plot(df = df_plt, soil_nodes_count = k, yaxis_attribute='flood_duration_list')
    per_node_count_plot(df = df_plt, soil_nodes_count = k, yaxis_attribute='flood_duration_total_list',cmap = plt.cm.Blues)


#ax_NI.set_ylim(bottom = -0.1, top = 1.6)
# kw = dict(prop="sizes", num=4, color=NI.cmap(0.5), fmt="{x:.0f}",
#           func=lambda s: np.log(s))
# legend = ax_NI.legend(*NI.legend_elements(**kw),
#                     bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="# of Nodes")
# legend.get_frame().set_edgecolor('k')

# fig, axes = plt.subplots(2,2)
# axes = axes.ravel()

# nodes_num = [1, 5, 10, 15]
# for k in range(4):
#     i = nodes_num[k]
#     is_set = df['soil_nodes_count'] == i
#     plt_set = df[is_set]
#     #print(plt_set.head())
#     #s = [10**i for i in plt_set['flood_duration_list']]
#     axes[k].scatter(plt_set['soil_node_degree_list'], plt_set['soil_node_elev_list'], c = 'red', #s = s,
#     alpha = 0.7)#, edgecolor = 'k')
#     #fig_dots.subplots_adjust(right=0.8)
#     #cbar_ax = fig_dots.add_axes([0.85, 0.5, 0.05, 0.35])
#     #cbar = fig.colorbar(dots)
#     #cbar.set_label('Flood Duration (Days)')
#     axes[k].set_ylabel("Topography Index " + str(i))
#     axes[k].set_xlabel("Neighbor Index " + str(i))
#     #ax_dots.set_ylim(bottom = -0.1, top = 0.5)
#     # kw = dict(prop="sizes", num=4, color=dots.cmap(0.5), fmt="{x:.0f}",
#     #         func=lambda s: np.log(s))
#     # legend = cbar_ax.legend(*dots.legend_elements(**kw),
#     #                     bbox_to_anchor=(1, -0.1), loc='upper center', borderaxespad=0., title="# of Nodes")
#     # legend.get_frame().set_edgecolor('k')

# fig, axes = plt.subplots(2,2)
# axes = axes.ravel()

# for k in range(4):
#     i = k+1
#     is_set = df['soil_nodes_count'] == i
#     plt_set = df[is_set]
#     #print(plt_set.head())
#     s = [10**i for i in plt_set['outlet_max_list']]
#     axes[k].scatter(plt_set['soil_node_degree_list'], plt_set['soil_node_elev_list'], s = s,
#     alpha = 0.7)#, edgecolor = 'k')
#     #fig_dots.subplots_adjust(right=0.8)
#     #cbar_ax = fig_dots.add_axes([0.85, 0.5, 0.05, 0.35])
#     #cbar = fig.colorbar(dots)
#     #cbar.set_label('Flood Duration (Days)')
#     axes[k].set_ylabel("TI " + str(i) + " Nodes")
#     axes[k].set_xlabel("NI " + str(i)+ " Nodes")
#     #ax_dots.set_ylim(bottom = -0.1, top = 0.5)
#     # kw = dict(prop="sizes", num=4, color=dots.cmap(0.5), fmt="{x:.0f}",
#     #         func=lambda s: np.log(s))
#     # legend = cbar_ax.legend(*dots.legend_elements(**kw),
#     #                     bbox_to_anchor=(1, -0.1), loc='upper center', borderaxespad=0., title="# of Nodes")
#     # legend.get_frame().set_edgecolor('k')
plt.show()