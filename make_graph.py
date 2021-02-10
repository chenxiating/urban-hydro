import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
import pandas as pd
import pickle
import os
import sys
from compile_datasets import compile_datasets

# datafile_name = sys.argv[1]
folder_name = sys.argv[1]
datafile_directory='/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro_datasets-compiled'
os.chdir(datafile_directory)

datafile_name = compile_datasets(folder_name)
print("sys.argv is", sys.argv)
print("datafile_name is", datafile_name)
# datafile_name = 'dataset_3.5-inch_20-nodes_20201231-1320'
pos0 = datafile_name.find('dataset_') + len('dataset_')
pos1 = datafile_name.find('-inch') + len('-inch')
pos2 = pos1 + len('_')
pos3 = datafile_name.find('-nodes')
nodes_num = int(datafile_name[pos2:pos3])
graph_nodes_count_string = datafile_name[pos2:pos3] + '-Nodes Graph with Mean Rainfall Intensity of ' + datafile_name[pos0:pos1] + '\n'

path=("/Users/xchen/Documents/UMN_PhD/urban_stormwater_analysis/figures/models/")
if not os.path.exists(path):
    os.makedirs(path)
df = pickle.load(open(datafile_name, 'rb'))
df['soil_nodes_count'] = [(len(k)) for k in df.soil_nodes_list]


def three_figure_plot(df = df, yaxis_attribute = 'flood_duration_list', cmap_on = False, save_plot = True,
cmap = plt.cm.Reds, datafile_name = datafile_name):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis) + 1
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Average Days of Flooding per Node',
    'outlet_water_level':'Water Level at Outlet (ft)', 'soil_nodes_count':'% Permeable', 'soil_node_elev_list': "Topography Index", 
    'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Area'} 
    if cmap_on: 
        color_dict = {'c': yaxis, 'alpha': 0.7, 'cmap': cmap}
    ylabel = label_dict[yaxis_attribute]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(311)
    ax_NI = fig.add_subplot(312)
    ax_nodes = fig.add_subplot(313)

    c = yaxis
    TI = ax_TI.scatter(df.soil_node_elev_list, yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_TI.xaxis.tick_top()
    ax_TI.set_xlabel("Topography Index")
    ax_TI.xaxis.set_label_position('top')
    ax_TI.set_ylim(bottom = -0.05, top = ymax)
    
    NI = ax_NI.scatter(df.soil_node_degree_list, yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_NI.set_xlabel("Neighbor Index")
    ax_NI.set_ylim(bottom = -0.05, top = ymax)

    nodes_fig = ax_nodes.scatter(df.soil_nodes_count/nodes_num*100, yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_nodes.set_xlabel("% Permeable")
    ax_nodes.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax_nodes.set_ylim(bottom = -0.05, top = ymax)

    if cmap_on: 
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
    plt.tight_layout()

    fig_name = datafile_name.replace(".pickle","") + yaxis_attribute
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def four_figure_plot(df = df, yaxis_attribute = 'flood_duration_list', cmap_on = False, save_plot = True,
cmap = plt.cm.Reds, datafile_name = datafile_name):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis) + 1
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Average Days of Flooding per Node',
    'outlet_water_level':'Water Level at Outlet (ft)', 'soil_nodes_count':'% Permeable', 'soil_node_elev_list': "Topography Index", 
    'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Area'} 
    if cmap_on: 
        color_dict = {'c': yaxis, 'alpha': 0.7, 'cmap': cmap}
    ylabel = label_dict[yaxis_attribute]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(411)
    ax_NI = fig.add_subplot(412)
    ax_nodes = fig.add_subplot(413)
    ax_area = fig.add_subplot(414)

    c = yaxis
    TI = ax_TI.scatter(df.soil_node_elev_list, yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_TI.xaxis.tick_top()
    ax_TI.set_xlabel("Topography Index")
    ax_TI.xaxis.set_label_position('top')
    ax_TI.set_ylim(bottom = -0.05, top = ymax)
    
    NI = ax_NI.scatter(df.soil_node_degree_list, yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_NI.set_xlabel("Neighbor Index")
    ax_NI.set_ylim(bottom = -0.05, top = ymax)

    nodes_fig = ax_nodes.scatter(df.soil_nodes_count/nodes_num*100, yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_nodes.set_xlabel("% Permeable")
    ax_nodes.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax_nodes.set_ylim(bottom = -0.05, top = ymax)

    area_fig = ax_area.scatter(df.soil_nodes_total_upstream_area, yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_area.set_xlabel(label_dict['soil_nodes_total_upstream_area'])
    ax_area.set_ylim(bottom = -0.05, top = ymax)

    if cmap_on: 
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
    plt.tight_layout()

    fig_name = datafile_name.replace(".pickle","") + yaxis_attribute
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_count', yaxis_attribute = 'soil_node_elev_list', color_attribute = 'flood_duration_list', 
cmap_on = True, save_plot = True, cmap = plt.cm.Reds, datafile_name = datafile_name):
    if xaxis_attribute == 'soil_nodes_count':
        xaxis = df[xaxis_attribute]/nodes_num*100
    else: 
        xaxis = df[xaxis_attribute]
    yaxis = df[yaxis_attribute]

    ymax = max(yaxis)
    label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Average Days of Flooding per Node',
    'outlet_water_level':'Water Level at Outlet (ft)', 'soil_nodes_count':'% Permeable', 'soil_node_elev_list': "Topography Index", 
    'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Area'} 
    color_dict = {'c': df[color_attribute], 'alpha': 0.4, 'cmap': cmap}

    fig = plt.figure()
    ax_double_axes = fig.add_subplot(111)

    TI = ax_double_axes.scatter(xaxis, yaxis, s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_double_axes.set_xlabel(label_dict[xaxis_attribute])
    ax_double_axes.set_ylabel(label_dict[yaxis_attribute])
    ax_double_axes.set_ylim(bottom = -0.05, top = ymax)
    ax_double_axes.yaxis.tick_right()
    ax_double_axes.yaxis.set_label_position('right')

    if cmap_on: 
        fig.subplots_adjust(top = 0.8, left = 0.28)
        cbar_ax = fig.add_axes([0.1, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(TI, cax=cbar_ax)
        cbar_ax.get_xaxis().labelpad = 15
        cbar_ax.yaxis.tick_left()
        cbar_ax.xaxis.set_label_position('top')
        cbar_ax.set_xlabel(label_dict[color_attribute])
        # fig.suptitle(graph_nodes_count_string + ylabel)
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')

    if save_plot:
        fig_name = datafile_name.replace(".pickle","") + color_attribute + xaxis_attribute + yaxis_attribute
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as '+ fig_name +'.png')


three_figure_plot(df = df, yaxis_attribute='flood_duration_list', cmap_on = False, save_plot = True)
three_figure_plot(df = df, yaxis_attribute='outlet_water_level',cmap = plt.cm.Greys, cmap_on = False, save_plot = True)

four_figure_plot(df = df, yaxis_attribute='flood_duration_list', cmap_on = False, save_plot = True)
four_figure_plot(df = df, yaxis_attribute='outlet_water_level',cmap = plt.cm.Greys, cmap_on = False, save_plot = True)

two_axis_plot(df = df, color_attribute = 'flood_duration_list', save_plot = True)
two_axis_plot(df = df, color_attribute = 'flood_duration_list', yaxis_attribute ='soil_node_degree_list', save_plot = True)
two_axis_plot(df = df, color_attribute = 'flood_duration_list', xaxis_attribute = 'soil_node_degree_list', save_plot = True)

two_axis_plot(df = df, color_attribute = 'outlet_water_level', cmap =  plt.cm.Greys, save_plot = True)
two_axis_plot(df = df, color_attribute = 'outlet_water_level', yaxis_attribute ='soil_node_degree_list', cmap =  plt.cm.Greys, save_plot = True)
two_axis_plot(df = df, color_attribute = 'outlet_water_level', xaxis_attribute = 'soil_node_degree_list', cmap =  plt.cm.Greys, save_plot = True)

two_axis_plot(df = df, yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_elev_list', save_plot = True)
two_axis_plot(df = df, yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_degree_list', save_plot = True)
two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_elev_list', save_plot = True)
two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_degree_list', save_plot = True)

two_axis_plot(df = df, xaxis_attribute = 'soil_node_degree_list', yaxis_attribute = 'soil_node_elev_list',color_attribute = 'soil_nodes_count', cmap =  plt.cm.Greys, save_plot = True)


plt.show()