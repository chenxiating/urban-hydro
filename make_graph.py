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
try: 
    folder_name = sys.argv[1]
except IndexError:
    # folder_name = 'datafiles_pool_20210216-1949_969279'
    folder_name = 'datafiles_pool_20210218-0817_993644'
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
nodes_num = int(datafile_name[0:pos3])
# graph_nodes_count_string = datafile_name[pos2:pos3] + '-Nodes Graph with Mean Rainfall Intensity of ' + datafile_name[pos0:pos1] + '\n'

path="/Users/xchen/Documents/UMN_PhD/urban_stormwater_analysis/figures/models/"+ datafile_name.replace(".pickle","/")
if not os.path.exists(path):
    os.makedirs(path)
df = pickle.load(open(datafile_name, 'rb'))
df['soil_nodes_count'] = [(len(k))/nodes_num*100 for k in df.soil_nodes_list]
df.flood_duration_total_list = df.flood_duration_total_list * nodes_num
# print(df.head)

label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Flooded Node-Day',
    'outlet_water_level':'Water Level at Outlet (ft)', 'soil_nodes_count':'% Permeable', 'soil_node_elev_list': "Topography Index", 
    'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Area','mean_rainfall': 'Mean Rainfall Intensity',
    'antecedent_soil':'Antecedent Soil Moisture', 'mean_disp_g':'Mean DG', 'mean_disp_kg': 'Mean DKG', 'max_disp_g':'Max DG', 
    'max_disp_kg':'Max DKG', 'mean_var_path_length':'Time Ensemble <L>', 'max_var_path_length': 'L max'} 

def two_figure_plot(df = df, y1_attribute = 'soil_node_elev_list', y2_attribute = 'soil_node_degree_list',
xaxis_attribute = 'flood_duration_list', cmap_on = False, save_plot = True, cmap = plt.cm.Reds, datafile_name = datafile_name, title = None):
    xaxis = df[xaxis_attribute]
    xmax = max(xaxis) + 1
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    # label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Average Days of Flooding per Node',
    # 'outlet_water_level':'Water Level at Outlet (ft)', 'soil_nodes_count':'% Permeable', 'soil_node_elev_list': "Topography Index", 
    # 'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Area'} 
    if cmap_on: 
        color_dict = {'c': xaxis, 'alpha': 0.7, 'cmap': cmap}
    xlabel = label_dict[xaxis_attribute]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(211)
    ax_NI = fig.add_subplot(212)

    ylim_min = min(min(df[y1_attribute]), min(df[y2_attribute])) 
    ylim_max = max(max(df[y1_attribute]), max(df[y2_attribute])) 

    c = xaxis
    TI = ax_TI.scatter(xaxis,  df[y1_attribute], s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_TI.xaxis.tick_top()
    ax_TI.set_ylabel(label_dict[y1_attribute])
    ax_TI.set_xlim(left = -0.05, right = xmax)
    ax_TI.set_ylim(bottom = ylim_min, top = ylim_max)
    
    NI = ax_NI.scatter(xaxis,  df[y2_attribute], s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_NI.set_ylabel(label_dict[y2_attribute])
    ax_NI.set_xlim(left = -0.05, right = xmax)
    ax_NI.set_ylim(bottom = ylim_min, top = ylim_max)
    plt.tight_layout()    
    if cmap_on: 
        fig.subplots_adjust(top = 0.8, left =0.3)
        cbar_ax = fig.add_axes([0.1, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(TI, cax=cbar_ax)
        # fig.suptitle(graph_nodes_count_string + xlabel)
    if title:
        fig.suptitle(title)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(xlabel)
    plt.tight_layout()

    fig_name = datafile_name.replace(".pickle","") + xaxis_attribute + y1_attribute + title
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def three_figure_plot(df = df, x1_attribute = 'soil_node_elev_list', x2_attribute = 'soil_node_degree_list', x3_attribute = 'soil_nodes_count', 
yaxis_attribute = 'flood_duration_list', cmap_on = False, save_plot = True, cmap = plt.cm.Reds, datafile_name = datafile_name, title = None):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis)
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    # label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Average Days of Flooding per Node',
    # 'outlet_water_level':'Water Level at Outlet (ft)', 'soil_nodes_count':'% Permeable', 'soil_node_elev_list': "Topography Index", 
    # 'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Area'} 
    if cmap_on: 
        color_dict = {'c': yaxis, 'alpha': 0.7, 'cmap': cmap}
    ylabel = label_dict[yaxis_attribute]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(311)
    ax_NI = fig.add_subplot(312)
    ax_nodes = fig.add_subplot(313)

    c = yaxis
    TI = ax_TI.scatter(df[x1_attribute], yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_TI.xaxis.tick_top()
    ax_TI.set_xlabel(label_dict[x1_attribute])
    ax_TI.xaxis.set_label_position('top')
    ax_TI.set_ylim(bottom = -0.05, top = ymax)
    
    NI = ax_NI.scatter(df[x2_attribute], yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_NI.set_xlabel(label_dict[x2_attribute])
    ax_NI.set_ylim(bottom = -0.05, top = ymax)

    nodes_fig = ax_nodes.scatter(df[x3_attribute], yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_nodes.set_xlabel(label_dict[x3_attribute])
    ax_nodes.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax_nodes.set_ylim(bottom = -0.05, top = ymax)

    if cmap_on: 
        fig.subplots_adjust(top = 0.8, left =0.3)
        cbar_ax = fig.add_axes([0.1, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(TI, cax=cbar_ax)
        # fig.suptitle(graph_nodes_count_string + ylabel)
    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    fig_name = datafile_name.replace(".pickle","") + yaxis_attribute
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if title:
        fig.suptitle(title)
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def four_figure_plot(df = df, x1_attribute = 'soil_node_elev_list', x2_attribute = 'soil_node_degree_list', x3_attribute = 'soil_nodes_count', 
x4_attribute ='soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_list', color_attribute = None, cmap_on = False, save_plot = True,
cmap = plt.cm.Reds, datafile_name = datafile_name, title = None):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis)
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    if not color_attribute:
        c = yaxis_attribute
    else: 
        c = df[color_attribute]
    if cmap_on:
        color_dict = {'c': c, 'alpha': 0.4, 'cmap': cmap}
    ylabel = label_dict[yaxis_attribute]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_TI = fig.add_subplot(411)
    ax_NI = fig.add_subplot(412)
    ax_nodes = fig.add_subplot(413)
    ax_area = fig.add_subplot(414)

    c = yaxis
    TI = ax_TI.scatter(df[x1_attribute], yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_TI.xaxis.tick_top()
    ax_TI.set_xlabel(label_dict[x1_attribute])
    ax_TI.xaxis.set_label_position('top')
    ax_TI.set_ylim(bottom = -0.05, top = ymax)
    
    NI = ax_NI.scatter(df[x2_attribute], yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_NI.set_xlabel(label_dict[x2_attribute])
    ax_NI.set_ylim(bottom = -0.05, top = ymax)

    nodes_fig = ax_nodes.scatter(df[x3_attribute], yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_nodes.set_xlabel(label_dict[x3_attribute])
    ax_nodes.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax_nodes.set_ylim(bottom = -0.05, top = ymax)

    area_fig = ax_area.scatter(df[x4_attribute], yaxis,  s = 5, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_area.set_xlabel(label_dict[x4_attribute])
    ax_area.set_ylim(bottom = -0.05, top = ymax)
    plt.tight_layout()

    fig.subplots_adjust(top = 0.8)
    if cmap_on:
        fig.subplots_adjust(left = 0.3)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(TI, cax=cbar_ax)
        cbar_ax.get_yaxis().labelpad = 15
        cbar_ax.set_ylabel(label_dict[color_attribute], rotation=270)
    if title:
        fig.suptitle(title)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel(ylabel)

    fig_name = datafile_name.replace(".pickle","") + yaxis_attribute + title
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_count', yaxis_attribute = 'soil_node_elev_list', color_attribute = 'flood_duration_list', 
s = 5, cmap_on = True, save_plot = True, cmap = plt.cm.Reds, datafile_name = datafile_name, title = None):
    if xaxis_attribute == 'soil_nodes_count':
        xaxis = df[xaxis_attribute]/nodes_num*100
    else: 
        xaxis = df[xaxis_attribute]
    yaxis = df[yaxis_attribute]

    ymax = max(yaxis)    

    fig = plt.figure()
    ax_double_axes = fig.add_subplot(111)  
    
    color_dict = {'alpha': 0.4}
    if cmap_on:
        color_dict['cmap'] = cmap
        color_dict['c'] = df[color_attribute]
    else: 
        color_dict['color'] = plt.cm.Greys(0.5)

    TI = ax_double_axes.scatter(xaxis, yaxis, s = s, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_double_axes.set_xlabel(label_dict[xaxis_attribute])
    ax_double_axes.set_ylabel(label_dict[yaxis_attribute])
    ax_double_axes.set_ylim(bottom = -0.05, top = ymax)
    ax_double_axes.yaxis.tick_right()
    ax_double_axes.yaxis.set_label_position('right')
        # fig.suptitle(graph_nodes_count_string + ylabel)
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    fig.subplots_adjust(top = 0.8)
    plt.tight_layout() 
    if cmap_on:
        fig.subplots_adjust(left = 0.28)
        cbar_ax = fig.add_axes([0.1, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(TI, cax=cbar_ax)
        cbar_ax.get_xaxis().labelpad = 15
        cbar_ax.yaxis.tick_left()
        cbar_ax.xaxis.set_label_position('top')
        cbar_ax.set_xlabel(label_dict[color_attribute])
    if title:
        fig.suptitle(title)
    if save_plot:
        fig_name = datafile_name.replace(".pickle","") + color_attribute + xaxis_attribute + yaxis_attribute + title
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as '+ fig_name +'.png')

def per_attribute_boxplot(x_attr_name = ['antecedent_soil', 'soil_nodes_count'], df = df, yaxis_attribute = 'flood_duration_total_list', 
color = 'C0', datafile_name = datafile_name, title = None):

    ylabel = label_dict[yaxis_attribute]
    if x_attr_name is not list:
        x_attr_name = [x_attr_name]

    # x_attr_name = [x1_attribute, x2_attribute]

    fig, axes = plt.subplots(len(x_attr_name),1)

    for i in range(len(x_attr_name)):
        x_attribute = x_attr_name[i]
        x_attr_list = list(set(df[x_attribute]))
        x_attr_list.sort()
        print(x_attr_list)
        A = []
        label = []
        try: 
            ax_plt = axes[i]
        except TypeError:
            ax_plt = axes
        for x in x_attr_list: 
            is_set0 = (df[x_attribute] == x)
            df_plt0 = df[is_set0]
            A.append(df_plt0[yaxis_attribute])
            # print(A)
            label.append(x)
            # print(label)
        ax_plt.boxplot(A)
        ax_plt.set_xticklabels(label,rotation=45, fontsize=6)
        ax_plt.set_xlabel(label_dict[x_attribute])
        ax_plt.set_ylabel(label_dict[yaxis_attribute])
        ax_plt.xaxis.set_label_position('top') 
    if title:
        fig.suptitle(title)
    else: 
        fig.suptitle('Box Plot')


mean_rainfall_set = set(df.mean_rainfall) - set([0])
mean_soil_set = set(df.antecedent_soil)
# A = []
# label = []
# k = 0
# for i in mean_rainfall_set:
#     for j in mean_soil_set:
#         title_name = 'Rainfall '+ str(i) + ' in; Antecedent soil moisture '+ str(j)
#         print(title_name)
#         is_set = (df.mean_rainfall == i) & (df.antecedent_soil == j)
#         df_plot = df.loc[is_set]
#         B = df_plot.flood_duration_total_list
#         label.append(str(i)+'-'+str(j))
#         A.append(B)

# plt.boxplot(A)
# ax = plt.gca()
# ax.set_xticklabels(label,rotation=45, fontsize=6)
# ax.set_xlabel(label_dict['antecedent_soil'])
# ax.set_ylabel(label_dict['flood_duration_total_list'])

        # three_figure_plot(df = df_plot, yaxis_attribute='flood_duration_total_list', cmap_on = False, save_plot = False)

        # four_figure_plot(df = df_plot, yaxis_attribute='flood_duration_total_list', cmap_on = False, save_plot = False, title = title_name)

        # two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', save_plot = False, title = title_name)
        # two_axis_plot(df = df_plot, color_attribute = 'flood_duration_list', save_plot = False, title = title_name)

        # two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', yaxis_attribute ='soil_node_degree_list', save_plot = False, title = title_name)
        # two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', xaxis_attribute = 'soil_node_degree_list', save_plot = False, title = title_name)
        # plt.show()
    
for i in mean_rainfall_set:
    title_name = 'Rainfall '+ str(i) + ' in'
    print(title_name)
    is_set = (df.mean_rainfall == i)
    df_plot = df.loc[is_set]
    # three_figure_plot(df = df_plot, yaxis_attribute='flood_duration_total_list', cmap_on = False, save_plot = False)

    # four_figure_plot(df = df_plot, yaxis_attribute='flood_duration_total_list', save_plot = True, title = title_name, cmap = plt.cm.Greys)

    # two_axis_plot(df = df_plot, xaxis_attribute='antecedent_soil', color_attribute = 'flood_duration_total_list', save_plot = False, title = title_name)
    # two_axis_plot(df = df_plot, color_attribute = 'flood_duration_list', save_plot = False, title = title_name)

    # two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', yaxis_attribute ='soil_node_degree_list', save_plot = False, title = title_name)
    # two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', xaxis_attribute = 'soil_node_degree_list', save_plot = False, title = title_name)

    # two_axis_plot(df = df_plot, xaxis_attribute = 'soil_nodes_count',yaxis_attribute = 'mean_disp_g', color_attribute = 'flood_duration_total_list', save_plot = True, title = title_name)
    # two_axis_plot(df = df_plot, xaxis_attribute = 'soil_nodes_count',yaxis_attribute = 'max_disp_g', color_attribute = 'flood_duration_total_list', save_plot = True, title = title_name)
    # two_axis_plot(df = df_plot, xaxis_attribute = 'soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_elev_list', save_plot = True)
    # two_axis_plot(df = df_plot, xaxis_attribute = 'soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_degree_list', save_plot = True)

    two_axis_plot(df = df_plot, xaxis_attribute = 'soil_node_degree_list', yaxis_attribute = 'flood_duration_total_list',color_attribute = 'soil_nodes_count', cmap =  plt.cm.Greys, save_plot = True, title = title_name)
    two_axis_plot(df = df_plot, xaxis_attribute = 'soil_node_elev_list', yaxis_attribute = 'flood_duration_total_list',color_attribute = 'soil_nodes_count', cmap =  plt.cm.Greys, save_plot = True, title = title_name)

    # two_figure_plot(df = df_plot, y1_attribute = 'mean_disp_g', y2_attribute = 'mean_disp_kg', xaxis_attribute='soil_nodes_count', cmap_on = False, save_plot = True, title = title_name)
    # two_figure_plot(df = df_plot, y1_attribute = 'max_disp_g', y2_attribute = 'max_disp_kg', xaxis_attribute='soil_nodes_count', cmap_on = False, save_plot = True, title = title_name)

## This is for changing intensity at different permeability 
# for i in [20, 50, 70]:
#     is_set = (df.soil_nodes_count >=i) & (df.soil_nodes_count <=i+10)
#     df_plot = df.loc[is_set]
#     title_name = str(i) + '-' + str(i + 10) + '% Permeable'
#     two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', xaxis_attribute = 'mean_rainfall', s = 10, save_plot = True, title = title_name)
#     two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', xaxis_attribute = 'mean_rainfall', yaxis_attribute = 'mean_disp_g', s = 10, save_plot = True, title = title_name)
#     two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', xaxis_attribute = 'mean_rainfall', yaxis_attribute = 'mean_disp_kg', s = 10, save_plot = True, title = title_name)
#     two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', xaxis_attribute = 'mean_rainfall', yaxis_attribute = 'max_disp_g', s = 10, save_plot = True, title = title_name)
#     two_axis_plot(df = df_plot, color_attribute = 'flood_duration_total_list', xaxis_attribute = 'mean_rainfall', yaxis_attribute = 'max_disp_kg', s = 10, save_plot = True, title = title_name)

plt.show()


# three_figure_plot(df = df, yaxis_attribute='flood_duration_list', cmap_on = False, save_plot = False)
# three_figure_plot(df = df, yaxis_attribute='outlet_water_level',cmap = plt.cm.Greys, cmap_on = False, save_plot = True)

# four_figure_plot(df = df, yaxis_attribute='flood_duration_list', cmap_on = False, save_plot = True)
# four_figure_plot(df = df, yaxis_attribute='outlet_water_level',cmap = plt.cm.Greys, cmap_on = False, save_plot = True)

# two_axis_plot(df = df, color_attribute = 'flood_duration_list', save_plot = True)
# two_axis_plot(df = df, color_attribute = 'flood_duration_list', yaxis_attribute ='soil_node_degree_list', save_plot = True)
# two_axis_plot(df = df, color_attribute = 'flood_duration_list', xaxis_attribute = 'soil_node_degree_list', save_plot = True)

# two_axis_plot(df = df, color_attribute = 'outlet_water_level', cmap =  plt.cm.Greys, save_plot = True)
# two_axis_plot(df = df, color_attribute = 'outlet_water_level', yaxis_attribute ='soil_node_degree_list', cmap =  plt.cm.Greys, save_plot = True)
# two_axis_plot(df = df, color_attribute = 'outlet_water_level', xaxis_attribute = 'soil_node_degree_list', cmap =  plt.cm.Greys, save_plot = True)

# two_axis_plot(df = df, yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_elev_list', save_plot = True)
# two_axis_plot(df = df, yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_degree_list', save_plot = True)
# two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_elev_list', save_plot = True)
# two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_list', color_attribute = 'soil_node_degree_list', save_plot = True)

# two_axis_plot(df = df, xaxis_attribute = 'soil_node_degree_list', yaxis_attribute = 'soil_node_elev_list',color_attribute = 'soil_nodes_count', cmap =  plt.cm.Greys, save_plot = True)