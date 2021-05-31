import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
import matplotlib
import pandas as pd
import seaborn as sns
import pickle
import os
import sys
from statistics import StatisticsError
from statistics import mean
from matplotlib.colors import PowerNorm
from matplotlib.colors import Normalize
from compile_datasets import compile_datasets
from matplotlib.patches import ConnectionPatch

matplotlib.rcParams['figure.figsize'] = (12, 6)
# try: 
#     folder_name = sys.argv[1]
# except IndexError:
#     # folder_name = 'datafiles_pool_20210326-2212_1849651'
#     folder_name = 'datafiles_pool_20210303-2012_1242945'
# datafile_directory='/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/datasets-compiled'
# os.chdir(datafile_directory)

datafile_name = r'./SWMM_20210529-0852/20210529-0852_full_dataset_100-nodes.pickle'
# datafile_name = '100-nodes_10-day_20210303-2012.pickle'
print("sys.argv is", sys.argv)
print("datafile_name is", datafile_name)
pos0 = datafile_name.find('dataset_') + len('dataset_')
pos1 = datafile_name.find('-inch') + len('-inch')
pos2 = pos1 + len('_')
pos3 = datafile_name.find('-nodes')
nodes_num = int(datafile_name[pos0:pos3])
# nodes_num = 100
# graph_nodes_count_string = datafile_name[pos2:pos3] + '-Nodes Graph with Mean Rainfall Intensity of ' + datafile_name[pos0:pos1] + '\n'

df = pickle.load(open(datafile_name, 'rb'))
df.flood_duration_total_list = df.flood_duration_total_list/24

datafile_name = datafile_name[2:19]
path="/Volumes/GoogleDrive/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
print(path)
if not os.path.exists(path):
    os.makedirs(path)
# df = df[df['soil_nodes_count']<=50]
# df['soil_nodes_average_upstream_area'] = df.soil_nodes_total_upstream_area/df.soil_nodes_count

# print(df.head)

label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Flooded Node-Day',
    'max_flow_cfs':'Max Flow Rate at Outlet (cfs)', 'soil_nodes_count':'% of Permeable Nodes', 'soil_node_elev_list': "Topography Index", 
    'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Upstream Area','mean_rainfall': 'Mean Rainfall Intensity',
    'antecedent_soil':'Antecedent Soil Moisture', 'mean_disp_g':'Mean DG', 'mean_disp_kg': 'Mean DKG', 'max_disp_g':'Max DG', 
    'max_disp_kg':'Max DKG', 'mean_var_path_length':'Time Averaged <L>', 'max_var_path_length': 'L max', 'max_flood_nodes':'Highest Number of Flooded Nodes',
    'max_flood_node_degree':'Average Connectedness when Highest Number of Nodes Flooded','max_flood_node_elev':'Average Distance to Outlet when Highest Number of Nodes Flooded',
    'mean_flood_nodes_TI':'Average Distance to Outlet of Flood Node (at All Times)','total_flooded_vol_MG':'Total Flood Volume (MG)','total_outflow_vol_MG':'Total Outflow Volume (MG)'} 

def two_figure_plot(df = df, y1_attribute = 'soil_node_elev_list', y2_attribute = 'soil_node_degree_list',
xaxis_attribute = 'flood_duration_total_list', cmap_on = False, save_plot = True, cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
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

    fig_name = datafile_name.replace(".pickle","") + xaxis_attribute + y1_attribute
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def three_figure_plot(df = df, x1_attribute = 'soil_node_elev_list', x2_attribute = 'soil_node_degree_list', x3_attribute = 'soil_nodes_count', 
yaxis_attribute = 'flood_duration_total_list', cmap_on = False, save_plot = True, cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
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
x4_attribute ='soil_nodes_total_upstream_area', yaxis_attribute = 'flood_duration_total_list', color_attribute = None, cmap_on = False, save_plot = True,
cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
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

    fig_name = datafile_name.replace(".pickle","") + yaxis_attribute + str(title)
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_count', yaxis_attribute = 'soil_node_elev_list', color_attribute = 'flood_duration_total_list', 
size_attribute = None, cmap_on = True, save_plot = True, cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
    if xaxis_attribute == 'soil_nodes_count':
        xaxis = df[xaxis_attribute]/nodes_num*100
    else: 
        xaxis = df[xaxis_attribute]
    yaxis = df[yaxis_attribute]

    ymax = max(yaxis)

    fig = plt.figure()
    ax_double_axes = fig.add_subplot(111)  
    
    color_dict = {'alpha': 0.8}
    if cmap_on:
        color_dict['cmap'] = cmap
        color_dict['c'] = df[color_attribute]
        if cmap != 'plt.cm.viridis':
            color_dict['edgecolor']='dimgray'
            color_dict['linewidths']=0.5
    else: 
        color_dict['color'] = plt.cm.Greys(0.5)

    if not size_attribute:
        s = 10
    else: 
        s = df[size_attribute]
    TI = ax_double_axes.scatter(xaxis, yaxis, s = s, marker = 's', **color_dict)#, edgecolor = 'k')
    ax_double_axes.set_xlabel(label_dict[xaxis_attribute])
    ax_double_axes.set_ylabel(label_dict[yaxis_attribute])
    ax_double_axes.set_ylim(bottom = -0.05, top = ymax)
    ax_double_axes.yaxis.tick_right()
    ax_double_axes.yaxis.set_label_position('right')
        # fig.suptitle(graph_nodes_count_string + ylabel)
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')

    plt.tight_layout() 

    fig.subplots_adjust(top = 0.8)
    if title:
        fig.suptitle(title)
    if cmap_on:
        fig.subplots_adjust(left = 0.3)
        cbar_ax = fig.add_axes([0.1, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(TI, cax=cbar_ax)
        cbar_ax.get_xaxis().labelpad = 15
        cbar_ax.yaxis.tick_left()
        cbar_ax.xaxis.set_label_position('top')
        cbar_ax.set_xlabel(label_dict[color_attribute])
    if save_plot:
        fig_name = datafile_name.replace(".pickle","") + color_attribute + xaxis_attribute + yaxis_attribute + str(title)
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

def soil_box_plot(save_plot=False):
    mean_rainfall_set = list(set(df.mean_rainfall) - set([0]))
    mean_rainfall_set.sort()
    mean_soil_set = list(set(df.antecedent_soil))
    mean_soil_set
    A = []
    label = []
    k = 0
    for i in mean_rainfall_set:
        for j in mean_soil_set:
            title_name = 'Rainfall '+ str(i) + ' in; Antecedent soil moisture '+ str(j)
            # print(title_name)
            is_set = (df.mean_rainfall == i) & (df.antecedent_soil == j)
            df_plot = df.loc[is_set]
            B = df_plot.flood_duration_total_list
            label.append(str(round(i,1))+'" -'+str(round(j*100,0))+'%')
            A.append(B)
    plt.figure()
    ax = plt.gca()
    ax.boxplot(A)
    ax.set_xticklabels(label,rotation=45, fontsize=6)
    ax.set_xlabel(label_dict['antecedent_soil'])
    ax.set_ylabel(label_dict['flood_duration_total_list'])
    fig_name = datafile_name.replace(".pickle","") + 'soil_boxplot'
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')


def multi_rainfall_figure_plot(df = df, ncols = 5, nrows = 1, xaxis_attribute = 'flood_duration_total_list', yaxis_attribute = 'soil_node_elev_list', color_attribute = 'soil_nodes_count', cmap_on = False, kde = False,save_plot = True,fitline=False,
cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
    # print(df)
    xaxis = df[xaxis_attribute]
    xmax = max(xaxis)
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis)
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    if not color_attribute:
        c = yaxis_attribute
    else: 
        c = round(df[color_attribute], -1)
    if cmap_on:
        color_dict = {'c': c, 'alpha': 0.6, 'cmap': cmap}
    ylabel = label_dict[yaxis_attribute]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=True)
    axes[0].set_ylabel(ylabel)
    axes[0].get_yaxis().labelpad = 10
    fig.suptitle(label_dict[xaxis_attribute])
    mean_rainfall_set = list(set(df.mean_rainfall) - set([0]))
    mean_rainfall_set.sort()
    k = 0
    vmin = min(df[color_attribute])
    vmax = max(df[color_attribute])
    color_iteration = np.linspace(vmin, vmax, 5, endpoint=False)
    df_centroid = pd.DataFrame()
    for i in mean_rainfall_set:
        x_label = str(round(i,1)) + '"'
        axes[k].grid(which = 'both', axis = 'both', alpha = 0.2)
        if kde: 
            ax_hold = axes[k].twiny()
        axes[k].set_xlabel(x_label)
        axes[k].xaxis.tick_top()
        plt.setp(axes[k].spines.values(), alpha = 0.2)
        axes[k].tick_params(labelcolor='w',top=False, bottom=False, left=False, right=False)
        axes[k].set_ylim([0, ymax])
        axes[k].set_xlim([0, xmax])
        x_centroid_list = []
        y_centroid_list = []
        centroid_color = []
        
        for j in color_iteration:
            line_color = plt.cm.viridis(j/(vmax-vmin))
            is_set = (df.mean_rainfall == i) & ((df[color_attribute] >= j) & (df[color_attribute] < j+color_iteration[1]-color_iteration[0]))
            # is_set = (df.antecedent_soil == i) & ((df[color_attribute] >= j) & (df[color_attribute] < j+color_iteration[1]-color_iteration[0]))
            df_plot = df.loc[is_set]

            c = round(df_plot[color_attribute], -1)
            if cmap_on:
                color_dict = {'c': c, 'alpha': 0.2}
            
            if kde:
                sns.kdeplot(y=df_plot[yaxis_attribute],color=line_color,ax=ax_hold,thresh=0.8)
            else:
                axes[k].scatter(df_plot[xaxis_attribute], df_plot[yaxis_attribute], s = 5, marker = 's', vmin = vmin, vmax = vmax, **color_dict)
                try:
                    x_centroid_list.append(mean(df_plot[xaxis_attribute][df_plot[xaxis_attribute]>0]))
                except StatisticsError:
                    x_centroid_list.append(0)
                try: 
                    y_centroid_list.append(mean(df_plot[yaxis_attribute][df_plot[yaxis_attribute]>0]))
                except StatisticsError:
                    y_centroid_list.append(0)
                centroid_color.append(j)

            if fitline:
                try:
                    m,b = np.polyfit(df_plot[xaxis_attribute], df_plot[yaxis_attribute],1)
                    axes[k].plot(df_plot[xaxis_attribute], m*df_plot[xaxis_attribute]+b,color=line_color,alpha=0.5)
                except TypeError:
                    pass

        scatter_fig = axes[k].scatter(x_centroid_list, y_centroid_list, c = centroid_color, s = 500/ncols, marker = 'o',edgecolor = 'w', vmin = vmin, vmax = vmax)
        # df_this_round = pd.DataFrame(list(zip(np.char.asarray(i,itemsize=len(color_iteration)), np.char.asarray(x_label,itemsize=len(color_iteration)), color_iteration, x_centroid_list, y_centroid_list, centroid_color),columns=['Ticks','Label','Permeability','x_coord','y_coord','color']))
        # df_centroid = pd.concat([df_centroid, df_this_round],ignore_index=True)
        k += 1
    axes[0].tick_params(labelcolor ='k')
        
    plt.tight_layout(w_pad= 0.0)
    if cmap_on:
        fig.subplots_adjust(left = 0.1, right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(scatter_fig, cax=cbar_ax)
        cbar_ax.get_yaxis().labelpad = 15
        cbar_ax.set_ylabel(label_dict[color_attribute], rotation=270)
        # cbar_ax.yaxis.set_label_position('left')
        # cbar_ax.yaxis.tick_left()
    if title:
        fig.suptitle(title)

    fig_name = datafile_name.replace(".pickle","") + xaxis_attribute + yaxis_attribute
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')
    
    # if separate_plot:
        # fig, ax = plt.subplots(1,1,1)
        # ax.plt(x_centroid_list,y_centroid_list)
        # print(df_centroid)

def multi_rainfall_histogram(df = df, ncols = 5, nrows = 1, yaxis_attribute = 'soil_node_elev_list', color_attribute = 'soil_nodes_count', cmap_on = False, save_plot = True,
cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
    # print(df)
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis)
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    if not color_attribute:
        c = yaxis_attribute
    else: 
        c = round(df[color_attribute], -1)
    if cmap_on:
        color_dict = {'c': c, 'alpha': 0.6, 'cmap': cmap}
    ylabel = label_dict[yaxis_attribute]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
    fig.suptitle('Kernel Density Estimates at Different Rainfall Intensities')

    axes[0].set_ylabel(ylabel)
    axes[0].get_yaxis().labelpad = 10
    mean_rainfall_set = list(set(df.mean_rainfall) - set([0]))
    mean_rainfall_set.sort()
    k = 0
    vmin = min(df[color_attribute])
    vmax = max(df[color_attribute])
    color_iteration = np.linspace(vmin, vmax, 5, endpoint=False)
    for i in mean_rainfall_set:
        x_label = str(round(i,1)) + '"'
        axes[k].set_xlabel(x_label)
        axes[k].grid(which = 'both', axis = 'both', alpha = 0.2)
        axes[k].xaxis.tick_top()
        plt.setp(axes[k].spines.values(), alpha = 0.2)
        axes[k].tick_params(labelcolor='w',top=False, bottom=False, left=False, right=False)
        axes[k].set_ylim([0, ymax])
        line_color = plt.cm.viridis(color_iteration/100)
        
        for j in color_iteration:
            is_set = (df.mean_rainfall == i) & ((df[color_attribute] >= j) & (df[color_attribute] < j+color_iteration[1]-color_iteration[0]))
            df_plot = df.loc[is_set]

            # c = plt.cm.viridis(j/(max(color_iteration)-min(color_iteration)))
            c = plt.cm.viridis(j/(vmax-vmin))
            if cmap_on:
                color_dict = {'histtype':'step', 'color': c,'density':True,'bins':5}
            # print('var',np.var(df_plot[yaxis_attribute]))
            # print(df_plot[yaxis_attribute])
            if np.var(df_plot[yaxis_attribute]) > 0:
                # axes[k].hist(df_plot[yaxis_attribute],**color_dict)
                sns.kdeplot(y=df_plot[yaxis_attribute],color=c,ax=axes[k])
        #     try:
        #         x_centroid_list.append(mean(df_plot[xaxis_attribute][df_plot[xaxis_attribute]>0]))
        #     except StatisticsError:
        #         x_centroid_list.append(0)
        #     try: 
        #         y_centroid_list.append(mean(df_plot[yaxis_attribute][df_plot[yaxis_attribute]>0]))
        #     except StatisticsError:
        #         y_centroid_list.append(0)
        #     centroid_color.append(j)


        # axes[k].scatter(x_centroid_list, y_centroid_list, c = centroid_color, s = 500/ncols, marker = 'o',edgecolor = 'w', vmin = vmin, vmax = vmax)
        k += 1
    axes[0].tick_params(labelcolor ='k')
        
    plt.tight_layout(w_pad= 0.0)
    if cmap_on:
        fig.subplots_adjust(left = 0.1, right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.35])
        norm=Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cbar_ax)
        cbar_ax.get_yaxis().labelpad = 15
        cbar_ax.set_ylabel(label_dict[color_attribute], rotation=270)
        # cbar_ax.yaxis.set_label_position('left')
        # cbar_ax.yaxis.tick_left()
    if title:
        fig.suptitle(title)

    fig_name = datafile_name.replace(".pickle","") + '_KDE_'+yaxis_attribute
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')
if __name__ == '__main__':
    # for y in ['flood_duration_total_list','max_flood_nodes','total_flooded_vol_MG','max_flow_cfs','total_outflow_vol_MG']:#, 'mean_var_path_length', 'mean_disp_kg', 'mean_disp_g']:
    #     for x in ['soil_node_degree_list','soil_node_elev_list']:#,'mean_flood_nodes_TI']:
            # multi_rainfall_figure_plot(df, ncols = 10, nrows = 1, xaxis_attribute = x, yaxis_attribute = y, cmap_on = True, save_plot=False)
            # multi_rainfall_histogram(df, ncols = 10, nrows = 1,yaxis_attribute = y,color_attribute='soil_nodes_count',cmap_on = True, save_plot=False)
    # two_axis_plot(xaxis_attribute='soil_node_degree_list',yaxis_attribute='flood_duration_total_list',save_plot=False)
    soil_box_plot()
    plt.show()