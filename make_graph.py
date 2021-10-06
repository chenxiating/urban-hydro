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

matplotlib.rcParams['figure.figsize'] = (6, 4)

# datafile_name = r'./SWMM_20210830-1058/20210830-1058_full_dataset_100-nodes.pickle' # This is only small rainfall, 10 + 20 % permeability 
# datafile_name = r'./SWMM_20210831-0907/20210831-0907_full_dataset_100-nodes.pickle' # This is only heavy rainfall, 10 + 20 % permeability 
datafile_name = r'./SWMM_20211002-1437/20211002-1437_full_dataset_100-nodes.pickle'
datafile_name = r'./SWMM_20210825-1653/20210825-1653_full_dataset_100-nodes.pickle'
# datafile_name = r'./SWMM_20211003-1553/20211003-1553_full_dataset_100-nodes.pickle'
print("sys.argv is", sys.argv)
print("datafile_name is", datafile_name)
nodes_num = 100
# graph_nodes_count_string = datafile_name[pos2:pos3] + '-Nodes Graph with Mean Rainfall Intensity of ' + datafile_name[pos0:pos1] + '\n'

df = pickle.load(open(datafile_name, 'rb'))
df.to_csv(datafile_name.replace('pickle','csv'))
# df = pd.read_csv(datafile_name.replace('pickle','csv'))
df.flood_duration_total_list = df.flood_duration_total_list/24
df.cumulative_node_drainage_area = df.cumulative_node_drainage_area/df.soil_nodes_count
print(df.head())

# df.dropna(inplace=True)
# print(df.head())

# df.soil_clustering = df.soil_clustering/df.soil_nodes_count

datafile_name = datafile_name[2:20]
path="/Volumes/GoogleDrive/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
print(path)
if not os.path.exists(path):
    os.makedirs(path)
# print(df.head())

label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Flooded Node-Day',
    'max_flow_cfs':'Peak Flow Rate at Outlet (cfs)', 'soil_nodes_count':'% of GI Nodes', 'soil_node_distance_list': "Mean Distance from GI Nodes to Outlet", 
    'soil_node_degree_list':'Neighbor Index', 'soil_nodes_total_upstream_area':'Cumulative Upstream Area','mean_rainfall': 'Mean Rainfall Intensity',
    'antecedent_soil':'Antecedent Soil Moisture', 'mean_disp_g':'Mean DG', 'mean_disp_kg': 'Mean DKG', 'max_disp_g':'Max DG', 
    'max_disp_kg':'Max DKG', 'mean_var_path_length':'Time Averaged <L>', 'max_var_path_length': 'L max', 'max_flood_nodes':'Number of Flooded Nodes',
    'max_flood_node_degree':'Average Connectedness when Highest Number of Nodes Flooded','max_flood_node_elev':'Average Distance to Outlet when Highest Number of Nodes Flooded',
    'mean_flood_nodes_TI':'Average Distance to Outlet of Flood Node (at All Times)','total_flooded_vol_MG':'Total Flood Volume (MG)','total_outflow_vol_MG':'Total Outflow Volume (MG)',
    'cumulative_node_drainage_area':'Average Cumulative Upstream Area (Acres)','flood_node_degree_list':'Neighbor Index of Flood Nodes', 
    'flood_node_distance_list':'Mean Distance from Flood Nodes to Outlet', 'soil_clustering':'GI Nodes Clustering Coefficient','beta':'beta'} 
x_label_dict = {1.44:'1', 1.69:'2-Year', 2.15:'5', 2.59:'10-Year', 3.29:'25-Year', 
    3.89:'50\n2-Hour Storm Return Periods (Year)', 4.55:'100-Year', 5.27:'200', 6.32:'500', 7.19:'1000'}
def two_figure_plot(df = df, y1_attribute = 'soil_node_distance_list', y2_attribute = 'soil_node_degree_list',
xaxis_attribute = 'flood_duration_total_list', cmap_on = False, save_plot = True, cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
    xaxis = df[xaxis_attribute]
    xmax = max(xaxis) + 1
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
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

def three_figure_plot(df = df, x1_attribute = 'soil_node_distance_list', x2_attribute = 'soil_node_degree_list', x3_attribute = 'soil_nodes_count', 
yaxis_attribute = 'flood_duration_total_list', cmap_on = False, save_plot = True, cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis)
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    # label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Average Days of Flooding per Node',
    # 'outlet_water_level':'Water Level at Outlet (ft)', 'soil_nodes_count':'% GI Nodes', 'soil_node_distance_list': "Topography Index", 
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

def four_figure_plot(df = df, x1_attribute = 'soil_node_distance_list', x2_attribute = 'soil_node_degree_list', x3_attribute = 'soil_nodes_count', 
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

def two_axis_plot(df = df, xaxis_attribute = 'soil_nodes_count', yaxis_attribute = 'soil_node_distance_list', color_attribute = 'flood_duration_total_list', 
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


def multi_rainfall_figure_plot(df = df, ncols = 10, nrows = 1, xaxis_attribute = 'soil_node_distance_list', yaxis_attribute = 'flood_duration_total_list', 
color_attribute = 'soil_nodes_count', cmap_on = True, kde = False,save_plot = False,fitline=False, cmap = plt.cm.viridis, datafile_name = datafile_name, title = None):
    # print(df)
    k = 0
    yaxis = df[yaxis_attribute]
    ymax = max(yaxis)
    ymin = min(yaxis)
    cmap0 = cm.get_cmap(cmap)
    color_dict = {'color': cmap0(0.8), 'alpha': 0.3}
    if not color_attribute:
        c = yaxis_attribute
    else: 
        c = round(df[color_attribute], -1)
    if cmap_on:
        color_dict = {'c': c, 'alpha': 0.6, 'cmap': cmap}
    ylabel = label_dict[yaxis_attribute]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=True,figsize=(12,8))
    if ncols == 1:
        ax_plt = axes
    else: 
        ax_plt = axes[k]
    if kde is True:
        fig.suptitle('Return Periods of 2-Hour Storm')
    else: 
        fig.suptitle(label_dict[xaxis_attribute])
    mean_rainfall_set = list(set(df.mean_rainfall) - set([0]))
    mean_rainfall_set.sort()
    ax_plt.set_ylabel(ylabel)
    ax_plt.get_yaxis().labelpad = 10
    vmin = min(df[color_attribute])
    vmax = max(df[color_attribute])
    n = len(mean_rainfall_set)
    color_iteration = np.linspace(vmin, vmax, n, endpoint=True)
    for i in mean_rainfall_set:
        # x_label = str(round(i,1)) + '"'
        x_label = x_label_dict[i]
        ax_plt.grid(which = 'both', axis = 'both', alpha = 0.2)
        ax_plt.set_xlabel(x_label)
        ax_plt.xaxis.tick_top()
        plt.setp(ax_plt.spines.values(), alpha = 0.2)
        ax_plt.tick_params(labelcolor='w',top=False, bottom=False, left=False, right=False)
        ax_plt.set_ylim([ymin, ymax])
        # axes[k].set_xlim([0, xmax])
        x_centroid_list = []
        y_centroid_list = []
        centroid_color = []
        
        for j in color_iteration:
            line_color = plt.cm.viridis(j/(vmax-vmin))
            is_set = (df.mean_rainfall == i) & ((df[color_attribute] >= j) & (df[color_attribute] < (j+color_iteration[1]-color_iteration[0])))
            df_plot = df.loc[is_set]

            # c = round(df_plot[color_attribute], -1)
            c = df_plot[color_attribute]
            if cmap_on:
                color_dict = {'c': c, 'alpha': 0.3}
            
            if kde: 
                if np.var(df_plot[yaxis_attribute]) > 0:
                    try: 
                        sns.kdeplot(y=df_plot[yaxis_attribute],color=line_color,ax=ax_plt,thresh=0.8)
                    except np.linalg.LinAlgError("singular matrix"): 
                        pass
            else:
                ax_plt.scatter(df_plot[xaxis_attribute], df_plot[yaxis_attribute], s = 10, marker = 's', vmin = vmin, vmax = vmax, **color_dict)
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
                    ax_plt.plot(df_plot[xaxis_attribute], m*df_plot[xaxis_attribute]+b,color=line_color)
                    x_text = max(df_plot[xaxis_attribute])
                    y_text = m*x_text+b
                    plt.text(x_text, y_text, r'$m =$'+f'{round(m,2)}',ha = 'left', va='bottom', 
                    transform = ax_plt.transData,color = line_color)
                except TypeError:
                    pass

        scatter_fig = ax_plt.scatter(x_centroid_list, y_centroid_list, c = centroid_color, s = 5*ncols, marker = 'o',edgecolor = 'w', vmin = vmin, vmax = vmax)
        xmax = max(df[xaxis_attribute])
        xmax = 8
        # print(xaxis_attribute, yaxis_attribute,xmax)
        ax_plt.set_xlim([0, xmax])

        if ncols > 1:
            k += 1
            try: 
                ax_plt = axes[k]
            except IndexError: 
                pass
            axes[0].tick_params(labelcolor ='k')
        else: 
            ax_plt.tick_params(labelcolor ='k')
        
    plt.tight_layout(w_pad= 0.0)
    if cmap_on:
        fig.subplots_adjust(left = 0.15, right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(scatter_fig, cax=cbar_ax)
        cbar_ax.get_yaxis().labelpad = 15
        cbar_ax.set_ylabel(label_dict[color_attribute], rotation=270)
        # cbar_ax.yaxis.set_label_position('left')
        # cbar_ax.yaxis.tick_left()
    
    if title:
        fig.suptitle(title)
    if kde: 
        fig_name = datafile_name.replace(".pickle","") + '_KDE_'+yaxis_attribute
    elif ncols == 1:
        fig_name = datafile_name.replace(".pickle","") + f'_{i}-inch_{yaxis_attribute}'
    else:
        fig_name = datafile_name.replace(".pickle","") + xaxis_attribute + yaxis_attribute
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def test_one_permeability(df, soil_nodes_count, x = 'soil_node_degree_list', y = 'max_flood_nodes',title=None,save_plot = False):
    is_set = (df.soil_nodes_count > soil_nodes_count - 10) & (df.soil_nodes_count <= soil_nodes_count)
    df1 = df.loc[is_set]
    _ = plt.figure(figsize=(6,8))
    ax = plt.axes()
    mean_rainfall_set = list(set(df1.mean_rainfall) - set([0]))
    mean_rainfall_set.sort()
    vmin = min(mean_rainfall_set)
    vmax = max(mean_rainfall_set)
    color_iteration = np.linspace(vmin, vmax, len(mean_rainfall_set), endpoint=True)
    for i in mean_rainfall_set:
        label = i
        is_set = (df1.mean_rainfall == i)
        df_plot = df1.loc[is_set]
        color = plt.cm.RdYlBu((i-vmin)/(vmax-vmin))
        ax.scatter(x=df_plot[x],y=df_plot[y],s = 2, color=color,label=label,alpha=0.4)

        m,b = np.polyfit(df_plot[x], df_plot[y],1)
        ax.plot(df_plot[x], m*df_plot[x]+b,color=color)

        # correlation_matrix = np.corrcoef(df_plot[x], df_plot[y])
        # correlation_xy = correlation_matrix[0,1]
        # r_squared = correlation_xy**2

        # res = df_plot[y] - (m*df_plot[x]+b)
        # mean_y = np.mean(df_plot[y])
        # SS_tot = sum((df_plot[y] - mean_y)**2)
        # SS_res = sum(res**2)
        # r_squared = 1 - SS_res/SS_tot

        x_text = max(df_plot[x])
        y_text = m*x_text+b
        plt.text(x_text, y_text, r'$m =$'+f'{round(m,2)}',ha = 'right', va='top', 
        transform = ax.transData,color = color)

    ax.set_xlabel(label_dict[x])
    ax.set_ylabel(label_dict[y])
    leg = plt.legend(title = '2-Hr Rainfall (inch)',bbox_to_anchor = (0.5, 1), loc = 'lower center', fontsize = 'small',ncol=len(mean_rainfall_set),markerscale=3)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        # lh.set_sizes(5)

    fig_name = datafile_name.replace(".pickle","") + 'test'+ x + '_' + y + '_' + str(soil_nodes_count)+'nodes'
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if title:
        plt.suptitle(title)
    if save_plot:
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def beta_figure(df, ncols=1):
    fig = plt.figure()
    ncols = len(set(df.mean_rainfall))
    gs=fig.add_gridspec(1,ncols)
    for i in range(ncols):
        mean_rainfall = list(set(df.mean_rainfall))[i]
        ax = fig.add_subplot(gs[:,i])
        df_plt = df[(df.mean_rainfall == mean_rainfall)]
        ax.scatter(df_plt.beta, df_plt.flood_duration_total_list)
        ax.set_xlabel(str(mean_rainfall))
    plt.legend(['0.2','0.5','0.8'])

if __name__ == '__main__':
    
    # beta_figure(df)
    is_set = (df.mean_rainfall == 1.69) | (df.mean_rainfall == 2.59) | (df.mean_rainfall == 3.29 ) | (df.mean_rainfall == 4.55)
    df_plot = df.loc[is_set]
    # two_axis_plot(df = df, xaxis_attribute='soil_node_distance_list', yaxis_attribute='soil_clustering')
    for y in ['max_flow_cfs']:#,'max_flow_cfs']:
        # for y in ['flood_duration_total_list','max_flood_nodes','total_flooded_vol_MG','max_flow_cfs','total_outflow_vol_MG','flood_node_degree_list', 'flood_node_distance_list']:#, 'mean_var_path_length', 'mean_disp_kg', 'mean_disp_g']:
        multi_rainfall_figure_plot(df=df_plot, ncols = 4, nrows = 1, yaxis_attribute = y, cmap_on = True, save_plot=True,kde=True)
    #     for x in ['soil_clustering','cumulative_node_drainage_area','soil_node_degree_list','soil_node_distance_list']:#,'mean_flood_nodes_TI']:
    # #     # for x in ['soil_node_degree_list']:
    #         # test_one_permeability(df=df, x=x, y=y, save_plot=True)
    #         multi_rainfall_figure_plot(df=df_plot, ncols = 4, nrows = 1, xaxis_attribute = x, yaxis_attribute = y, cmap_on = True, save_plot=True)
    # test_one_permeability(df=df, soil_nodes_count = 30, x= 'soil_clustering',y='flood_node_degree_list', title = f'Permeability at {30}%', save_plot=False)
    # permeability_marker = [10, 20, 30, 40]
    # for i in permeability_marker:
    #     for x in ['soil_clustering']:#,'soil_node_degree_list','soil_node_distance_list']:
    #         test_one_permeability(df=df, soil_nodes_count = i, x=x, y='max_flow_cfs', save_plot=False,title = f'Permeability at {i}%')
    plt.show()