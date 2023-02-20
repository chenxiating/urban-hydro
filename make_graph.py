import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import pickle
import os
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.patches import Patch

matplotlib.rcParams['figure.figsize'] = (7, 5)
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial') 

label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Flooded Node-Day',
    'max_flow_cfs':f'Peak Flow Rate\nat Outlet (cfs)', 'avg_flow_cfs':'Average Flow Rate (cfs)','soil_nodes_count':'% of GI Nodes', 'soil_node_distance_list': "Mean Distance from GI Nodes to Outlet", 
    'soil_node_degree_list':'Nodes with More than 1 Upstream Contributor', 'cumulative_node_drainage_area':'Cumulative Upstream Area','mean_rainfall': 'Mean Rainfall Intensity',
    'antecedent_soil':'Antecedent Soil Moisture', 'mean_disp_g':'Mean DG', 'mean_disp_kg': 'Mean DKG', 'max_disp_g':'Max DG', 
    'max_disp_kg':'Max DKG', 'mean_var_path_length':'Time Averaged <L>', 'max_var_path_length': 'L max', 'max_flood_nodes':'Number of Flooded Nodes',
    'max_flood_node_degree':'Average Connectedness when Highest Number of Nodes Flooded','max_flood_node_elev':'Average Distance to Outlet when Highest Number of Nodes Flooded', 'flood_node_distance_norm':'Normalized Flood Node\n Distance to Outlet (%)','flood_node_distance_list':'Mean Flood Nodes Distance',
    'mean_flood_nodes_TI':'Average Distance to Outlet of Flood Node (at All Times)','total_flooded_vol_MG':'Total Flood Volume\n(MG)','total_outflow_vol_MG':'Total Outflow Volume (MG)',
    'cumulative_node_drainage_area':'Average Cumulative Upstream Area (Acres)','flood_node_degree_list':'Neighbor Index of Flood Nodes', 
    'flood_node_distance_list':'Mean Distance from\n Flood Nodes to Outlet', 'soil_clustering':'GI Nodes Clustering Coefficient','beta':r'Network Gibbs Parameter $\beta$', 
    'path_diff':r'Network Path Difference $H (s)$','rounded_path':"\nRounded Network Path Meandering $H (s)$",'rounded_distance':'Rounded Mean Distance from GI Nodes to Outlet',
    'path_diff_prime':r"Normalized Flow Path Meandering $H$", 'pipe_cap': r'Pipe Capacity (ft$^3$)', 'flow_to_outflow': 'Peak flow to Outflow Ratio', 'percent_flooded': f'Flooding Loss\n(% of Total Precip.)',
    'flood_to_outflow': 'Flood to Outflow'} 
rain_label_dict = {1.44:'1-Year', 1.69:'2-Year', 2.15:'5-Year', 2.59:'10-Year', 3.29:'25-Year', 
    3.89:'50\n2-Hour Storm Return Periods (Year)', 4.55:'100-Year', 5.27:'200', 6.32:'500', 7.19:'1000'}

def set_box_plot_props(color='k'):
    whp = dict(linewidth=0.3,color=color)
    fp = dict(marker='o',markersize=0,markeredgecolor=color,
    markerfacecolor=color,alpha=0.5)
    bp = dict(linewidth=0.5,alpha=0.5,facecolor=color)
    mp = dict(color=color)
    ax_dict=dict(whiskerprops = whp,showcaps=False,flierprops = fp,medianprops=mp, boxprops=bp)
    return ax_dict 

def two_figure_sns_box_plot(df,y1_attribute = 'max_flow_cfs', y2_attribute = 'total_flooded_vol_MG', x_attribute = 'soil_nodes_count', sort_attribute = 'mean_rainfall', 
hue_attr = None, ncols = 0, logx=False, pos = np.arange(0,6,1), datafile_name = None, draw_title = False, alpha_scatter = 0.05):
    if ncols == 0:
        ncols = len(set(df[sort_attribute]))
    pane_set = list(set(df[sort_attribute]))
    pane_set.sort()
    # fig = plt.figure(figsize=(9,12))
    fig = plt.figure()
    gs = fig.add_gridspec(2,ncols)
    k = 0
    ax_top_model = fig.add_subplot(gs[0,k])
    model_axis_config(ax_top_model,ylabel=label_dict[y1_attribute],logx=logx)
    ax_bottom_model = fig.add_subplot(gs[1,k])
    model_axis_config(ax_bottom_model,ylabel=label_dict[y2_attribute],logx=logx)
    fig.align_ylabels([ax_top_model,ax_bottom_model])
    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(label_dict[x_attribute],fontweight='bold')

    for i in pane_set:
        rain_label = rain_label_dict[i]
        if not hue_attr: 
            if sort_attribute == 'mean_rainfall':
                rain_label = rain_label_dict[i]
                hue_attr = 'path_diff'
            elif sort_attribute == 'path_diff':
                rain_label = rf'$H(s) =$ {int(i)}'
                hue_attr = 'mean_rainfall'
        is_set = (df[sort_attribute] == i)
        df_plot = df.loc[is_set]
        ax_top = fig.add_subplot(gs[0,k],sharex=ax_top_model,sharey=ax_top_model)        
        ax_top.set_title(rain_label,fontsize=10)
        ax_bottom = fig.add_subplot(gs[1,k],sharex=ax_bottom_model,sharey=ax_bottom_model)
        x_list = list(set(df[x_attribute]))
        x_list.sort() 
        # pos = np.arange(0,6,1)
        i = -0.2
        hue_att = list(set(df_plot[hue_attr]))
        hue_att.sort()
        labels = []
        for hue in hue_att:
            y1_plot = list()
            y2_plot = list()
            x_att = list(set(df_plot[x_attribute]))
            x_att.sort()
            pos = np.arange(0, len(x_att), 1) # DEBUG!!
            j = 0
            for x in x_att:
                df_a = df_plot[(df_plot[hue_attr] == hue) & (df_plot[x_attribute] == x)]
                y1_plot.append(df_a[y1_attribute])
                y2_plot.append(df_a[y2_attribute])
                
                xtop = np.random.normal(pos[j]+i, 0.08, len(df_a[y1_attribute]))
                ax_top.scatter(xtop, df_a[y1_attribute],alpha = alpha_scatter, c='grey',s = 0.2)
                xbot = np.random.normal(pos[j]+i, 0.08, len(df_a[y2_attribute]))
                ax_bottom.scatter(xbot, df_a[y2_attribute],alpha = alpha_scatter, c='grey',s = 0.2)
                j+=1

            top_boxes = ax_top.boxplot(y1_plot, 
                patch_artist=True,
                positions=pos+i,
                widths = 0.4,
                **set_box_plot_props(color='#6b6b6b'))
                # **set_box_plot_props(color='#144c73'))
            
            
            bottom_boxes = ax_bottom.boxplot(y2_plot, 
                patch_artist=True,
                positions=pos+i,
                widths = 0.4,
                **set_box_plot_props(color='#6b6b6b'))
            i +=0.4
            if hue_attr == 'path_diff':
                labels.append(f'$H(s)$={hue}')
                # labels = ['low H(s)', 'high H(s)']
            elif hue_attr == 'changing_diam':
                if hue:
                    labels.append("Sized to design storm")
                else: 
                    labels.append("Pipes fixed to ø 1.5'")
            else: 
                labels.append(f'{hue}')
                # labels.append(f'Changing diam: {hue}')
        for box in top_boxes['boxes']:
            box.set(facecolor = "None")
        for m in top_boxes['medians']:
            m.set(color='k')
        
        for box in bottom_boxes['boxes']:
            box.set(facecolor = 'None')
        for m in bottom_boxes['medians']:
            m.set(color='k')

        for ax in [ax_top, ax_bottom]:
            ax.set_xticks(np.arange(0,len(x_att),1),minor=False)
            # ax.grid(True,which='minor',linewidth = 1, alpha = 0.5)
            ax.tick_params(which='minor',color='w')
        ax_bottom.set_xticklabels(x_att)
        plt.setp(ax_bottom.get_xticklabels(), fontsize=8)
        axis_config(ax_top,bottom=False,left=False)
        axis_config(ax_bottom,left=False)
        k +=1
    
    plt.subplots_adjust(wspace=0, hspace=0 )
    # low_H1 = Patch(fc = '#1b699e',linewidth=0,alpha=0.5)
    # low_H2 = Patch(fc = '#c15a00',linewidth=0,alpha=0.5)
    # high_H1 = Patch(fc = '#6b6b6b',linewidth=0,alpha=0.5)
    # high_H2 = Patch(fc = '#ffd1a9',linewidth=0,alpha=0.5)
    # handles = [low_H1, high_H1, low_H2, high_H2]
    print(labels)
    low_H = Patch(fc = '#6b6b6b', ec='k', linewidth=0.5,alpha=0.5)
    high_H = Patch(fc = 'w', ec='k', linewidth=0.5,alpha=0.5)
    plt.legend(handles=[low_H, high_H], labels = labels)
    
    # plt.legend(handles=handles,#(low_H1, high_H1,low_H2, high_H2),
    #     labels=labels,
    #     frameon=True,
    #     # numpoints=1, 
    #     ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
    #     loc='lower right')

    if draw_title:
        fig.suptitle('2-Hour Storm Return Periods',fontweight='bold')
        plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')


def model_axis_config(axis,logx = False,ylabel = None):
    axis.tick_params(labelcolor='k',top=False, bottom=False, left=False, right=False, labelbottom = False)
    plt.setp(axis.spines.values(), alpha = 0)
    # for _, spine in axis.spines.items():
    #     spine.set_visible(False)
    axis.margins(x=0.2,y=0.1)
    if ylabel:
        axis.set_ylabel(ylabel,fontsize=10,fontweight='bold')
    if logx:
        axis.set_xscale('log')

def axis_config(axis, bottom = True, left = False, sns_box = False):
    # ONE TIME PLOT TO SET TOP AXIS YLIM
    # if not bottom:
    #     axis.set_ylim([10, 70])
    # else:
    #     axis.set_ylim([0, 15])

    ##
    plt.setp(axis.get_xticklabels(), fontsize=8)
    axis.tick_params(labelcolor='k',labeltop=False, labelleft=left, labelright=False, labelbottom=bottom)
    plt.xticks(rotation=45)
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    if not bottom: 
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(axis.spines.values(), alpha = 0.2)
    axis.grid(which = 'major', axis = 'y', alpha = 0.2)
    axis.set(xlabel=None,ylabel=None)
    
    try: 
        axis.get_legend().remove()
    except AttributeError:
        pass

def trendline(x,y,axis,c):
    m,b = np.polyfit(x, y, 1)
    x_array = np.sort(np.array(x))
    axis.plot(x_array,m*x_array+b,c=c)

def two_figure_strip_plot(df, ncols = 0, sort_attribute='mean_rainfall', x_attribute = 'soil_nodes_count', y1_attribute = 'max_flow_cfs', y2_attribute = 'total_flooded_vol_MG', hue_attr = None,
color_attribute = 'soil_nodes_count', cmap_on = True, save_plot = False,draw_trendline=False, logx = False, cmap = plt.cm.viridis, datafile_name = None, draw_title = False, title = None):
    if ncols == 0:
        ncols = len(set(df[sort_attribute]))
    column_set = list(set(df[sort_attribute]))
    column_set.sort()
    # fig = plt.figure(figsize=(9,12))
    fig = plt.figure()
    gs = fig.add_gridspec(2,ncols)
    k = 0
    title_phrase=''
    ax_top_model = fig.add_subplot(gs[0,k])
    model_axis_config(ax_top_model,ylabel=label_dict[y1_attribute],logx = logx)
    ax_bottom_model = fig.add_subplot(gs[1,k])
    model_axis_config(ax_bottom_model,ylabel=label_dict[y2_attribute],logx = logx)
    fig.align_ylabels([ax_top_model,ax_bottom_model])
    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(label_dict[x_attribute],fontweight='bold')
    # set up color schemes
    c = df[color_attribute]
    if sort_attribute=='mean_rainfall':
        title = '2-Hour Storm Return Periods'
    elif sort_attribute == 'soil_nodes_count':
        title = f'{label_dict[sort_attribute]}'
    # color_iteration = np.linspace(min(c), max(c), len(set(c)), endpoint=True)
    for i in column_set:
        if sort_attribute=='mean_rainfall':
            top_label = rain_label_dict[i]
        elif sort_attribute == 'soil_nodes_count':
            top_label = f'{i}%'
        ax_top = fig.add_subplot(gs[0,k],sharex=ax_top_model,sharey=ax_top_model)
        ax_top.set_title(top_label,fontsize=10)
        ax_bottom = fig.add_subplot(gs[1,k],sharex=ax_bottom_model,sharey=ax_bottom_model)

        for x in set(df[color_attribute]):
            # print(set(df[color_attribute]))
            if sort_attribute == 'soil_nodes_count':
                is_set = (df[sort_attribute] == i)
                rainfall= list(set(df.mean_rainfall))[0]
                title_phrase = f'{rain_label_dict[rainfall]} 2-Hour Storm \n'
            else:
                is_set = (df[sort_attribute] == i)
            if color_attribute == 'path_diff':
                x = 0.5*(min(df.soil_nodes_count) + max(df.soil_nodes_count))
                title_phrase = f'{int(x)}% GI Nodes \n'
            df_plot = df.loc[is_set]
            if (len(set(df[color_attribute])) > 1) and cmap_on: 
                color_dict = dict(c=df_plot[color_attribute], alpha=0.8, cmap=cmap, vmin=min(c),vmax=max(c))
                line_color = plt.cm.viridis((x-min(c))/(max(c)-min(c)))
            else: 
                line_color = 'C0'
                color_dict = dict(color=line_color, alpha=0.8)
                cmap_on = False
        ax_top = sns.boxplot(data=df_plot,x=x_attribute,y=y1_attribute,ax=ax_top,**set_box_plot_props(color='C0'))
        # ax_top = sns.stripplot(data=df_plot,x=x_attribute,y=y1_attribute,ax=ax_top,color='C0',size=2)
        ax_bottom= sns.boxplot(data=df_plot,x=x_attribute,y=y2_attribute,ax=ax_bottom,**set_box_plot_props(color='C1'))
        # ax_bottom = sns.stripplot(data=df_plot,x=x_attribute,y=y2_attribute,ax=ax_bottom,color='C1',size=2)
        axis_config(ax_top,bottom=False)
        axis_config(ax_bottom)
        if draw_trendline:
            trendline(df_plot[x_attribute], df_plot[y1_attribute], ax_top,c=line_color)
            trendline(df_plot[x_attribute], df_plot[y2_attribute], ax_bottom,c=line_color)
        k +=1
    
    if len(set(df.soil_nodes_count)) == 1:
        title = title_phrase + title
        x_attribute = f'{str(set(df.soil_nodes_count))}_{x_attribute}'
    
    if len(set(df.mean_rainfall)) == 1:
        title = title_phrase + title
        x_attribute = f'{str(set(df.mean_rainfall))}_{x_attribute}'
    
    if len(set(df.path_diff)) == 1:
        title_phrase = rf'$H(s)$ = {[int(x) for x in set(df.path_diff)][0]}' + ',\t'
        title = title_phrase + title
        x_attribute = f'{str(set(df.path_diff))}_{x_attribute}'
        
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(ax_bottom.get_xticklabels(), fontsize=8)
    fig_name = f'{datafile_name.replace(".pickle","")}_strip_{x_attribute}_{y1_attribute}_{y2_attribute}'
    if draw_title:
        fig.suptitle(title,fontweight='bold')
        plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        path="/Users/xchen/Library/CloudStorage/GoogleDrive-chen7090@umn.edu/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
        plt.savefig(path + fig_name +'.pdf')
        print('Plot is saved as', fig_name +'.pdf')

# def two_figure_scatter_plot(df, ncols = 0, sort_attribute='mean_rainfall', x_attribute = 'soil_clustering', y1_attribute = 'max_flow_cfs', y2_attribute = 'total_flooded_vol_MG', 
# color_attribute = 'soil_nodes_count', cmap_on = True, save_plot = False,draw_trendline=False, logx = False, cmap = plt.cm.viridis, datafile_name = None, title = None):
#     if ncols == 0:
#         ncols = len(set(df[sort_attribute]))
#     column_set = list(set(df[sort_attribute]))
#     column_set.sort()
#     fig = plt.figure(figsize=(9,12))
#     gs = fig.add_gridspec(2,ncols)
#     k = 0
#     title_phrase=''
#     ax_top_model = fig.add_subplot(gs[0,k])
#     model_axis_config(ax_top_model,ylabel=label_dict[y1_attribute],logx = logx)
#     ax_bottom_model = fig.add_subplot(gs[1,k])
#     model_axis_config(ax_bottom_model,ylabel=label_dict[y2_attribute],logx = logx)
#     fig.align_ylabels([ax_top_model,ax_bottom_model])
#     fig.add_subplot(111,frameon=False)
#     plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
#     plt.xlabel(label_dict[x_attribute],fontweight='bold')
#     # set up color schemes
#     c = df[color_attribute]
#     if sort_attribute=='mean_rainfall':
#         title = '2-Hour Storm Return Periods'
#     elif sort_attribute == 'soil_nodes_count':
#         title = f'{label_dict[sort_attribute]}'
#     # color_iteration = np.linspace(min(c), max(c), len(set(c)), endpoint=True)
#     for i in column_set:
#         if sort_attribute=='mean_rainfall':
#             top_label = rain_label_dict[i]
#         elif sort_attribute == 'soil_nodes_count':
#             top_label = f'{i}%'
#         ax_top = fig.add_subplot(gs[0,k],sharex=ax_top_model,sharey=ax_top_model)
#         ax_top.set_title(top_label,fontsize=10)
#         ax_bottom = fig.add_subplot(gs[1,k],sharex=ax_bottom_model,sharey=ax_bottom_model)
#         axis_config(ax_top,bottom=False)
#         axis_config(ax_bottom)
#         for x in set(df[color_attribute]):
#             # print(set(df[color_attribute]))
#             if sort_attribute == 'soil_nodes_count':
#                 is_set = (df[sort_attribute] == i)
#                 rainfall= list(set(df.mean_rainfall))[0]
#                 title_phrase = f'{rain_label_dict[rainfall]} 2-Hour Storm \n'
#             else:
#                 is_set = (df[sort_attribute] == i) & (df[color_attribute] == x) 
#             if color_attribute == 'path_diff':
#                 x = 0.5*(min(df.soil_nodes_count) + max(df.soil_nodes_count))
#                 title_phrase = f'{int(x)}% GI Nodes \n'
#             df_plot = df.loc[is_set]
#             if (len(set(df[color_attribute])) > 1) and cmap_on: 
#                 color_dict = dict(c=df_plot[color_attribute], alpha=0.8, cmap=cmap, vmin=min(c),vmax=max(c))
#                 line_color = plt.cm.viridis((x-min(c))/(max(c)-min(c)))
#             else: 
#                 line_color = 'C0'
#                 color_dict = dict(color=line_color, alpha=0.8)
#                 cmap_on = False
#                 # title_phrase = f'{int(x)}% GI Nodes \n'
#             scatter_fig = ax_top.scatter(df_plot[x_attribute], df_plot[y1_attribute],s = 5, **color_dict)
#             ax_bottom.scatter(df_plot[x_attribute], df_plot[y2_attribute],s = 5, color = 'C1')
#             if draw_trendline:
#                 trendline(df_plot[x_attribute], df_plot[y1_attribute], ax_top,c=line_color)
#                 trendline(df_plot[x_attribute], df_plot[y2_attribute], ax_bottom,c=line_color)
#         k +=1
    
#     if len(set(df.soil_nodes_count)) == 1:
#         title = title_phrase + title
#         x_attribute = f'{str(set(df.soil_nodes_count))}_{x_attribute}'
    
#     if len(set(df.mean_rainfall)) == 1:
#         title = title_phrase + title
#         x_attribute = f'{str(set(df.mean_rainfall))}_{x_attribute}'
    
#     if len(set(df.path_diff)) == 1:
#         title_phrase = rf'$H(s)$ = {[int(x) for x in set(df.path_diff)][0]}' + ',\t'
#         title = title_phrase + title
#         x_attribute = f'{str(set(df.path_diff))}_{x_attribute}'
        
#     plt.subplots_adjust(wspace=0, hspace=0)
#     # plt.setp(ax_bottom, xticks=[y + 1 for y in range(len(x_list))],
#     # xticklabels=list(str(int(x)) for x in x_list))
#     # plt.setp(ax_bottom.get_xticklabels(), fontsize=8)

#     if cmap_on:
#         fig.subplots_adjust(left = 0.15, right = 0.8)
#         cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.35])
#         cbar = fig.colorbar(scatter_fig, cax=cbar_ax)
#         cbar_ax.get_yaxis().labelpad = 15
#         cbar_ax.set_ylabel(label_dict[color_attribute], rotation=270)
#         # cbar_ax.yaxis.set_label_position('left')
#         # cbar_ax.yaxis.tick_left()
#     fig.suptitle(title,fontweight='bold')
#     fig_name = f'{datafile_name.replace(".pickle","")}_scatter_{x_attribute}_{y1_attribute}_{y2_attribute}'
#     plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
#     if save_plot:
#         path="/Users/xchen/Library/CloudStorage/GoogleDrive-chen7090@umn.edu/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
#         plt.savefig(path + fig_name +'.pdf')
#         print('Plot is saved as', fig_name +'.pdf')

# def test_one_permeability(df, soil_nodes_count, x = 'soil_node_degree_list', y = 'max_flood_nodes',title=None,save_plot = False):
#     is_set = (df.soil_nodes_count > soil_nodes_count - 10) & (df.soil_nodes_count <= soil_nodes_count)
#     df1 = df.loc[is_set]
#     # _ = plt.figure(figsize=(6,8))
#     _ = plt.figure()
#     ax = plt.axes()
#     mean_rainfall_set = list(set(df1.mean_rainfall) - set([0]))
#     mean_rainfall_set.sort()
#     vmin = min(mean_rainfall_set)
#     vmax = max(mean_rainfall_set)
#     color_iteration = np.linspace(vmin, vmax, len(mean_rainfall_set), endpoint=True)
#     for i in mean_rainfall_set:
#         label = i
#         is_set = (df1.mean_rainfall == i)
#         df_plot = df1.loc[is_set]
#         color = plt.cm.RdYlBu((i-vmin)/(vmax-vmin))
#         ax.scatter(x=df_plot[x],y=df_plot[y],s = 2, color=color,label=label,alpha=0.4)

#         m,b = np.polyfit(df_plot[x], df_plot[y],1)
#         ax.plot(df_plot[x], m*df_plot[x]+b,color=color)

#         # correlation_matrix = np.corrcoef(df_plot[x], df_plot[y])
#         # correlation_xy = correlation_matrix[0,1]
#         # r_squared = correlation_xy**2

#         # res = df_plot[y] - (m*df_plot[x]+b)
#         # mean_y = np.mean(df_plot[y])
#         # SS_tot = sum((df_plot[y] - mean_y)**2)
#         # SS_res = sum(res**2)
#         # r_squared = 1 - SS_res/SS_tot

#         x_text = max(df_plot[x])
#         y_text = m*x_text+b
#         plt.text(x_text, y_text, r'$m =$'+f'{round(m,2)}',ha = 'right', va='top', 
#         transform = ax.transData,color = color)

#     ax.set_xlabel(label_dict[x])
#     ax.set_ylabel(label_dict[y])
#     leg = plt.legend(title = '2-Hr Rainfall (inch)',bbox_to_anchor = (0.5, 1), loc = 'lower center', fontsize = 'small',ncol=len(mean_rainfall_set),markerscale=3)
#     for lh in leg.legendHandles: 
#         lh.set_alpha(1)
#         # lh.set_sizes(5)

#     fig_name = datafile_name.replace(".pickle","") + 'test'+ x + '_' + y + '_' + str(soil_nodes_count)+'nodes'
#     plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
#     if title:
#         plt.suptitle(title)
#     if save_plot:
#         plt.savefig(path + fig_name +'.pdf')
#         print('Plot is saved as', fig_name +'.pdf')

def gibbs_pdf(all_deltaH_df):
    fig, ax0 = plt.subplots(figsize = (6,4))
    i = 0
    color = ['C0','C1','C2','C3']
    for beta in [0.2, 0.4, 0.6, 0.8]:
        new_list = all_deltaH_df[all_deltaH_df.beta == beta]["delta H"]
        ax0.hist(new_list,density=True,orientation='vertical',color=color[i],alpha=0.5, label= f"$\beta$ = {beta}")
        
        i+=1
        # ax0.legend([rf'$\beta$ = {beta}'])
    ax0.legend([rf'$\beta$ = 0.2', rf'$\beta$ = 0.4', rf'$\beta$ = 0.6', rf'$\beta$ = 0.8'],
    loc='lower right',ncol=1)
    # ax0.legend(loc = 'lower right', ncol=1)
    ax0.set_ylabel('Histogram frequency')
    ax0.set_xlabel(r'Path difference between actual path and shortest path $H (s)$')
    ax0.axes.xaxis.set_ticks([])
    ax0.axes.yaxis.set_ticks([])
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    

def initialize(datafile_name):
    df = pickle.load(open(datafile_name, 'rb'))
    print(f'{datafile_name}: {df.shape}')
    df.to_csv(datafile_name.replace('pickle','csv'))
    df.flood_duration_total_list = df.flood_duration_total_list/24
    df.cumulative_node_drainage_area = df.cumulative_node_drainage_area/df.soil_nodes_count
    df.soil_nodes_count = df.soil_nodes_count.astype(int)
    df['flow_to_outflow'] = df['max_flow_cfs']/df['total_outflow_vol_MG']
    df['percent_flooded'] = 100*df['total_flooded_vol_MG']/(df.total_precip_MG)
    df['flood_to_outflow'] = df['total_flooded_vol_MG']/(df.total_outflow_vol_MG)
    # df['percent_flooded'] = 100*df['total_flooded_vol_MG']/(100*df.mean_rainfall)
    try: 
        df.flood_node_distance_norm = df.flood_node_distance_norm*100
        df.flood_node_distance_list = df.flood_node_distance_list
    except AttributeError:
        pass

    # df.dropna(inplace=True)
    # df.soil_clustering = df.soil_clustering/df.soil_nodes_count
    path="/Users/xchen/Library/CloudStorage/GoogleDrive-chen7090@umn.edu/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
    if not os.path.exists(path):
        os.makedirs(path)

    return df

def import_pickles(path):
    try: 
        os.chdir(path)
        all_files = os.listdir(path)
        all_files = [path+one_file for one_file in all_files if (one_file[-6:] == "pickle")]
    except NotADirectoryError:
        all_files.append(path)
    main_df = pd.DataFrame()

    for one_file in all_files:
        df = pickle.load(open(one_file,'rb'))
        df = df.rename(columns = {0: "delta H"})
        find_beta = one_file.find('beta')+4
        beta = float(one_file[find_beta:find_beta+3])
        df['beta'] = beta
        main_df = pd.concat([main_df,df])
    return main_df

if __name__ == '__main__':
    ## Initialization:
    path = r'/Users/xchen/Library/CloudStorage/GoogleDrive-chen7090@umn.edu/My Drive/urban-stormwater-analysis/writing/GI_network/figures/DYNWAVE_202302/'
 
    # ## Gibbs PDF
    # deltaH_pickle_path = r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20220805-1152/'
    # deltaH_df = import_pickles(deltaH_pickle_path)
    # gibbs_pdf(deltaH_df)
    # # plt.savefig(path+'beta_vs_H.pdf',dpi=300)
    
    def round_path(df_raw):
        # n = 300
        n=0.5
        rainfall_list = [1.69, 2.59, 3.29,4.55]
        print(f'min diam is {set(df_raw.min_diam)}')
        df = df_raw[(df_raw.mean_rainfall.isin(rainfall_list))]
        # df["rounded_path"] = np.floor(df.path_diff/n)*n
        df["rounded_path"] = round(df.path_diff_prime/n)*n
        # df["rounded_path"] = df["rounded_path"].astype(int)
        df = df[df["path_diff_prime"] < 1.5]
        # df['rounded_path_group'] = df.apply(lambda x: f"$H \geq 1$" if (x["path_diff_prime"] >= 1) else f"$H < 0.5$" if (x["path_diff_prime"] < 0.5) else f"$0.5 \leq H < 1$", axis = 1)
        # print("$H < 0.5$:", df[(df.mean_rainfall == 1.69) & (df.rounded_path_group == "$H < 0.5$")].shape )
        # print("0.5 < H <= 1:", df[(df.mean_rainfall == 1.69) & (df.rounded_path_group == "$0.5 \leq H < 1$")].shape )
        # print("$H >= 1$: ", df[(df.mean_rainfall == 1.69) & (df.rounded_path_group == "$H \geq 1$")].shape)
        df['rounded_path_group'] = df.apply(lambda x: r"$H (s) \approx 0.2$" if (x["path_diff_prime"] >= 0.1) else r"$H (s) \approx 0.02$", axis = 1)

        return df

    ### Network structure ###
    """Figure 1"""
    # # fixed_diam = r'./data_beta_generator/20211010_fixing_diam.pickle'
    # # fixed_diam = r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/SWMM_20220929-1422/summary.pickle'
    # # fixed_diam = r'./SWMM_20221019-1319-64nodes/summary_min2.pickle'
    
    # # fixed_diam = r'./SWMM_20221024-1022-121nodes/summary_min2_fixed_length.pickle'
    # # fixed_diam = r'./SWMM_20220929-2303-100nodes/summary_min2_fixed_length.pickle'
    # fixed_diam = r'./SWMM_20221027-1146/summary_min2_fixed_length.pickle'
    # # fixed_diam = r'./SWMM_20221020-1859-81nodes/summary_min2_fixed_length.pickle'
    # # fixed_diam = r'./SWMM_20221019-1319-64nodes/summary_min2_fixed_length.pickle'
    # # fixed_diam = r'./SWMM_20221020-1135-49nodes/summary_min2_fixed_length.pickle'
    # # fixed_diam = r'./SWMM_20221020-1127-36nodes/summary_min2.pickle'
    # # fixed_diam = r'./SWMM_20221007-1542-25nodes/summary_min2_fixed_length.pickle'

    # big_df0 = initialize(r'./SWMM_20220929-2303-100nodes/summary_min2_fixed_length.pickle')
    # # big_df = round_path(big_df0[big_df0.changing_diam == 1])
    # big_df = round_path(big_df0)

    # df0 = initialize(fixed_diam)
    # # df = round_path(df0[df0.changing_diam == 1])
    # df = round_path(df0)
    # new_df = pd.concat([big_df, df])
    # new_df = pd.concat([big_df, df])

    # df0 = initialize(r'./SWMM_20221028-1327/summary_min2_fixed_length.pickle')
    # # df = round_path(df0[df0.changing_diam == 1])
    # df = round_path(df0)
    # new_df = pd.concat([big_df, df])

    # # df = initialize(r'./SWMM_20220929-2303-100nodes/GI_changin_diam_summary.pickle')

    ## Highly Impervious
    df1 = initialize(r'./SWMM_20220929-2303-100nodes/GI_changin_diam_summary_high_imp_DYNWAVE.pickle')
    df2 = initialize(r'./SWMM_20220929-2303-100nodes/GI_changin_diam_summary_suburb_DYNWAVE.pickle') 
    df1['land_type'] = 'High Imp'
    df2['land_type'] = 'Suburb'
    df = pd.concat([df1, df2])
    df.to_csv(r'./SWMM_20220929-2303-100nodes/GI_changin_diam_summary_concat.csv')

    def select_df(df, inc = 0.5, n = 250):
        np.random.seed(0)
        new_df = pd.DataFrame()
        df = df[df.changing_diam == True]
        df = round_path(df)
        df = df[df['path_diff_prime'] <=1.5]
        for land in set(df['land_type']):
            for rain in set(df['mean_rainfall']):
                rain_df = df[(df['mean_rainfall'] == rain) & (df['land_type'] == land)]
                for k in np.arange(0,1.5,inc):
                    sub_df = rain_df[(rain_df['path_diff_prime']< k+inc) & (rain_df['path_diff_prime']>= k)]
                    # print(rain, k, sub_df.shape)
                    sel_df = sub_df.sample(n)
                    # print(rain, k, sub_df.shape, sel_df.shape)
                    new_df = pd.concat([new_df, sel_df])
        return new_df

    _, ax = plt.subplots()
    df[(df['mean_rainfall'] < 2)]['path_diff_prime'].hist(grid = False,bins= [0,1.0,1.5])
    print('old', df.shape)
    ax.set_ylabel('Count of networks')
    ax.set_xlabel(f"$H$")
    # plt.savefig(path+"hist_bw_Hp_all_highImp.pdf")

    # new_df = round_path(df)
    new_df = select_df(df, n = 100)
    print('old', df.shape, 'new', new_df.shape)
    _, ax = plt.subplots()
    new_df[(new_df['mean_rainfall'] < 2)]['path_diff_prime'].hist(grid = False, bins= [0,1.0,1.5])
    
    ax.set_ylabel('Count of networks')
    ax.set_xlabel(f"$H$")
    # plt.savefig(path+"hist_bw_Hp_all_highImp.pdf")

    # two_figure_sns_box_plot(new_df, x_attribute='rounded_path', y1_attribute='max_flow_cfs', sort_attribute = 'mean_rainfall',y2_attribute='percent_flooded', datafile_name='', hue_attr = 'land_type')
    # plt.savefig(path+"network_structure_max_flow.pdf")
    two_figure_sns_box_plot(new_df, x_attribute='rounded_path', y1_attribute='avg_flow_cfs', sort_attribute = 'mean_rainfall',y2_attribute='percent_flooded', datafile_name='', hue_attr = 'land_type')
    plt.savefig(path+"network_structure_avg_flow.pdf")

    # fig, ax = plt.subplots()
    # ax.set_ylim([0,45])
    # sns.regplot(data = new_df[new_df.mean_rainfall.isin([1.69]) & new_df.changing_diam == True], x = 'path_diff', y = 'max_flow_cfs', scatter_kws = {'s': 0.5, 'alpha': 0.1})
    # sns.regplot(data = new_df[new_df.mean_rainfall.isin([3.29]) & new_df.changing_diam == True], x = 'path_diff', y = 'max_flow_cfs', scatter_kws = {'s': 0.5, 'alpha': 0.1})

    # fig, ax = plt.subplots()
    # ax.set_ylim([0,45])
    # sns.regplot(data = new_df[new_df.mean_rainfall.isin([1.69]) & new_df.changing_diam == True], x = 'path_diff', y = 'max_flow_cfs', scatter_kws = {'s': 0.5, 'alpha': 0.1})
    # sns.regplot(data = new_df[new_df.mean_rainfall.isin([3.29]) & new_df.changing_diam == True], x = 'path_diff', y = 'max_flow_cfs', scatter_kws = {'s': 0.5, 'alpha': 0.1})
    
    ### Interaction between network and green infrastructure ###
    """Figure 2"""
    def select_df_Hprime(df, n = 200):
        np.random.seed(0)
        new_df = pd.DataFrame()
        df = df[df.changing_diam == True]
        df = round_path(df)
        df = df[df['path_diff_prime'] <=1.5]
        for land in set(df['rounded_path_group']):
            for rain in set(df['mean_rainfall']):
                rain_df = df[(df['mean_rainfall'] == rain) & (df['rounded_path_group'] == land)]
                for k in set(df['soil_nodes_count']):
                    sub_df = rain_df[(rain_df['soil_nodes_count']== k)]
                    sel_df = sub_df.sample(n)
                    new_df = pd.concat([new_df, sel_df])
        print('shape of fig 2 df:', new_df.shape)
        return new_df
    # datafile_name= r'./SWMM_20211107-1609/20211108-1036_full_dataset_100-nodes.pickle'
    # df = initialize(datafile_name)

    # GI_placement_datafile_name1 = r'./SWMM_placement_20221207_smallH/GI_coverage_summary_highly_impervious.pickle'
    # GI_placement_datafile_name2 = r'./SWMM_placement_20221207_largeH/GI_coverage_summary_highly_impervious.pickle'
    # df1 = initialize(GI_placement_datafile_name1)
    # df1['H type'] = f'H(s) < 400'
    # df2 = initialize(GI_placement_datafile_name2)
    # df2['H type'] = f'H(s) > 700'
    # df = pd.concat([df1, df2])

    ## Highly impervious
    # df = initialize(r'./gibbs10_20221219-1304_H200+800/GI_coverage_summary_highly_imp.pickle')
    # df = initialize(r'./10-grid_20221207_lt400+gt700/GI_coverage_summary_highly_imp.pickle')
    df = initialize(r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20221227-Hp=0.02+0.2/20221227-2212_GI_coverage_summary_highly_imp.pickle')
    df = initialize(r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20221227-Hp=0.02+0.2/20230217-1035_GI_coverage_summary_highly_imp_DYNWAVE.pickle')
    _, ax = plt.subplots()
    df[(df['mean_rainfall'] < 2) & (df['soil_nodes_count']== 0)]['path_diff_prime'].hist(grid = False)
    ax.set_ylabel('Count of networks')
    ax.set_xlabel(f"$H$")
    # plt.savefig(path+"hist_bw_Hp0.02-0.2_highImp.pdf")


    df = round_path(df)
    df = df[df.changing_diam == True]
    # df = df[df.rounded_path_group != "$0.5 \leq H < 1$"]
    df = select_df_Hprime(df, n=200)
    two_figure_sns_box_plot(df, y1_attribute = 'avg_flow_cfs', y2_attribute = 'percent_flooded', sort_attribute = 'mean_rainfall', hue_attr = 'rounded_path_group')
    plt.savefig(path+"GI_network_interaction_avg_flow.pdf")
    two_figure_sns_box_plot(df, y2_attribute = 'percent_flooded', sort_attribute = 'mean_rainfall', hue_attr = 'rounded_path_group')
    plt.savefig(path+"GI_network_interaction_max_flow.pdf")

    
    # sns.histplot(data = df_plt, x = 'soil_nodes_count')

    # GI_placement_datafile_name1 = r'./SWMM_placement_20221207_smallH/GI_coverage_summary_highly_impervious.pickle'
    # GI_placement_datafile_name2 = r'./SWMM_placement_20221207_largeH/GI_coverage_summary_highly_impervious.pickle'
    # df1 = initialize(GI_placement_datafile_name1)
    # df1['H type'] = f'H(s) < 400'
    # df2 = initialize(GI_placement_datafile_name2)
    # df2['H type'] = f'H(s) > 700'
    # df = pd.concat([df1, df2])
    # df = df[df.changing_diam == True]
    # two_figure_sns_box_plot(df[df.mean_rainfall.isin([1.69,3.29])], sort_attribute = 'mean_rainfall', hue_attr = 'H type')
    # plt.savefig(path+"GI_network_interaction_newyaxis_bw_Htype_highImp.pdf")

    # ## Suburb
    # df = initialize(r'./gibbs10_20221219-1304_H200+800/GI_coverage_summary_suburb.pickle')
    # df = df[df.changing_diam == True]
    # two_figure_sns_box_plot(df[df.mean_rainfall.isin([1.69,3.29])], sort_attribute = 'mean_rainfall', hue_attr = 'path_diff')
    # plt.savefig(path+"GI_network_interaction_newyaxis_bw_200+800_suburb.pdf")

    ### Interaction between network and placement ###
    """Figure 3"""
    def round_distance(df):
        n = 4
        df["rounded_distance"] = np.floor(df.soil_node_distance_list/n)*n
        df["rounded_distance"] = df["rounded_distance"].astype(int)
        df['rounded_distance'].where(df['rounded_distance'] < 20, 16, inplace = True) # DEBUG!
        df = df[df.changing_diam == True]
    
    # # ## High Imp - H Type
    # # GI_placement_datafile_name1 = r'./SWMM_placement_20221207_smallH/GI_distance_summary_highly_impervious.pickle'
    # # GI_placement_datafile_name2 = r'./SWMM_placement_20221207_largeH/GI_distance_summary_highly_impervious.pickle'
    # # df1 = initialize(GI_placement_datafile_name1)
    # # df1['H type'] = f'H(s) < 400'
    # # df2 = initialize(GI_placement_datafile_name2)
    # # df2['H type'] = f'H(s) > 700'
    # # df = pd.concat([df1, df2])
    # # round_distance(df)
    # # two_figure_sns_box_plot(df[df.mean_rainfall.isin([1.69,3.29])],x_attribute = 'rounded_distance', datafile_name = 'network_placement', pos = np.arange(0,5,1), hue_attr = 'H type')
    # # plt.savefig(path+'GI_placement_interaction_newyaxis_bw_htype_highImp.pdf')
    
    ## High Imp 
    # df = initialize(r'./gibbs10_20221219-1304_H200+800/GI_distance_summary_highly_impervious.pickle')
    df = initialize(r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20221227-Hp=0.02+0.2/20221227-2047_GI_distance_summary_highly_impervious.pickle')
    df = initialize(r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/gibbs10_20221227-Hp=0.02+0.2/20230217-2326_GI_distance_summary_highly_impervious_DYNWAVE.pickle')

    df = round_path(df)
    round_distance(df)
    two_figure_sns_box_plot(df[df.mean_rainfall.isin([1.69, 2.59, 3.29, 4.55])],x_attribute = 'rounded_distance', y2_attribute = 'percent_flooded', datafile_name = 'network_placement', 
    pos = np.arange(0,4,1), hue_attr = 'rounded_path_group', alpha_scatter=0.01)
    plt.savefig(path+'GI_placement_max_flow.pdf')

    two_figure_sns_box_plot(df[df.mean_rainfall.isin([1.69, 2.59, 3.29, 4.55])],x_attribute = 'rounded_distance', y1_attribute = 'avg_flow_cfs', y2_attribute = 'percent_flooded', datafile_name = 'network_placement', 
    pos = np.arange(0,4,1), hue_attr = 'rounded_path_group', alpha_scatter=0.01)
    plt.savefig(path+'GI_placement_avg_flow.pdf')

    # df_plt = df[df.mean_rainfall.isin([1.69])]# * (df.path_diff == 200)]
    # print(df_plt.shape)

    # plt.subplots()
    # df_plt = df[df.mean_rainfall.isin([1.69])]
    # sns.histplot(data = df_plt, x = 'path_diff_prime', hue = 'path_diff')

    # ## Suburb
    # df = initialize(r'./gibbs10_20221219-1304_H200+800/GI_distance_summary_suburb.pickle')
    # round_distance(df)
    # two_figure_sns_box_plot(df[df.mean_rainfall.isin([1.69,3.29])],x_attribute = 'rounded_distance', datafile_name = 'network_placement', pos = np.arange(0,5,1), hue_attr = 'path_diff')
    # plt.savefig(path+'GI_placement_interaction_newyaxis_bw_200+800_suburb.pdf')

    # ## APPENIX FIGURES
    # GI_placement_datafile_name = r'./SWMM_placement_128_20220517/GI_distance_summary.pickle'
    # df = initialize(GI_placement_datafile_name)
    # round_distance(df)
    # two_figure_strip_plot(df, x_attribute = 'rounded_distance', save_plot=False, datafile_name = 'GI_placement')
    # # # plt.savefig(path+'GI_placement28_new_rainfalls.pdf')

    # ## Green infrastructure number
    # two_figure_strip_plot(df[df.mean_rainfall.isin([1.69, 2.59, 3.29, 4.55]) & (df.path_diff == 28)],
    # save_plot=True, datafile_name = datafile_name[2:20])
    # plt.savefig(path+"GI_number.pdf")

    plt.show()