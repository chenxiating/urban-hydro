import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch



matplotlib.rcParams['figure.figsize'] = (6, 6)
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial') 

label_dict = {'flood_duration_list':'Days with Flooding', 'flood_duration_total_list': 'Flooded Node-Day',
    'max_flow_cfs':'Peak Flow Rate at Outlet\n(cfs)', 'soil_nodes_count':'% of GI Nodes', 'soil_node_distance_list': "Mean Distance from GI Nodes to Outlet", 
    'soil_node_degree_list':'Neighbor Index', 'cumulative_node_drainage_area':'Cumulative Upstream Area','mean_rainfall': 'Mean Rainfall Intensity',
    'antecedent_soil':'Antecedent Soil Moisture', 'mean_disp_g':'Mean DG', 'mean_disp_kg': 'Mean DKG', 'max_disp_g':'Max DG', 
    'max_disp_kg':'Max DKG', 'mean_var_path_length':'Time Averaged <L>', 'max_var_path_length': 'L max', 'max_flood_nodes':'Number of Flooded Nodes',
    'max_flood_node_degree':'Average Connectedness when Highest Number of Nodes Flooded','max_flood_node_elev':'Average Distance to Outlet when Highest Number of Nodes Flooded',
    'mean_flood_nodes_TI':'Average Distance to Outlet of Flood Node (at All Times)','total_flooded_vol_MG':'Total Flood Volume\n(MG)','total_outflow_vol_MG':'Total Outflow Volume (MG)',
    'cumulative_node_drainage_area':'Average Cumulative Upstream Area (Acres)','flood_node_degree_list':'Neighbor Index of Flood Nodes', 
    'flood_node_distance_list':'Mean Distance from Flood Nodes to Outlet', 'soil_clustering':'GI Nodes Clustering Coefficient','beta':r'Network Gibbs Parameter $\beta$', 
    'path_diff':r'Network Path Difference $H$','rounded_path':'\nRounded Network Path Difference $H$','rounded_distance':'Rounded Mean Distance from GI Nodes to Outlet'} 
rain_label_dict = {1.44:'1-Year', 1.69:'2-Year', 2.15:'5-Year', 2.59:'10-Year', 3.29:'25-Year', 
    3.89:'50\n2-Hour Storm Return Periods (Year)', 4.55:'100-Year', 5.27:'200', 6.32:'500', 7.19:'1000'}

def set_box_plot_props(color='k'):
    whp = dict(linewidth=0.3,color=color)
    fp = dict(marker='o',markersize=1,markeredgecolor=color,
    markerfacecolor=color,alpha=0.5)
    bp = dict(linewidth=0,alpha=0.5,facecolor=color)
    mp = dict(color=color)
    ax_dict=dict(whiskerprops = whp,showcaps=False,flierprops = fp,medianprops=mp, boxprops=bp)
    return ax_dict 

def two_figure_sns_box_plot(df,y1_attribute = 'max_flow_cfs', y2_attribute = 'total_flooded_vol_MG', x_attribute = 'soil_nodes_count', sort_attribute = 'mean_rainfall', 
ncols = 0, logx=False, title=None, datafile_name = None, draw_title = False, save_plot = True):
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
        pos = np.arange(0,6,1)
        i = -0.2
        hue_att = list(set(df_plot[hue_attr]))
        hue_att.sort()
        for hue in hue_att:
            y1_plot = list()
            y2_plot = list()
            x_att = list(set(df_plot[x_attribute]))
            x_att.sort()
            for x in x_att:
                df_a = df_plot[(df_plot[hue_attr] == hue) & (df_plot[x_attribute] == x)]
                y1_plot.append(df_a[y1_attribute])
                y2_plot.append(df_a[y2_attribute])
            top_boxes = ax_top.boxplot(y1_plot, 
                patch_artist=True,
                positions=pos+i,
                widths = 0.4,
                **set_box_plot_props(color='#1b699e'))
                # **set_box_plot_props(color='#144c73'))
            
            bottom_boxes = ax_bottom.boxplot(y2_plot, 
                patch_artist=True,
                positions=pos+i,
                widths = 0.4,
                **set_box_plot_props(color='#c15a00'))
            i +=0.4
        for box in top_boxes['boxes']:
            box.set(facecolor = '#82bfe9')
        for m in top_boxes['medians']:
            m.set(color='#419ede')
        
        for box in bottom_boxes['boxes']:
            box.set(facecolor = '#ffd1a9')
        for m in bottom_boxes['medians']:
            m.set(color='#ff9a42')

        for ax in [ax_top, ax_bottom]:
            ax.set_xticks(np.arange(0,6,1),minor=False)
            ax.set_xticks(np.arange(0,5,1)+0.5,minor=True)
            # ax.grid(True,which='minor',linewidth = 1, alpha = 0.5)
            ax.tick_params(which='minor',color='w')
        ax_bottom.set_xticklabels(np.arange(0,60,10))
        plt.setp(ax_bottom.get_xticklabels(), fontsize=8)
        axis_config(ax_top,bottom=False,left=False)
        axis_config(ax_bottom,left=False)
        k +=1
    
    plt.subplots_adjust(wspace=0, hspace=0, top =0.85)
    low_H1 = Patch(fc = '#1b699e',linewidth=0,alpha=0.5)
    low_H2 = Patch(fc = '#c15a00',linewidth=0,alpha=0.5)
    high_H1 = Patch(fc = '#82bfe9',linewidth=0,alpha=0.5)
    high_H2 = Patch(fc = '#ffd1a9',linewidth=0,alpha=0.5)
    handles = ((low_H1, high_H1),(low_H2, high_H2),)
    labels = ['$H(s)=28.0$','$H(s)=74.0$',]
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(),by_label.keys(),frameon=False,
    # bbox_to_anchor=(-0.,2.25),loc='upper center',ncol = len(labels), borderaxespad=0.)
    plt.legend(handles=(low_H1, high_H1,low_H2, high_H2),
        labels=['','','$H(s)=28.0$','$H(s)=74.0$'],
        frameon=False,
        bbox_to_anchor=(-0.,2.25),
        numpoints=1, 
        ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
        loc='upper center')


    # pa1 = Patch(facecolor='red', edgecolor='black')
    # pa2 = Patch(facecolor='blue', edgecolor='black')
    # pa3 = Patch(facecolor='green', edgecolor='black')
    # #
    # pb1 = Patch(facecolor='pink', edgecolor='black')
    # pb2 = Patch(facecolor='orange', edgecolor='black')
    # pb3 = Patch(facecolor='purple', edgecolor='black')

    # ax.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3],
    #         labels=['', '', '', '', 'First', 'Second'],
    #         ncol=3, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
    #         loc='center', fontsize=16)




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
    
    plt.setp(axis.get_xticklabels(), fontsize=8)
    axis.tick_params(labelcolor='k',labeltop=False, labelleft=left, labelright=False, labelbottom=bottom)
    plt.xticks(rotation=45)
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
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

def two_figure_strip_plot(df, ncols = 0, sort_attribute='mean_rainfall', x_attribute = 'soil_nodes_count', y1_attribute = 'max_flow_cfs', y2_attribute = 'total_flooded_vol_MG', 
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
        path="/Volumes/GoogleDrive/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def two_figure_scatter_plot(df, ncols = 0, sort_attribute='mean_rainfall', x_attribute = 'soil_clustering', y1_attribute = 'max_flow_cfs', y2_attribute = 'total_flooded_vol_MG', 
color_attribute = 'soil_nodes_count', cmap_on = True, save_plot = False,draw_trendline=False, logx = False, cmap = plt.cm.viridis, datafile_name = None, title = None):
    if ncols == 0:
        ncols = len(set(df[sort_attribute]))
    column_set = list(set(df[sort_attribute]))
    column_set.sort()
    fig = plt.figure(figsize=(9,12))
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
        axis_config(ax_top,bottom=False)
        axis_config(ax_bottom)
        for x in set(df[color_attribute]):
            # print(set(df[color_attribute]))
            if sort_attribute == 'soil_nodes_count':
                is_set = (df[sort_attribute] == i)
                rainfall= list(set(df.mean_rainfall))[0]
                title_phrase = f'{rain_label_dict[rainfall]} 2-Hour Storm \n'
            else:
                is_set = (df[sort_attribute] == i) & (df[color_attribute] == x) 
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
                # title_phrase = f'{int(x)}% GI Nodes \n'
            scatter_fig = ax_top.scatter(df_plot[x_attribute], df_plot[y1_attribute],s = 5, **color_dict)
            ax_bottom.scatter(df_plot[x_attribute], df_plot[y2_attribute],s = 5, color = 'C1')
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
    # plt.setp(ax_bottom, xticks=[y + 1 for y in range(len(x_list))],
    # xticklabels=list(str(int(x)) for x in x_list))
    # plt.setp(ax_bottom.get_xticklabels(), fontsize=8)

    if cmap_on:
        fig.subplots_adjust(left = 0.15, right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.35])
        cbar = fig.colorbar(scatter_fig, cax=cbar_ax)
        cbar_ax.get_yaxis().labelpad = 15
        cbar_ax.set_ylabel(label_dict[color_attribute], rotation=270)
        # cbar_ax.yaxis.set_label_position('left')
        # cbar_ax.yaxis.tick_left()
    fig.suptitle(title,fontweight='bold')
    fig_name = f'{datafile_name.replace(".pickle","")}_scatter_{x_attribute}_{y1_attribute}_{y2_attribute}'
    plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')
    if save_plot:
        path="/Volumes/GoogleDrive/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
        plt.savefig(path + fig_name +'.png')
        print('Plot is saved as', fig_name +'.png')

def test_one_permeability(df, soil_nodes_count, x = 'soil_node_degree_list', y = 'max_flood_nodes',title=None,save_plot = False):
    is_set = (df.soil_nodes_count > soil_nodes_count - 10) & (df.soil_nodes_count <= soil_nodes_count)
    df1 = df.loc[is_set]
    # _ = plt.figure(figsize=(6,8))
    _ = plt.figure()
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



def initialize(datafile_name):
    nodes_num = 100
    df = pickle.load(open(datafile_name, 'rb'))
    df.to_csv(datafile_name.replace('pickle','csv'))
    # df = pd.read_csv(datafile_name.replace('pickle','csv'))
    df.flood_duration_total_list = df.flood_duration_total_list/24
    df.cumulative_node_drainage_area = df.cumulative_node_drainage_area/df.soil_nodes_count
    df.soil_nodes_count = df.soil_nodes_count.astype(int)
    # print(df.head())

    # df.dropna(inplace=True)
    # print(df.head())
    # df.soil_clustering = df.soil_clustering/df.soil_nodes_count
    path="/Volumes/GoogleDrive/My Drive/urban-stormwater-analysis/figures/models/"+ datafile_name+"/"
    # print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return df

if __name__ == '__main__':
    ## Initialization:
    # datafile_name = r'./gibbs/20211201_multi_beta_fixed_diam.pickle'
    # datafile_name = r'./SWMM_20211107-1609/20211108-1036_full_dataset_100-nodes.pickle'
    # datafile_name = r'./SWMM_20211130-distance/compiled.pickle'
    path = r'/Volumes/GoogleDrive/My Drive/urban-stormwater-analysis/writing/GI_network/figures/'
    
    def round_path(df_raw):
        n = 400
        rainfall_list = [1.69, 2.59, 3.29,4.55]
        print(f'min diam is {set(df_raw.min_diam)}')
        df = df_raw[(df_raw.mean_rainfall.isin(rainfall_list))]
        df["rounded_path"] = round(df.path_diff/n)*n
        df["rounded_path"] = df["rounded_path"].astype(int)
        print(df.head())
        return df

    ## Network structure 
    fixed_diam = r'./data_beta_generator/20211010_fixing_diam.pickle'
    df0 = initialize(fixed_diam)
    df = round_path(df0[df0.changing_diam == 0])
    two_figure_strip_plot(df, x_attribute='rounded_path',sort_attribute = 'mean_rainfall', datafile_name='')
    plt.savefig(path+"network_structure.png")

    ## Interaction between network and green infrastructure
    datafile_name= r'./SWMM_20211107-1609/20211108-1036_full_dataset_100-nodes.pickle'
    df = initialize(datafile_name)
    two_figure_sns_box_plot(df[df.mean_rainfall.isin([2.15,3.29])],datafile_name = datafile_name[2:20])
    plt.savefig(path+"GI_network_interaction.png")

    ## Green infrastructure number
    two_figure_strip_plot(df[df.mean_rainfall.isin([1.69, 2.59, 3.29, 4.55]) & (df.path_diff == 28)],
    save_plot=True, datafile_name = datafile_name[2:20])
    plt.savefig(path+"GI_number.png")
    
    ## GI placement
    def round_distance(df):
        n = 3
        df["rounded_distance"] = np.ceil(df.soil_node_distance_list/n)*n
        df["rounded_distance"] = df["rounded_distance"].astype(int)
        
        print(df.head())

    GI_placement_datafile_name = r'./SWMM_placement_28_20221030/GI_distance_summary.pickle'
    df = initialize(GI_placement_datafile_name)
    round_distance(df)
    two_figure_strip_plot(df, x_attribute = 'rounded_distance', save_plot=False, datafile_name = 'GI_placement')
    plt.savefig(path+'GI_placement28_new_rainfalls.png')
 
    plt.show()