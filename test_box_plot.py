from readline import append_history_file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from make_graph import model_axis_config, axis_config, initialize
from make_graph import two_figure_sns_box_plot as sns_plot

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
    cp = dict(linewidth=0.8,color='w',alpha=0.8)
    fp = dict(marker='s',markersize=1,markeredgecolor=color,
    markerfacecolor=color,alpha=0.5)
    mp = dict(color=color)
    bp = dict(linewidth=0.1,alpha=0.5,facecolor=color)
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
            print(f'Hue attributes is {hue}')
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
                widths = 0.3,
                **set_box_plot_props(color='#1f77b4'))
                # **set_box_plot_props(color='#144c73'))
            
            bottom_boxes = ax_bottom.boxplot(y2_plot, 
                patch_artist=True,
                positions=pos+i,
                widths = 0.3,
                **set_box_plot_props(color='#c15a00'))
            i +=0.4
        for box in top_boxes['boxes']:
            box.set(facecolor = '#82bfe9')
        for m in top_boxes['medians']:
            m.set(color='#419ede')
        
        for box in bottom_boxes['boxes']:
            box.set(facecolor = '#ffd1a8')
        for m in bottom_boxes['medians']:
            m.set(color='#ff7f0f')

        for ax in [ax_top, ax_bottom]:
            ax.set_xticks(np.arange(0,6,1),minor=False)
            ax.set_xticks(np.arange(0,5,1)+0.5,minor=True)
            ax.grid(True,which='minor',linewidth = 1, alpha = 0.5)
            ax.tick_params(which='minor',color='w')
        ax_bottom.set_xticklabels(np.arange(0,60,10))
        plt.setp(ax_bottom.get_xticklabels(), fontsize=8)
        axis_config(ax_top,bottom=False,left=False)
        axis_config(ax_bottom,left=False)
        
        k +=1
    
    plt.subplots_adjust(wspace=0, hspace=0, top =0.85)
    low_H = mpatches.Rectangle((0,0),1,1, fc ='#000000',linewidth=1,alpha=0.5)
    high_H = mpatches.Rectangle((0,0),1,1, fc ='#8e99a1',linewidth=1,alpha=0.5)
    handles = ((low_H,low_H),high_H)
    labels = ['$H(s)=28.0$','$H(s)=74.0$']
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(),by_label.keys(),frameon=False,
    bbox_to_anchor=(-0.,2.25),loc='upper center',ncol = len(labels), borderaxespad=0.)
    
    if draw_title:
        fig.suptitle('2-Hour Storm Return Periods',fontweight='bold')
        plt.figtext(0, 0, "Source: "+datafile_name.replace(".pickle",""), fontsize = 6, color = '#696969')


if __name__ == "__main__":
    ## Interaction between network and green infrastructure
    datafile_name= r'./SWMM_20211107-1609/20211108-1036_full_dataset_100-nodes.pickle'
    df = initialize(datafile_name)
    two_figure_sns_box_plot(df[df.mean_rainfall.isin([2.15,3.29])],datafile_name = datafile_name[2:20])
    # sns_plot(df[df.mean_rainfall.isin([2.15,3.29])],datafile_name = datafile_name[2:20])
    
    from matplotlib.colors import to_hex
    print(to_hex('C1'))


    plt.show()