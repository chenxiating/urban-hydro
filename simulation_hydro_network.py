import hydro_network as hn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import datetime as date
import datetime
import time
from statistics import mean
from scipy.special import factorial
from scipy.stats import gamma
import os

def main(nodes_num = int(100), process_core_name = None, antecedent_soil_moisture = 0.1, mean_rainfall_inch = 1, days = 50, dt_str = None, kernel=None, soil_nodes_range = [0, 100, 100]):
    ## Assign Network Properties ##
    # In this step we build the network based on different criteria
    np.random.seed(seed = 1358)
    #G = pickle.load(open('graph_10nodes', 'rb'))
    node_drainage_area = 5000
    outlet_level = {0: 1} 
    outlet_elev = {0: 85}                 
    outlet_node_drainage_area = {0: node_drainage_area*10e5}             # set the river area to very large
    soil_depth = 6
    init_level = 0.0
    flood_level = 10
    soil_moisture = antecedent_soil_moisture

    ## Precipitation
    # Rainfall generation. Units will be presented in foot. 
    dt = 0.1
    simulation_timesteps = round(days/dt)
    #npad = round(simulation_timesteps/2)
    depth = hn.rainfall_func(size=simulation_timesteps,freq=0.1,meanDepth_inch=mean_rainfall_inch, dt = dt, is_pulse=True)
    #depth = np.pad([1], (npad, simulation_timesteps - npad - 1), 'constant', constant_values = (0))
    timesteps = np.linspace(0, simulation_timesteps*dt, num = simulation_timesteps)

    if not dt_str:
        today = date.datetime.today()
        dt_str = today.strftime("%Y%m%d-%H%M")
        print('Not dt_str')

    # Simulations
    for network in range(1):
        new_network_time = time.time()
        time_before_random_sample_soil_nodes = time.time()
        soil_nodes_combo, soil_nodes_combo_count = hn.random_sample_soil_nodes(range_min = soil_nodes_range[0], range_max = soil_nodes_range[1], range_count = soil_nodes_range[2], nodes_num = nodes_num)
        main_df = pd.DataFrame()
        datafile_name = 'dataset_'+str(round(mean_rainfall_inch,1))+'-inch_'+str(nodes_num)+'-nodes_'+str(days)+'-day_'+dt_str+'soil_moisture-'+str(round(antecedent_soil_moisture,1))+'_'+str(process_core_name)+'_'+str(soil_nodes_range[1])+'.pickle'
        # output_columns =['soil_nodes_list', "flood_duration_list", "flood_duration_total_list", 'max_outlet_water_level', 'mean_rainfall', 'antecedent_soil'
        # "soil_node_degree_list", "soil_node_elev_list", 'soil_nodes_total_upstream_area','mean_disp_g','mean_disp_kg','max_disp_g',
        # 'max_disp_kg','mean_var_path_length', 'max_var_path_length','max_flood_nodes','max_flood_node_degree','max_flood_node_elev',
        # 'flood_node_degree','flood_node_elev','mean_flood_nodes_TI']
        output_df = pd.DataFrame()#, columns=output_columns)
        output_df.loc[:,'soil_nodes_list'] = soil_nodes_combo
        k = 0
        kk = 0
        for soil_nodes in output_df['soil_nodes_list']:
            time_to_create_network = time.time()
            H = hn.create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_drainage_area = node_drainage_area, outlet_level = outlet_level, 
            outlet_node_drainage_area = outlet_node_drainage_area, outlet_elev= outlet_elev, kernel=kernel)
            rain_nodes = H.nodes
            degrees = dict(H.degree())
            mean_of_edges = sum(degrees.values())/len(degrees)
            # print('mean degree: ', mean_of_edges)
            soil_nodes_length = len(soil_nodes)
            soil_nodes_total_upstream_area = hn.accumulate_downstream(H, soil_nodes = soil_nodes)

            soil_nodes_init = {k: {'soil_depth': soil_depth, 'soil_moisture': antecedent_soil_moisture} for k in soil_nodes}
            nx.set_node_attributes(H, soil_nodes_init)
            non_soil_nodes_init = hn.fill_numbers(nx.get_node_attributes(H, 'soil_moisture'), H.nodes, number = 1)
            nx.set_node_attributes(H, non_soil_nodes_init, 'soil_moisture')

            flood_nodes_list = []
            max_flood_nodes_list = []
            max_flood_nodes = 0
            max_outlet_water_level = H.nodes[0]['level']
            flood_nodes = 0
            flood_nodes_TI_list = [0]
            flood_time = 0
            disp_df = pd.DataFrame()
            var_path_length_list = []
            disp_g_list = []
            disp_kg_list = []
            outlet_level_list = []
            # time_before_simulation = time.time()
            for i in range(0,simulation_timesteps):
                # print("day = ", i*dt)
                # print('dhdt',{k: H.nodes[k]['dhdt'] for k in H.nodes})

                #time_openf = hn.print_time(start_time)
                #### Need to add Darcy's Law to this part. Gets a bit tricky if both overland
                #### flows and subsurface flows occur in the network. 
                # hn.Manning_func(gph = H, flood_level = flood_level)
                h_new, soil_moisture = hn.rainfall_nodes_func(gph = H, dt = dt, s = soil_moisture, zr = soil_depth, soil_nodes = soil_nodes, 
                rain_nodes = rain_nodes, depth = depth[i])
                hn.Manning_func(gph = H)    # calculate hydraulic radius and calculate flow rate
                var_path_length, disp_g, disp_kg = hn.dispersion_func(gph = H)
                # if depth[i] > 0:
                    # print(depth[i])
                    # print('dhdt',{k: H.nodes[k]['dhdt'] for k in H.nodes})
                    # print('level',{k: H.nodes[k]['level'] for k in H.nodes})
                    # print('overflow',{k: H.nodes[k]['overflow'] for k in H.nodes})


                var_path_length_list.append(var_path_length)
                disp_g_list.append(disp_g)
                disp_kg_list.append(disp_kg)
                outlet_level_list.append(H.nodes[0]['level'])

                if H.nodes[0]['level'] > max_outlet_water_level:
                    max_outlet_water_level = H.nodes[0]['level']
                flood_nodes_this_round = [k for k in H.nodes if (k != 0) & (h_new[k]>= flood_level)]
                if len(flood_nodes_this_round) > max_flood_nodes:
                    max_flood_nodes_list = flood_nodes_this_round
                    max_flood_nodes = len(flood_nodes_this_round)
                    # print({k: H.nodes[k]['level'] for k in flood_nodes_this_round})
                    # print({m: H.edges[m]['edge_dq'] for m in H.edges})

                    # print("day = ", i*dt, max_flood_nodes)
                flood_nodes_list.extend(flood_nodes_this_round) 
                flood_nodes_list = list(set(flood_nodes_list))
                if flood_nodes_this_round:
                    flood_nodes_TI_list.append(hn.ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1 
            for k in flood_nodes_list),len(flood_nodes_list)))
                flood_nodes = flood_nodes + sum(h_new[k]>= flood_level for k in H.nodes if k != 0)
                flood_time = flood_time + (max(h_new.values()) >= flood_level)
            # hn.draw_network_timestamp(gph = H, soil_nodes = soil_nodes, label_on = False, flood_level = flood_level)
            # # hn.graph_histogram(gph=H,kernel=kernel)
            # plotstuff(gph=H, x = np.array(range(i+1))*dt, depth=depth[0:i+1], dispersion = [],outlet_level=outlet_level_list)

            # plt.show()
            flood_duration = dt*flood_time
            flood_duration_total = dt*flood_nodes
            soil_node_degree = hn.ignore_zero_div(sum(degrees.get(k,0) for k in soil_nodes),soil_nodes_length)
            soil_node_elev = hn.ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1 
            for k in soil_nodes),soil_nodes_length)
            max_flood_node_degree = hn.ignore_zero_div(sum(degrees.get(k,0) for k in max_flood_nodes_list),len(max_flood_nodes_list))
            max_flood_node_elev = hn.ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1 
            for k in max_flood_nodes_list),len(max_flood_nodes_list))
            # flood_node_degree = hn.ignore_zero_div(sum(degrees.get(k,0) for k in flood_nodes_list),len(flood_nodes_list))
            # flood_node_elev = hn.ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1 
            # for k in flood_nodes_list),len(flood_nodes_list))
            mean_flood_nodes_TI = mean(flood_nodes_TI_list)
            # out_edges = H.in_edges(0, data = False)
            # out_edge_wl = [0]
            # for i in out_edges:
            #     out_edge_wl = out_edge_wl + edge_wl[i]
            
            output_df.loc[k,'flood_duration_list'] = flood_duration
            output_df.loc[k,'flood_duration_total_list'] = flood_duration_total
            output_df.loc[k,'soil_node_degree_list'] = soil_node_degree
            output_df.loc[k,'soil_node_elev_list'] = soil_node_elev
            output_df.loc[k,'soil_nodes_count'] = soil_nodes_length/nodes_num*100
            # print(output_df)
            # print(k)
            # print('max_outlet_water_level', max_outlet_water_level)
            output_df.loc[k,'max_outlet_water_level'] = max_outlet_water_level
            output_df.loc[k,'soil_nodes_total_upstream_area'] = soil_nodes_total_upstream_area
            output_df.loc[k,'mean_disp_g'] = mean(disp_g_list)
            output_df.loc[k,'mean_disp_kg'] = mean(disp_kg_list)
            output_df.loc[k,'max_disp_g'] = max(disp_g_list)
            output_df.loc[k,'max_disp_kg'] = max(disp_kg_list)
            output_df.loc[k,'mean_rainfall'] = mean_rainfall_inch
            output_df.loc[k,'antecedent_soil'] = antecedent_soil_moisture
            output_df.loc[k,'max_flood_nodes'] = max_flood_nodes
            output_df.loc[k,'max_flood_node_degree'] = max_flood_node_degree
            output_df.loc[k,'max_flood_node_elev'] = max_flood_node_elev
            output_df.loc[k,'mean_var_path_length'] = mean(var_path_length_list)
            output_df.loc[k,'max_var_path_length'] = max(var_path_length_list)
            
            # output_df.loc[k,'flood_node_degree'] = flood_node_degree
            # output_df.loc[k,'flood_node_elev'] = flood_node_elev
            output_df.loc[k,'mean_flood_nodes_TI'] = mean_flood_nodes_TI
            k += 1
            kk += 2
        print("Process core:", process_core_name, "| Antecedent soil moisture: ", antecedent_soil_moisture, "| Mean rainfall:", mean_rainfall_inch)
        print("Run time: ")
        hn.print_time(new_network_time)
        main_df = pd.concat([main_df, output_df], ignore_index=True)
        f = open(datafile_name,'wb')
        pickle.dump(output_df, f)
        f.close()
        plt.show()
    
    # print("File name is: ", datafile_name, "File size: ", os.path.getsize(datafile_name), "Total time: ")
def four_subplots(days, simulation_timesteps, depth, disp_df, outlet_level):
    """
    this plot function is created for debugging.
    """
    x = np.linspace(0,days,num=simulation_timesteps)
    fig, axs = plt.subplots(2,1)
    #axs = plt.gca()
    disp_g_1 = list(disp_df.loc[:,0])
    axs[0].scatter(x, disp_g_1, s=1, label = '25 nodes')
    ax0 = axs[0].twinx()
    ax0.plot(x,depth, label = 'Precipitation (ft)',alpha = 0.5)
    ax0.plot(x, outlet_level,label = 'Outlet Level (ft)')
    
    disp_kg_1 = list(disp_df.loc[:,1])
    axs[1].scatter(x, disp_kg_1, s=1, label = '25 nodes')
    ax1 = axs[1].twinx()
    ax1.plot(x,depth, label = 'Precipitation (ft)',alpha = 0.5)
    ax1.plot(x, outlet_level,label = 'Outlet Level (ft)')

    disp_g_2 = list(disp_df.loc[:,2])
    axs[0].scatter(x, disp_g_2, s=1, label = '50 nodes')

    disp_kg_2 = list(disp_df.loc[:,3])
    axs[1].scatter(x, disp_kg_2, s=1, label = '50 nodes')
    axs[0].legend()
    axs[1].legend()

def plotstuff(gph, x, depth, dispersion, outlet_level):
    """
    this plot function is created for debugging.
    """
    # try: 
    #     x
    #     print(x)
    # except NameError:
    #     x = np.linspace(0,days,num=simulation_timesteps)
    # print('x', len(x))
    # print('depth', len(depth))
    # print('disp', len(dispersion))
    plt.subplots()
    ax = plt.gca()
    # ax.scatter(x, dispersion)
    ax.set_ylabel('Precipitation (ft)', color = 'C0')
    ax2 = ax.twinx()
    ax.plot(x,depth, color = 'C0')
    ax2.plot(x,outlet_level,color = 'C1')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Outlet Level (ft)',color = 'C1')
    # fig2, ax3 = plt.subplots()
    # hn.draw_varying_size(gph, ax = ax3, attribute='level', edge_attribute = 'length', node_drawing_ratio=0)
    # plt.legend()

# def plotstuff(gph, x, depth, dispersion, outlet_level, days = days, simulation_timesteps = simulation_timesteps):
#     try: 
#         x
#         print(x)
#     except NameError:
#         x = np.linspace(0,days,num=simulation_timesteps)
#     plt.subplots()
#     ax = plt.gca()
#     ax.scatter(x, dispersion)
#     ax.set_ylabel = 'Dispersion Coefficient (L2/T)'
#     ax2 = ax.twinx()
#     ax2.plot(x,depth, label = 'Precipitation (ft)')
#     ax2.plot(x,outlet_level,label = 'Outlet Level (ft)')
#     ax2.set_xlabel = 'Feet'
#     # fig2, ax3 = plt.subplots()
#     # hn.draw_varying_size(gph, ax = ax3, attribute='level', edge_attribute = 'length', node_drawing_ratio=0)
#     plt.legend()

