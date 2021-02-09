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
import os

def main(nodes_num = int(100), process_core_name = None, antecedent_soil_moisture = 0.1, mean_rainfall_inch = 1):
    ## Assign Network Properties ##
    # In this step we build the network based on different criteria
    np.random.seed(seed = 1358)
    #G = pickle.load(open('graph_10nodes', 'rb'))
    outlet_level = {0: 1}                  
    outlet_node_area = {0: 10e8}             # set the river area to very large
    soil_depth = 6
    init_level = 0.0
    flood_level = 1.5
    soil_moisture = antecedent_soil_moisture

    G = hn.create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_area = 500, 
    outlet_level = outlet_level, outlet_node_area = outlet_node_area, kernel = lambda x: x**10)
    rain_nodes = G.nodes

    ## Precipitation
    # Rainfall generation. Units will be presented in foot. 
    dt = 0.1
    days = 50
    simulation_timesteps = round(days/dt)
    #npad = round(simulation_timesteps/2)
    depth = hn.rainfall_func(size=simulation_timesteps,freq=0.1,meanDepth_inch=mean_rainfall_inch, dt = dt, is_pulse=True)
    #depth = np.pad([1], (npad, simulation_timesteps - npad - 1), 'constant', constant_values = (0))
    timesteps = np.linspace(0, simulation_timesteps*dt, num = simulation_timesteps)

    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")

    # Simulations
    for network in range(5):
        new_network_time = time.time()
        G = hn.create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_area = 500, 
        outlet_level = outlet_level, outlet_node_area = outlet_node_area)
        time_before_random_sample_soil_nodes = time.time()
        soil_nodes_combo, soil_nodes_combo_count = hn.random_sample_soil_nodes(range_min = 0, range_max = 100, range_count = 100, nodes_num = nodes_num)
        print("Process core:", process_core_name, "antecedent soil moisture: ", antecedent_soil_moisture, "mean rainfall:", mean_rainfall_inch)
        print("network:", network + 1, "Soil nodes count:", soil_nodes_combo_count)
        time_after_random_sample_soil_nodes = hn.print_time(time_before_random_sample_soil_nodes)
        # print("Time after random sample soil nodes:")
        # print(time_after_random_sample_soil_nodes)
        main_df = pd.DataFrame()
        datafile_name = 'dataset_'+str(mean_rainfall_inch)+'-inch_'+str(nodes_num)+'-nodes_'+str(days)+'-day_'+dt_str+'network_count-'+str(network)+'_'+str(process_core_name)+'.pickle'
        output_columns =['soil_nodes_list', "flood_duration_list", "flood_duration_total_list", 'outlet_water_level', 'mean_rainfall', 'antecedent_soil'
        "soil_node_degree_list", "soil_node_elev_list", 'soil_nodes_total_upstream_area','mean_disp_g','mean_disp_kg','max_disp_g',
        'max_disp_kg','mean_var_path_length', 'max_var_path_length']
        output_df = pd.DataFrame(np.nan, index=range(soil_nodes_combo_count), columns=output_columns)
        output_df.loc[:,'soil_nodes_list'] = soil_nodes_combo
        k = 0
        # disp_df = pd.DataFrame()
        kk = 0
        for soil_nodes in output_df['soil_nodes_list']:
            H = G.copy()
            soil_nodes_total_upstream_area = hn.accumulate_downstream(H, soil_nodes = soil_nodes)
            # print(soil_nodes_total_upstream_area)
            soil_nodes_length = len(soil_nodes)
            soil_nodes_depth = dict(zip(soil_nodes, np.ones(soil_nodes_length)*soil_depth))
            nx.set_node_attributes(H, soil_nodes_depth, "soil_depth")
            # sl = pd.DataFrame(np.nan, index=range(0,simulation_timesteps+1), columns=G.nodes)
            # water_level = []
            # wl = pd.DataFrame(np.nan, index=range(0,simulation_timesteps+1), columns=G.nodes)
            # edge_wl = pd.DataFrame(np.nan, index=range(0,simulation_timesteps+1), columns=G.edges)
            flood_nodes = 0
            flood_time = 0
            var_path_length_list = []
            disp_g_list = []
            disp_kg_list = []
            outlet_level_list = []
            for i in range(0,simulation_timesteps):
                # print("day = ", i*dt)
                #time_openf = hn.print_time(start_time)
                #### Need to add Darcy's Law to this part. Gets a bit tricky if both overland
                #### flows and subsurface flows occur in the network. 
                hn.Manning_func(gph = H)    # calculate hydraulic radius and calculate flow rate
                h_new, soil_moisture = hn.rainfall_nodes_func(gph = H, dt = dt, s = soil_moisture, zr = soil_depth, soil_nodes = soil_nodes, 
                rain_nodes = rain_nodes, depth = depth[i])
                # if depth[i] > 0:
                    # print('Network =', network,'soil_nodes', soil_nodes, 'day = ', i*dt, '. It rained!')
                var_path_length, disp_g, disp_kg = hn.dispersion_func(gph = H)
                # print(disp_g, disp_kg)
                var_path_length_list.append(var_path_length)
                disp_g_list.append(disp_g)
                disp_kg_list.append(disp_kg)
                outlet_level_list.append(H.nodes[0]['level'])

                # print(list(edge_travel_time))
                # water_level.append(max(h_new.values()))
                # sl.loc[simulation_timesteps] = s
                # wl.loc[simulation_timesteps] = h_new
                # flood_nodes = flood_nodes + len([v for k, v in h_new.items() if (k >0 and v>= flood_level)])
                flood_nodes = flood_nodes + sum(h_new[k]>= flood_level for k in H.nodes if k != 0)
                flood_time = flood_time + (max(h_new.values()) >= flood_level)
                ## count how many nodes were above flood level!!!!!!!
                # edge_wl.loc[simulation_timesteps]=çedge_h
            #     if i%50 == 0:
            #         hn.draw_network_timestamp(gph = H, soil_nodes = soil_nodes)
            #         plotstuff(gph = H, x = np.array(range(i+1))*dt, depth = depth[0:i+1], 
            # dispersion = disp_g_list, outlet_level = outlet_level_list)
            #         plotstuff(gph = H, x = np.array(range(i+1))*dt, depth = depth[0:i+1], 
            # dispersion = disp_kg_list, outlet_level = outlet_level_list)
            #         plotstuff(gph = H, x = np.array(range(i+1))*dt, depth = depth[0:i+1], 
            # dispersion = var_path_length_list, outlet_level = outlet_level_list)

            ## Properties and Performance of the network 
            #print("Run", k,"of", soil_nodes_combo_count, soil_nodes, "Network no.", network + 1, "|| node count", len(soil_nodes))
            #print("Time to run Manning & simulation: ")
            #hn.print_time(time_soil_nodes)
            degrees = dict(H.degree())
            mean_of_edges = sum(degrees.values())/len(degrees)
            flood_duration = dt*flood_time
            flood_duration_total = dt*flood_nodes/nodes_num
            outlet_water_level = H.nodes[0]['level']
            # soil_node_degree = hn.ignore_zero_div(sum(degrees.get(k,0) for k in soil_nodes),len(H.nodes))
            soil_node_degree = hn.ignore_zero_div(sum(degrees.get(k,0) for k in soil_nodes),soil_nodes_length)
            # soil_node_elev = hn.ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1
            # for k in soil_nodes),len(H.nodes))
            soil_node_elev = hn.ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1 
            for k in soil_nodes),soil_nodes_length)
            # out_edges = H.in_edges(0, data = False)
            # out_edge_wl = [0]
            # for i in out_edges:
            #     out_edge_wl = out_edge_wl + edge_wl[i]
            
            output_df.loc[k,'flood_duration_list'] = flood_duration
            output_df.loc[k,'flood_duration_total_list'] = flood_duration_total
            output_df.loc[k,'soil_node_degree_list'] = soil_node_degree
            output_df.loc[k,'soil_node_elev_list'] = soil_node_elev
            output_df.loc[k,'soil_nodes_combo_count'] = soil_nodes_length
            output_df.loc[k,'outlet_water_level'] = outlet_water_level
            output_df.loc[k,'soil_nodes_total_upstream_area'] = soil_nodes_total_upstream_area
            output_df.loc[k,'mean_disp_g'] = mean(disp_g_list)
            output_df.loc[k,'mean_disp_kg'] = mean(disp_kg_list)
            output_df.loc[k,'max_disp_g'] = max(disp_g_list)
            output_df.loc[k,'max_disp_kg'] = max(disp_kg_list)
            output_df.loc[k,'mean_rainfall'] = mean_rainfall_inch
            output_df.loc[k,'antecedent_soil'] = antecedent_soil_moisture

            # print(output_df)
            #output_df['outlet_max_list'].loc[k] = max(out_edge_wl)
            # disp_df.loc[:,kk] = disp_g_list
            # disp_df.loc[:,kk+1] = disp_kg_list
            #print(disp_df)
            k += 1
            kk += 2
        # four_subplots(days = days, simulation_timesteps = simulation_timesteps, depth = depth, disp_df = disp_df, outlet_level = outlet_level_list)
        print("Run time: ")
        hn.print_time(new_network_time)
        main_df = pd.concat([main_df, output_df], ignore_index=True)
        f = open(datafile_name,'wb')
        pickle.dump(output_df, f)
        f.close()
        # plt.show()
    
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
    ax.scatter(x, dispersion)
    ax.set_ylabel = 'Dispersion Coefficient (L2/T)'
    ax2 = ax.twinx()
    ax2.plot(x,depth, label = 'Precipitation (ft)')
    ax2.plot(x,outlet_level,label = 'Outlet Level (ft)')
    ax2.set_xlabel = 'Feet'
    # fig2, ax3 = plt.subplots()
    # hn.draw_varying_size(gph, ax = ax3, attribute='level', edge_attribute = 'length', node_drawing_ratio=0)
    plt.legend()

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

if __name__ == '__main__':
    main()
