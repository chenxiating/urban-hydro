import hydro_network as hn
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import datetime as date
import datetime
import time
import os

def main(nodes_num = int(100), process_core_name = None):
    ## Assign Network Properties ##
    # In this step we build the network based on different criteria
    np.random.seed(seed = 1358)
    #G = pickle.load(open('graph_10nodes', 'rb'))
    outlet_level = {0: 0.2}                  # set outlet river water level to be constant
    outlet_node_area = {0: 10e8}           # set the river area to very large
    soil_depth = 6
    init_level = 0.05
    flood_level = 1.5
    s = 0.1                               # initial soil moisture
    # nodes_num = int(10)

    G = hn.create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_area = 500, 
    outlet_level = outlet_level, outlet_node_area = outlet_node_area)
    rain_nodes = G.nodes

    ## Precipitation
    # Rainfall generation. Units will be presented in foot. 
    dt = 0.1
    days = 50
    simulation_timesteps = round(days/dt)
    #simulation_timesteps = 10
    #npad = round(simulation_timesteps/2)
    meanDepth_inch = 1
    depth = hn.rainfall_func(size=simulation_timesteps,freq=0.1,meanDepth_inch=meanDepth_inch, dt = dt)
    #depth = np.pad([1], (npad, simulation_timesteps - npad - 1), 'constant', constant_values = (0))
    timesteps = np.linspace(0, simulation_timesteps*dt, num = simulation_timesteps)
    precip0 = [0]* len(G.nodes)

    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    time_openf = time.time()
    file_directory = os.path.dirname(os.path.abspath(__file__))
    datafile_directory=file_directory +'/datafiles_'+dt_str
    print('os.path.exists(datafile_directory)', os.path.exists(datafile_directory))
    if not os.path.exists(datafile_directory):
        os.makedirs(datafile_directory)
    os.chdir(datafile_directory)

    # Simulations
    for network in range(4):
        new_network_time = time.time()
        G = hn.create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_area = 500, 
        outlet_level = outlet_level, outlet_node_area = outlet_node_area)
        time_before_random_sample_soil_nodes = time.time()
        soil_nodes_combo, soil_nodes_combo_count = hn.random_sample_soil_nodes(range_min = 0, range_max = 100, range_count = 100, nodes_num = nodes_num)
        time_after_random_sample_soil_nodes = hn.print_time(time_before_random_sample_soil_nodes)
        print("Time after random sample soil nodes:")
        print(time_after_random_sample_soil_nodes)
        # main_df = pd.DataFrame()
        datafile_name = 'dataset_'+str(meanDepth_inch)+'-inch_'+str(nodes_num)+'-nodes_'+str(days)+'-day_'+dt_str+'network_count-'+str(network)+'_'+str(process_core_name)+'.pickle'
        output_columns =['soil_nodes_list', "flood_duration_list", "flood_duration_total_list", 'outlet_water_level', 
        "soil_node_degree_list", "soil_node_elev_list", 'soil_nodes_total_upstream_area']
        output_df = pd.DataFrame(np.nan, index=range(soil_nodes_combo_count), columns=output_columns)
        output_df.loc[:,'soil_nodes_list'] = soil_nodes_combo
        k = 0
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
            time_soil_nodes = time.time()
            for i in range(0,simulation_timesteps):
                #print("day = ", i*dt)
                #time_openf = hn.print_time(start_time)
                hn.Manning_func(gph = H)    # calculate hydraulic radius and calculate flow rate
                h_new = hn.rainfall_nodes_func(gph = H, dt = dt, s = s, zr = soil_depth, soil_nodes = soil_nodes, rain_nodes = rain_nodes, depth = depth[i])

                # water_level.append(max(h_new.values()))
                # sl.loc[simulation_timesteps] = s
                # wl.loc[simulation_timesteps] = h_new
                #flood_nodes = flood_nodes + len([v for k, v in h_new.items() if (k >0 and v>= flood_level)])
                flood_nodes = flood_nodes + sum(h_new[k]>= flood_level for k in H.nodes if k != 0)
                flood_time = flood_time + (max(h_new.values()) >= flood_level)
                ## count how many nodes were above flood level!!!!!!!
                # edge_wl.loc[simulation_timesteps]=edge_h

            ## Properties and Performance of the network 
            #print("Run", k,"of", soil_nodes_combo_count, soil_nodes, "Network no.", network + 1, "|| node count", len(soil_nodes))
            #print("Time to run Manning & simulation: ")
            #hn.print_time(time_soil_nodes)
            degrees = dict(H.degree())
            mean_of_edges = sum(degrees.values())/len(degrees)
            flood_duration = dt*flood_time
            flood_duration_total = dt*flood_nodes/nodes_num
            outlet_water_level = H.nodes[0]['level']
            #soil_node_degree = hn.ignore_zero_div(sum(degrees.get(k,0) for k in soil_nodes),len(H.nodes))
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
            #output_df['outlet_max_list'].loc[k] = max(out_edge_wl)
            k += 1
        print("network: ", network + 1, "run time: ")
        hn.print_time(new_network_time)
        # main_df = pd.concat([main_df, output_df], ignore_index=True)

        f = open(datafile_name,'wb')
        pickle.dump(output_df, f)
        f.close()
    print("File name is: ", datafile_name, "File size: ", os.path.getsize(datafile_name), "Total time: ")
    hn.print_time(time_openf)

if __name__ == '__main__':
    main()