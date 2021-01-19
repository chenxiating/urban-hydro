import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import networkx as nx
import pickle
import itertools
import datetime as date
import datetime
import time
from random import sample
import os
import sys

## Functions
def create_networks(g_type = 'gn', nodes_num = 10, n = 0.01, diam = 1, changing_diam = True, diam_increment = 0.1, soil_depth = 0, 
slope = 0.008, elev_min = 90, elev_max = 100, level = 0.5, node_area = 500, outlet_level = None, outlet_node_area = None):
    elev_range = np.linspace(elev_min, elev_max, num=nodes_num)
    # if g_type == 'random tree': 
    #     gph = nx.scale_free_graph(nodes_num)
    # else: 
    #     gph = nx.gn_graph(nodes_num)
    gph = nx.gn_graph(nodes_num)
    nx.topological_sort(gph)
    if outlet_level is not None: 
        nx.set_node_attributes(gph, outlet_level, "level")
    if outlet_node_area is not None: 
        nx.set_node_attributes(gph, outlet_node_area, "node_area")
    max_path_length = max(len(nx.shortest_path(gph, source = k, target = 0)) for k in gph.nodes)
    for k in gph.nodes:
        elev = elev_range[k]
        a = dict(zip(["elev", "level", "node_area", "soil_depth"], [elev, level, node_area, soil_depth]))
        b = dict(zip([k], [a]))
        nx.set_node_attributes(gph, b)
    for k in gph.edges:
        #length = np.random.normal(l_mean, l_var)
        length = abs(k[0] - k[1])/slope
        downstream_degree_to_outlet = len(nx.shortest_path(gph, source = k[0], target = 0))
        diam0 = ((max_path_length - downstream_degree_to_outlet) * diam * diam_increment)*changing_diam + diam
        a = dict(zip(["n", "length", "diam"], [n, length, diam0]))
        b = dict(zip([k], [a]))
        nx.set_edge_attributes(gph, b)
    return gph

def fill_numbers(dictionary, full_list, number = 0):
    # for x in full_list:
    #     try: 
    #         dictionary[x] 
    #     except KeyError:
    #         dictionary[x] = number
    my_dict = dict.fromkeys(full_list, number)
    my_dict.update(dictionary)
    return my_dict

def draw_varying_size(gph, ax = None, attribute = 'storage', node_drawing_ratio = 20):
    node_sizes = []
    node_colors = []
    labels = {}
    edge_labels = {}
    pos = nx.spring_layout(gph)
    cnt = len(gph.nodes)
    #for n in nx.topological_sort(gph):
    for n in gph.nodes:
        x = gph.nodes[n].get(attribute)
        node_sizes.append(node_drawing_ratio*x)
        node_colors.append(n/cnt)
        labels[n] = str(n) + ": " + str(round(x, 1))
    
    for m in gph.edges: 
        edge_labels[m] = str(m) #+ ": " + str(round(gph.edges[m].get("length"),1))
    
    nx.draw(gph, pos, ax, node_color = node_colors, node_size = node_sizes, labels=labels, with_labels=True, cmap = plt.cm.rainbow)
    nx.draw_networkx_edge_labels(gph, pos, edge_labels=edge_labels)
    #node_colors = np.pad(node_colors[0:(len(node_colors)-1)], (1, 0), constant_values = 0)
    #node_colors_array = [plt.cm.rainbow(x) for x in node_colors]node_colors_array = [plt.cm.rainbow(x) for x in node_colors]
    rainbow = plt.cm.get_cmap('rainbow',cnt)
    node_colors_array = [rainbow(x) for x in node_colors]
    #print(dict(zip(["node", "node_colors", "node_colors_array"],[gph.nodes, node_colors, node_colors_array])))
    return node_colors_array

# def accumulate_downstream(gph, accum_attr='local_area', cumu_attr_name=None, split_attr='flow_split_frac'):
#     """
#     pass through the graph from upstream to downstream and accumulate the value
#     an attribute found in nodes and edges, and assign the accumulated value
#     as a new attribute in each node and edge.
#     Where there's a flow split, apply an optional split fraction to
#     coded in the upstream edge. (This is from Adam Erispaha's Sewergraph Package)
#     """
#     G1 = gph.copy()

#     if cumu_attr_name is None:
#         cumu_attr_name = 'cumulative_{}'.format(accum_attr)

#     for n in nx.topological_sort(G1):

#         # grab value in current node
#         attrib_val = G1.nodes[n].get(accum_attr, 0)

#         # sum with cumulative values in upstream nodes and edges
#         for p in G1.predecessors(n):
#             # add cumulative attribute val in upstream node, apply flow split fraction
#             attrib_val += G1.nodes[p][cumu_attr_name] * G1[p][n].get(split_attr, 1)

#             # add area routed directly to upstream edge/sewer
#             attrib_val += G1[p][n].get(accum_attr, 0)

#             # store cumulative value in upstream edge
#             G1[p][n][cumu_attr_name] = attrib_val

#         # store cumulative attribute value in current node
#         G1.nodes[n][cumu_attr_name] = attrib_val

#     return G1

def rainfall_func(size = 10 , freq= 0.1, meanDepth_inch = 10):
        # generate uniform and exponentially distributed random variables 
        meanDepth = meanDepth_inch/12
        depthExponential = np.random.exponential( meanDepth * np.ones(size) ) 
        freqUniform = np.random.random( size=size )
        depth = np.zeros(size)
        # the occurence of rainfall in any independent interval is lambda*dt 
        yesrain = freqUniform<np.tile(freq,size)*dt
        # rain falls according to prob within an increment 
        depth[yesrain] = depthExponential[yesrain]
        return depth

def soil_moisture_func(s, nporo = 0.45, zr = 10, emax = 0.05):
    #eta = emax/nporo/zr    # decay function: rho(s) = eta * s = emax/n zr * s
    #gamma_soil = (nporo*zr)/meanDepth 
    old_s = s
    ds = (depth[i] - emax * old_s * dt)/(zr * nporo) 
    new_s = old_s + ds
    if (new_s > 1):
        #print(depth[i])
        new_s = 1
    if (new_s < 0):
        new_s = 0
    s = new_s
    #print("Previous soil moisture:", old_s, "ds", ds, "New:", new_s)
    return s

def bioretention(gph, soil_nodes = [], s_old = 0.3, nporo = 0.45, zr = 10, emax = 0.05):
    #nodes = len(gph.nodes)
    #soil_nodes = np.random.choice(nodes, soil_nodes, replace=False)
    s = soil_moisture_func(s = s_old, nporo = nporo, zr = zr, emax = emax)
    #print("New soil moisture:", s)
    soil_array = np.ones(len(soil_nodes))*s
    soil_moisture = dict(zip(soil_nodes, soil_array))
    soil_moisture = fill_numbers(soil_moisture, gph.nodes)
    #print("Runoff reduction: ",soil_moisture)
    nx.set_node_attributes(gph, soil_moisture, 'soil_moisture')
    return s

# def attr_array_func(attr_name, gph, elem = 'edge', column = 1, ignore_outlet = False, ignore_attr = None):
#     if elem == 'node': 
#         dict0 = nx.get_node_attributes(gph,attr_name)
#     else:
#         dict0 = nx.get_edge_attributes(gph,attr_name)
#     if ignore_outlet == True:
#         last_elem = list(nx.topological_sort(gph))[-1]
#         dict0[last_elem]=ignore_attr[last_elem]

#     data0 = list(dict0.items())
#     data = np.array(data0, dtype = object)
#     array = data[:,column]
#     return array 

def Manning_func(gph, elev = 'elev', level = 'level', width = 'diam', n_name = 'n', l_name = 'length', shape = 'circular'):
    edge_list = []
    node_list = gph.nodes
    # node_area = attr_array_func('node_area', gph, elem = 'node')
    node_area_dict = nx.get_node_attributes(gph, 'node_area')
    node_area = np.array([node_area_dict[k] for k in gph.nodes])
    h_list = []
    h_radius_list = []
    dqdt0_list = []
    area_list = []
    for m in gph.edges:
        us_node = m[0]
        ds_node = m[1]
        h = 0.5*(gph.nodes[us_node].get(level) + gph.nodes[ds_node].get(level))
        elevdiff = gph.nodes[us_node].get(elev) + gph.nodes[us_node].get(level) - gph.nodes[ds_node].get(elev) - gph.nodes[ds_node].get(level)
        d = gph.edges[m].get(width)
        n = gph.edges[m].get(n_name)
        l = gph.edges[m].get(l_name)
        s = abs(elevdiff)/l
        if shape == 'circular':
            if h > d: 
                h = d
            if h < 0: 
                h = 0
                R = 0
                A = 0
            else: 
                theta = 2*np.arccos(1-2*abs(h)/d)
                #print("h", h, "d", d, "theta", theta, sep = "\t")
                R = 1/4*(1 - np.sin(theta)/theta)*d
                A = 1/8*(theta - np.sin(theta))*d**2
                #print("edge", m, "h", h, "d", d, sep = "\t")
                
        dqdt0 = np.sign(elevdiff)*1.49/n*A*R**(2/3)*s**(1/2) # Manning's Equation (Imperial Unit)
        if dqdt0 > gph.nodes[us_node].get("node_area")*gph.nodes[us_node].get(level):
            dqdt0 = gph.nodes[us_node].get("node_area")*gph.nodes[us_node].get(level)
            #print("edge", m, "R", R, "dqdt0", dqdt0)
        # if np.sign(elevdiff) < 0: 
        #     print("us node", us_node, "ds node", ds_node, "us flow", gph.nodes[us_node].get(elev) + gph.nodes[us_node].get(level), 
        #     "ds flow", gph.nodes[ds_node].get(elev) + gph.nodes[ds_node].get(level), "edge", m, "hydraulic radius", R, "area", A, 
        #     "slope", s, "edge_dqdt", dqdt0, sep = '\t')
        dqdt0_list.append(dqdt0)
        edge_list.append(m)
        h_list.append(h)
        h_radius_list.append(R)
        area_list.append(A)

    # # calculate ACAt
    Q = dqdt0_list*np.eye(len(dqdt0_list))
    A = (nx.incidence_matrix(gph, oriented=True)).toarray()
    edge_cnt = len(gph.edges)
    J = np.ones(edge_cnt)
    #print("A", A, "Q", Q, "J", J, 'node area', node_area, sep = "\t")
    #print("A@Q@J", A@Q@J, sep = "\t")
    
    dhdt_list = np.divide(A@Q@J,node_area)
   
    # record results
    dict_dqdt0 = dict(zip(edge_list, dqdt0_list))
    dict_h0 = dict(zip(edge_list, h_list))
    dict_rad = dict(zip(edge_list, h_radius_list))
    dict_h1 = dict(zip(node_list, dhdt_list))
    #print("edge water level: ", dict_h0, "dh at node", dict_h1)
    nx.set_edge_attributes(gph, dict_dqdt0, 'edge_dqdt0')
    nx.set_edge_attributes(gph, dict_h0, 'edge_h')
    nx.set_edge_attributes(gph, dict_rad, 'h_rad')
    nx.set_node_attributes(gph, dict_h1, 'dhdt')

    # # # translate to increase at nodes (CALCULATIONS CHECK)
    # dhdt1_list = []
    # node_list = []
    # for n in gph.nodes:
    #     inflow = 0
    #     outflow = 0
    #     n_area = gph.nodes[n].get('node_area')
    #     for i in gph.in_edges(n):
    #         #print("node", n, "in_edge", i, "edge_dqdt", gph.edges[i].get('edge_dqdt'), sep = " ")
    #         inflow = inflow + gph.edges[i].get('edge_dqdt')
    #     for j in gph.out_edges(n):
    #         #print("node", n, "out_edge", j, "edge_dqdt", gph.edges[j].get('edge_dqdt'), sep = " ")
    #         outflow = outflow + gph.edges[j].get('edge_dqdt')
    #     dhdt1 = (+ inflow - outflow)/n_area
    #     print("node", n, "height change", dhdt1, " ")
    #     node_list.append(n)
    #     dhdt1_list.append(dhdt1)

def rainfall_nodes_func(gph, s, nporo = 0.45, zr = 10, emax = 0.05): 
    #   This function calculates the runoff on nodes after a result of bioretention activities. 
    #   Runoff is the amount of rainfall that is unable to be absorbed by the soil, and needs to
    #   be received by the stormwater pipes. 
    bioretention(gph = gph, soil_nodes = soil_nodes, s_old = s)
    #bioretention(gph, soil_nodes = [])
    rain_nodes_count = len(rain_nodes)
    precip = fill_numbers(dict(zip(rain_nodes, np.ones(rain_nodes_count)*depth[i])), gph.nodes) # precipitation at the highest node, exponential distribution like in the project
    nx.set_node_attributes(gph, precip, 'precip')
    soil_depth = nx.get_node_attributes(gph,'soil_depth')
    soil_moisture = nx.get_node_attributes(gph, 'soil_moisture')
    runoff = {k: (precip[k]>(1-soil_moisture[k])*soil_depth[k])*(precip[k]-(1-soil_moisture[k])*soil_depth[k]) for k in precip} # check this part
    # nx.set_node_attributes(gph, runoff, 'runoff')
    # h0 = attr_array_func('level', gph = gph, elem = 'node', ignore_outlet = True, ignore_attr=outlet_level)
    h0 = nx.get_node_attributes(gph,'level')
    # runoff0 = attr_array_func('runoff', gph = gph, elem = 'node')
    # dh = attr_array_func('dhdt', gph = gph, elem = 'node') * dt
    dhdt = nx.get_node_attributes(gph, 'dhdt')
    #what happens if h0 + dh < 0?
    # h_new = dict(zip(gph.nodes, h0 + dh + runoff0))
    ##test here
    dhdt = nx.get_node_attributes(gph, 'dhdt')
    h_new = {k: (h0[k] + dhdt[k]*dt + runoff[k]) for k in gph.nodes}
    # if max(runoff0_test.values())>0:
    #     print("new node level: ", h_new)
    #     print("new node level_TEST: ", h_new_test)
    #     print(h0)
    #     print(h0_test)
    #print(h_new)
    nx.set_node_attributes(gph, h_new, 'level')  
    return h_new#, edge_h  

def random_sample_soil_nodes(range_min = 1, range_max = 20, range_count = 10):
    soil_nodes_combo_all = []
    # range_min = 1
    # range_max = 20
    # range_count = 10
    if range_max > nodes_num:
        range_max = nodes_num
    range_len = range_max - range_min + 1
    if range_count > range_len:
        range_count = range_len 
    combo_iter_list = np.linspace(range_min, range_max, num = range_count, dtype = int)
    for combo in combo_iter_list:
        soil_nodes_combo_to_add_full = (list(itertools.combinations(range(1, nodes_num), combo)))
        # soil_nodes_combo_to_add = sample(soil_nodes_combo_to_add_full, int(np.ceil(len(soil_nodes_combo_add_full)/nodes_num**(nodes_num+1))))
        soil_nodes_combo_to_add = sample(soil_nodes_combo_to_add_full, 1)
        soil_nodes_combo_all = soil_nodes_combo_all + soil_nodes_combo_to_add
        print("How many nodes? ", combo, "How many combos?", len(soil_nodes_combo_to_add))
    #soil_nodes_combo = sample(soil_nodes_combo_to_add, nodes_num*10)
    soil_nodes_combo = pd.Series(soil_nodes_combo_all, dtype = object)
    soil_nodes_combo_count = len(soil_nodes_combo)
    print("Soil nodes count:", soil_nodes_combo_count)
    return soil_nodes_combo, soil_nodes_combo_count

def ignore_zero_div(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0

def print_time(earlier_time):
    now_time = time.time()
    print("--- %s seconds ---" % round((time.time() - earlier_time),5))
    return now_time

## Assign Network Properties ##
# In this step we build the network based on different criteria
np.random.seed(seed = 1358)
#G = pickle.load(open('graph_10nodes', 'rb'))
outlet_level = {0: 0.2}                  # set outlet river water level to be constant
outlet_node_area = {0: 10000}           # set the river area to be 10 times 
soil_depth = 6
init_level = 0.05
flood_level = 1.5
s = 0.1                               # initial soil moisture
nodes_num = int(25)
main_df = pd.DataFrame()

G = create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_area = 500, 
outlet_level = outlet_level, outlet_node_area = outlet_node_area)
filename = 'graph_' + str(nodes_num) + 'nodes'
outfile = open(filename,'wb')
pickle.dump(G,outfile)
outfile.close()

rain_nodes = G.nodes

fig_init, ax_init = plt.subplots(1,1)
node_colors1 = draw_varying_size(G, ax = ax_init, attribute = 'elev', node_drawing_ratio = 1)
fig_init.suptitle('Initial Set-up')

## Precipitation
# Rainfall generation. Units will be presented in foot. 
dt = 0.1
days = 50
n = round(days/dt)
#n = 10
#npad = round(n/2)
meanDepth_inch = 1
depth = rainfall_func(size=n,freq=0.1,meanDepth_inch=meanDepth_inch)
#depth = np.pad([1], (npad, n - npad - 1), 'constant', constant_values = (0))
timesteps = np.linspace(0, n*dt, num = n)
precip0 = [0]* len(G.nodes)

today = date.datetime.today()
dt_str = today.strftime("%Y%m%d-%H%M")
# path="/Users/xchen/Documents/UMN_PhD/urban_stormwater_analysis/figures/ten-node-model/"+dt_str+"/"
# if not os.path.exists(path):
#     os.makedirs(path)
datafile_name = 'dataset_'+str(meanDepth_inch)+'-inch_'+str(nodes_num)+'-nodes_'+str(days)+'-day_'+dt_str+'.pickle'
time_openf = time.time()
file_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_directory)
output_file_name = "output_" + dt_str + ".txt"
terminal_output = open(output_file_name, 'w')
sys.stdout = terminal_output

print("Output File:", output_file_name)
print("==========")
print("==========")
# Simulations

for network in range(10):
    new_network_time = time.time()
    G = create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_area = 500, 
    outlet_level = outlet_level, outlet_node_area = outlet_node_area)
    soil_nodes_combo, soil_nodes_combo_count = random_sample_soil_nodes(range_min = 0, range_max = 50, range_count = 50)
    # fig_init, ax_init = plt.subplots(1,1)
    # node_colors1 = draw_varying_size(G, ax = ax_init, attribute = 'elev', node_drawing_ratio = 1)
    # fig_init.suptitle('Initial Set-up')
    # plt.show()
    # mapname = 'graph_' + str(nodes_num) + 'nodes_'+ str(network) + '_' + dt_str
    # mapfile = open(mapname,'wb')
    # pickle.dump(G,mapfile)
    # mapfile.close()

    output_columns =['soil_nodes_list', "flood_duration_list", "flood_duration_total_list", 'outlet_water_level', 
    "soil_node_degree_list", "soil_node_elev_list"]
    output_df = pd.DataFrame(np.nan, index=range(soil_nodes_combo_count), columns=output_columns)
    #print("count of combo", len(soil_nodes_combo))
    #print(output_df.loc[:,'soil_nodes_list'])
    output_df.loc[:,'soil_nodes_list'] = soil_nodes_combo
    k = 0
    for soil_nodes in output_df['soil_nodes_list']:
        H = G.copy()
        soil_nodes_length = len(soil_nodes)
        soil_nodes_depth = dict(zip(soil_nodes, np.ones(soil_nodes_length)*soil_depth))
        nx.set_node_attributes(H, soil_nodes_depth, "soil_depth")
        # sl = pd.DataFrame(np.nan, index=range(0,n+1), columns=G.nodes)
        # water_level = []
        # wl = pd.DataFrame(np.nan, index=range(0,n+1), columns=G.nodes)
        # edge_wl = pd.DataFrame(np.nan, index=range(0,n+1), columns=G.edges)
        flood_nodes = 0
        flood_time = 0
        time_soil_nodes = time.time()
        for i in range(0,n):
            #print("day = ", i*dt)
            #time_openf = print_time(start_time)
            Manning_func(gph = H)    # calculate hydraulic radius and calculate flow rate
            #
            h_new = rainfall_nodes_func(gph = H, s = s, zr = soil_depth)

            # water_level.append(max(h_new.values()))
            # sl.loc[n] = s
            # wl.loc[n] = h_new
            #flood_nodes = flood_nodes + len([v for k, v in h_new.items() if (k >0 and v>= flood_level)])
            flood_nodes = flood_nodes + sum(h_new[k]>= flood_level for k in H.nodes if k != 0)
            flood_time = flood_time + (max(h_new.values()) >= flood_level)
            ## count how many nodes were above flood level!!!!!!!
            # edge_wl.loc[n]=edge_h

        ## Properties and Performance of the network 
        #print("Run", k,"of", soil_nodes_combo_count, soil_nodes, "Network no.", network + 1, "|| node count", len(soil_nodes))
        #print("Time to run Manning & simulation: ")
        #print_time(time_soil_nodes)
        degrees = dict(H.degree())
        mean_of_edges = sum(degrees.values())/len(degrees)
        flood_duration = dt*flood_time
        flood_duration_total = dt*flood_nodes/nodes_num
        outlet_water_level = H.nodes[0]['level']
        #soil_node_degree = ignore_zero_div(sum(degrees.get(k,0) for k in soil_nodes),len(H.nodes))
        soil_node_degree = ignore_zero_div(sum(degrees.get(k,0) for k in soil_nodes),soil_nodes_length)
        # soil_node_elev = ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1
        # for k in soil_nodes),len(H.nodes))
        soil_node_elev = ignore_zero_div(sum(len(nx.shortest_path(H, source=k, target = 0)) - 1 
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
        #output_df['outlet_max_list'].loc[k] = max(out_edge_wl)
        k += 1
    print("network: ", network + 1, "run time: ")
    print_time(new_network_time)
    main_df = pd.concat([main_df, output_df], ignore_index=True)
f = open(datafile_name,'wb')
pickle.dump(main_df, f)
f.close()
print("File name is: ", datafile_name, "File size: ", os.path.getsize(datafile_name), "Total time: ")
print_time(time_openf)

terminal_output.close() 

# ## Plots
# # Plot 1: 
# fig_dots, ax_dots = plt.subplots(1,1)
# #plt.subplot(211)
# cmap0 = plt.cm.Reds
# c = flood_duration_list
# s = [np.exp(len(k)) for k in soil_nodes_list]
# dots = plt.scatter(soil_node_degree_list, soil_node_elev_list, c = c, s = s,
# cmap = cmap0)
# fig_dots.subplots_adjust(right=0.8)
# cbar_ax = fig_dots.add_axes([0.85, 0.5, 0.05, 0.35])
# cbar = fig_dots.colorbar(dots, cax=cbar_ax)
# cbar.set_label('Flood Duration (Days)')
# ax_dots.set_ylabel("Topography Index")
# ax_dots.set_xlabel("Neighbor Index")
# #ax_dots.set_ylim(bottom = -0.1, top = 0.5)
# kw = dict(prop="sizes", num=range_max-range_min - 1, color=dots.cmap(0.5), fmt="{x:.0f}",
#           func=lambda s: np.log(s))
# legend = cbar_ax.legend(*dots.legend_elements(**kw),
#                     bbox_to_anchor=(1, -0.1), loc='upper center', borderaxespad=0., title="# of Nodes")
# legend.get_frame().set_edgecolor('k')
# plt.savefig(path + "flood_"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')

# # Plot 2
# fig, [ax_TI, ax_NI] = plt.subplots(2,1)
# #gs = gridspec.GridSpec(2, 1, width_ratios=[4, 3]) 
# #ax_TI = plt.subplot(211)
# cmap = plt.cm.Reds
# c = flood_duration_list
# s = [np.exp(len(k)) for k in soil_nodes_list]
# TI = ax_TI.scatter(soil_node_elev_list, flood_duration_list, c = c, s = s,
# cmap = cmap)
# ax_TI.xaxis.tick_top()
# ax_TI.set_xlabel("Topography Index")
# ax_TI.xaxis.set_label_position('top') 
# ax_TI.set_ylabel("Flood Duration (Days)")
# NI = ax_NI.scatter(soil_node_degree_list, flood_duration_list, c = c, s = s, 
# cmap = cmap)
# # cbar = fig.colorbar(NI, ax=ax_NI)
# # cbar.set_label("Number of GI Nodes")
# ax_NI.set_xlabel("Neighbor Index")
# ax_NI.set_ylabel("Flood Duration (Days)")
# fig.subplots_adjust(top = 0.8, right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.5, 0.05, 0.35])
# cbar = fig.colorbar(TI, cax=cbar_ax)
# cbar.set_label('Flood Duration (Days)')
# #ax_NI.set_ylim(bottom = -0.1, top = 1.6)
# kw = dict(prop="sizes", num=range_max-range_min - 1, color=NI.cmap(0.5), fmt="{x:.0f}",
#           func=lambda s: np.log(s))
# legend = ax_NI.legend(*NI.legend_elements(**kw),
#                     bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="# of Nodes")
# legend.get_frame().set_edgecolor('k')
# plt.savefig(path + "flood_TI_and_NI-"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')

# ## new plots for outlet max
# fig_dots, ax_dots = plt.subplots(1,1)
# #plt.subplot(211)
# cmap0 = plt.cm.Blues
# c = outlet_max_list
# s = [np.exp(len(k)) for k in soil_nodes_list]
# dots = plt.scatter(soil_node_degree_list, soil_node_elev_list, c = c, s = s,
# cmap = cmap0, alpha = 0.7, edgecolor = 'k')
# fig_dots.subplots_adjust(right=0.8)
# cbar_ax = fig_dots.add_axes([0.85, 0.5, 0.05, 0.35])
# cbar = fig_dots.colorbar(dots, cax=cbar_ax)
# cbar.set_label('Highest Flow at Outlet (ft)')
# ax_dots.set_ylabel("Topography Index")
# ax_dots.set_xlabel("Neighbor Index")
# #ax_dots.set_ylim(bottom = -0.1, top = 0.5)
# kw = dict(prop="sizes", num=4, color=dots.cmap(0.5), fmt="{x:.0f}",
#           func=lambda s: np.log(s))
# legend = cbar_ax.legend(*dots.legend_elements(**kw),
#                     bbox_to_anchor=(1, -0.1), loc='upper center', borderaxespad=0., title="# of Nodes")
# legend.get_frame().set_edgecolor('k')
# plt.savefig(path + "outletflow_"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')


# fig, [ax_TI, ax_NI] = plt.subplots(2,1)
# #gs = gridspec.GridSpec(2, 1, width_ratios=[4, 3]) 
# #ax_TI = plt.subplot(211)
# cmap = plt.cm.Blues
# c = outlet_max_list
# s = [np.exp(len(k)) for k in soil_nodes_list]
# TI = ax_TI.scatter(soil_node_elev_list, outlet_max_list, c = c, s = s,
# cmap = cmap, alpha = 0.7, edgecolor = 'k')
# ax_TI.xaxis.tick_top()
# ax_TI.set_xlabel("Topography Index")
# ax_TI.xaxis.set_label_position('top') 
# ax_TI.set_ylabel("Highest Flow at Outlet (ft)")
# NI = ax_NI.scatter(soil_node_degree_list, outlet_max_list, c = c, s = s, 
# cmap = cmap, alpha = 0.7, edgecolor = 'k')
# # cbar = fig.colorbar(NI, ax=ax_NI)
# # cbar.set_label("Number of GI Nodes")
# ax_NI.set_xlabel("Neighbor Index")
# #ax_NI.set_ylabel("Highest Flow at Outlet (ft)")
# fig.subplots_adjust(top = 0.8, right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.5, 0.05, 0.35])
# cbar = fig.colorbar(TI, cax=cbar_ax)
# cbar.set_label('Highest Flow at Outlet (ft)')
# #ax_NI.set_ylim(bottom = -0.1, top = 1.6)
# kw = dict(prop="sizes", num=4, color=NI.cmap(0.5), fmt="{x:.0f}",
#           func=lambda s: np.log(s))
# legend = ax_NI.legend(*NI.legend_elements(**kw),
#                     bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="# of Nodes")
# legend.get_frame().set_edgecolor('k')
# plt.savefig(path + "outletflow_TI_and_NI-"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')


# # Plot 3
# fig_last, ax_last = plt.subplots(1,1)
# fig_last.suptitle('Final Set-up')
# draw_varying_size(H, ax = ax_last, attribute = 'level', node_drawing_ratio = 100)
# plt.savefig(path + "NetworkOutput-"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')

# # fig_wl, ax_wl = plt.subplots(1,1)
# # ax_p = ax_wl.twinx()
# # ax_p.plot(timesteps, depth, linewidth = 0.5, alpha = 0.8)
# # ax_p.set_ylabel('Precipitation (ft)')
# # #upstream = [n for n in nx.traversal.bfs_tree(H, 0, reverse=True) if n != 0]
# # for i in range(1,len(H.nodes)):
# #     label = "Node " + str(i)
# #     #print("node ", i, "node_colors ", node_colors1[i])
# #     ax_wl.plot(timesteps, wl[i], color = node_colors1[i], label = label)
# # ax_wl.set_xlabel('Days')
# # ax_wl.set_ylabel('Water Level (ft)')
# # ax_wl.legend()
# # plt.savefig(path + "NodeWL-"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')

# # fig_ewl, ax_ewl = plt.subplots(1,1)
# # ax_p = ax_ewl.twinx()
# # ax_p.plot(timesteps, depth, linewidth = 0.5, alpha = 0.8)
# # ax_p.set_ylabel('Precipitation (ft)')
# # out_edges = H.in_edges(0, data = False)
# # out_edge_wl = [0]
# # for i in out_edges:
# #     out_edge_wl = out_edge_wl + edge_wl[i]
# #     #print("out_edge_wl",out_edge_wl)
# # ax_ewl.plot(timesteps, out_edge_wl, label = 'Outlet')
# # for i in H.edges:
# #     label = "Edge " + str(i)
# #     us_node = i[0]
# #     ax_ewl.plot(timesteps, edge_wl[i], color = node_colors1[us_node], label = label)
# # ax_ewl.set_xlabel('Days')
# # ax_ewl.set_ylabel('Water Level (ft)')
# # ax_ewl.legend()
# # plt.savefig(path + "EdgeWL-"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')

# # fig_hgl, ax_hgl = plt.subplots(1,1)
# # ax_p = ax_hgl.twinx()
# # ax_p.plot(timesteps, depth, linewidth = 0.5, alpha = 0.8)
# # ax_p.set_ylabel('Precipitation (ft)')
# # #upstream = [n for n in nx.traversal.bfs_tree(H, 0, reverse=True) if n != 0]
# # for i in range(1,len(H.nodes)):
# #     label = "Node" + str(i)
# #     ax_hgl.plot(timesteps, wl[i] + H.nodes[i].get('elev'), color = node_colors1[i], label = label)
# # ax_hgl.set_xlabel('Days')
# # ax_hgl.set_ylabel('Water Level + Elevation (ft)')
# # ax_hgl.legend()
# # plt.savefig(path + "NodeElevWL-"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')

# fig_sl, ax_sl = plt.subplots(1,1)
# ax_p = ax_sl.twinx()
# ax_p.plot(timesteps, depth, linewidth = 0.5, alpha = 0.8)
# ax_p.set_ylabel('Precipitation (ft)')
# ax_sl.plot(timesteps, sl, color = node_colors1[i], label = label)
# ax_sl.set_xlabel('Days')
# ax_sl.set_ylabel('Soil Moisture Level')
# plt.savefig(path + "SoilMoisture-"+ str(meanDepth_inch) +'-inch' + dt_str + '.png')
# os.system("python3 ./urban-hydro/test_graph.py " + datafile_name)
#plt.show()
