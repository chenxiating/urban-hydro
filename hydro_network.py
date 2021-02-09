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
from math import comb
from math import log10
import statistics
import os
import sys

## Functions
def create_networks(g_type = 'gn', nodes_num = 10, n = 0.01, diam = 1, changing_diam = True, diam_increment = 0.1, soil_depth = 0, 
slope = 0.008, elev_min = 90, elev_max = 100, level = 0.5, node_area = 500, conductivity = 0.5,
outlet_elev = None, outlet_level = None, outlet_node_area = None, seed = None, kernel = None):
    """
    create a random network with different properties. the slope has been defaulted to be the same in
    the entire network, and the diameter is defaulted to go up as the network is further away from the
    outlet.
    conductivity shoudl somehow link to the porosity of soil. 
    """
    elev_range = np.linspace(elev_min, elev_max, num=nodes_num)
    gph = nx.gn_graph(nodes_num, seed = seed, kernel = kernel)
    nx.topological_sort(gph)
    max_path_order = max(len(nx.shortest_path(gph, source = k, target = 0)) for k in gph.nodes)
    for k in gph.nodes:
        elev = elev_range[k]
        a = dict(zip(["elev", "level", "node_area", "soil_depth"], [elev, level, node_area, soil_depth]))
        b = dict(zip([k], [a]))
        nx.set_node_attributes(gph, b)
    for k in gph.edges:
        length = abs(k[0] - k[1])/slope
        downstream_degree_to_outlet = len(nx.shortest_path(gph, source = k[0], target = 0))
        diam0 = ((max_path_order - downstream_degree_to_outlet) * diam * diam_increment)*changing_diam + diam
        a = dict(zip(["n", "length", "diam",'conductivity'], [n, length, diam0, conductivity]))
        b = dict(zip([k], [a]))
        nx.set_edge_attributes(gph, b)
    if outlet_level is not None: 
        nx.set_node_attributes(gph, outlet_level, "level")
    if outlet_node_area is not None: 
        nx.set_node_attributes(gph, outlet_node_area, "node_area")
    if outlet_elev is None: 
        try: 
            outlet_elev = elev_min - outlet_level
            nx.set_node_attributes(gph, outlet_elev, 'elev')
        except TypeError: 
            pass
    return gph

def fill_numbers(dictionary, full_list, number = 0):
    """
    a function to pad a dictionary with one number. this is used in the soil moisture calculation, 
    where non-water-retentive nodes automatically get s = 1 (saturated). 
    """
    my_dict = dict.fromkeys(full_list, number)
    my_dict.update(dictionary)
    return my_dict

def draw_varying_size(gph, ax = None, attribute = 'storage', edge_attribute = None, node_drawing_ratio = 20):
    """
    a function to draw the network. not being used.
    """
    node_sizes = []
    node_colors = []
    labels = {}
    edge_labels = {}
    pos = nx.spring_layout(gph)
    cnt = len(gph.nodes)
    for n in nx.topological_sort(gph):
        x = gph.nodes[n].get(attribute)
        node_sizes.append(node_drawing_ratio*x)
        node_colors.append(n/cnt)
        labels[n] = str(n) + ": " + str(round(x, 1))
    
    for m in gph.edges: 
        if edge_attribute is None:
            edge_labels[m] = str(m)
        else:
            edge_labels[m] = gph.edges[m].get(edge_attribute)
    
    nx.draw(gph, pos, ax, node_color = node_colors, node_size = node_sizes, labels=labels, with_labels=True, cmap = plt.cm.rainbow)
    nx.draw_networkx_edge_labels(gph, pos, edge_labels=edge_labels)
    #node_colors = np.pad(node_colors[0:(len(node_colors)-1)], (1, 0), constant_values = 0)
    #node_colors_array = [plt.cm.rainbow(x) for x in node_colors]node_colors_array = [plt.cm.rainbow(x) for x in node_colors]
    rainbow = plt.cm.get_cmap('rainbow',cnt)
    node_colors_array = [rainbow(x) for x in node_colors]
    #print(dict(zip(["node", "node_colors", "node_colors_array"],[gph.nodes, node_colors, node_colors_array])))
    return node_colors_array

def draw_network_timestamp(gph, ax = None, edge_attribute = 'edge_velocity', soil_nodes = None):
    """
    draw the network flow and the dispersion coefficients at a single timestep. 
    """
    pos = nx.spring_layout(gph)
    edge_color = [gph.edges[m].get(edge_attribute) for m in gph.edges]
    node_color = []
    for node in gph: 
        if node in set(soil_nodes):
            node_color.append('C2')
        else:
            node_color.append('C0')
    node_label = {n: str(n) + ':' + str(round(gph.nodes[n].get('level') + gph.nodes[n].get('elev') ,1)) for n in gph.nodes}
    edge_label = {m: str(round(gph.edges[m].get(edge_attribute), 2)) for m in gph.edges}
    cmap = plt.cm.Greys
    fig0, ax0 = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]})
    nx.draw(gph, pos, ax0[1], node_color = node_color, edge_color = edge_color, labels = node_label, with_labels = True, edge_cmap = cmap)
    nx.draw_networkx_edge_labels(gph, pos, edge_labels=edge_label)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = 0, vmax=max(edge_color)))
    sm._A = []
    #fig = plt.gcf()
    #cbar_ax = fig0.add_axes([0.1, 0.25, 0.05, 0.35])
    plt.colorbar(sm, cax = ax0[0])
    ax0[0].set_xlabel('Edge dQ')

def accumulate_downstream(gph, accum_attr='node_area', cumu_attr_name=None, soil_nodes=None):
    """
    pass through the graph from upstream to downstream and accumulate the value
    an attribute found in nodes and edges, and assign the accumulated value
    as a new attribute in each node and edge.
    Where there's a flow split, apply an optional split fraction to
    coded in the upstream edge. (This is from Adam Erispaha's Sewergraph Package)
    believe this isnt' being used.
    """

    if cumu_attr_name is None:
        cumu_attr_name = 'cumulative_{}'.format(accum_attr)

    for topo_node in nx.topological_sort(gph):
        # grab value in current node
        attrib_val = gph.nodes[topo_node].get(accum_attr, 0)

        # sum with cumulative values in upstream nodes and edges
        for p in gph.predecessors(topo_node):
            # add cumulative attribute val in upstream node, apply flow split fraction
            attrib_val += gph.nodes[p][cumu_attr_name]

            # store cumulative value in upstream edge
            gph[p][topo_node][cumu_attr_name] = attrib_val

            # add area routed directly to upstream edge/sewer
            attrib_val += gph[p][topo_node].get(accum_attr, 0)
        # store cumulative attribute value in current node
        gph.nodes[topo_node][cumu_attr_name] = attrib_val

    if soil_nodes is not None: 
        sum_cumu_attr_downstream = sum(gph.nodes[k][cumu_attr_name] for k in soil_nodes)

    else: 
        sum_cumu_attr_downstream = sum(gph.nodes[k][cumu_attr_name] for k in gph.nodes)
    return sum_cumu_attr_downstream

def rainfall_func(size = 10, freq= 0.1, meanDepth_inch = 2, dt = 0.1, is_pulse = False):
    """
    the function to generate stochastic rainfalls or pulse rain event.
    """
    # mean depth in foot
    meanDepth = meanDepth_inch/12   
    depth = np.zeros(size)
    # generate one single pulse rainfall event
    if is_pulse: 
        depth[10] = meanDepth
    
    else: 
        # generate uniform and exponentially distributed random variables 
        depthExponential = np.random.exponential( meanDepth * np.ones(size) ) 
        freqUniform = np.random.random( size=size )
        # the occurence of rainfall in any independent interval is lambda*dt 
        yesrain = freqUniform<np.tile(freq,size)*dt
        # rain falls according to prob within an increment 
        depth[yesrain] = depthExponential[yesrain]
    return depth

def soil_moisture_func(s, depth, dt, nporo = 0.45, zr = 10, emax = 0.05):
    """
    this function calculated soil moisture decays overtime on one node. this is assuming a linear decay.
    eta = emax/nporo/zr    # decay function: rho(s) = eta * s = emax/n zr * s
    gamma_soil = (nporo*zr)/meanDepth 
    """
    old_s = s
    ds = (depth - emax * old_s * dt)/(zr * nporo) 
    new_s = old_s + ds
    if (new_s > 1):
        new_s = 1
    if (new_s < 0):
        new_s = 0
    s = new_s
    return s

def bioretention(gph, depth, dt, soil_nodes = [], old_s = 0.3, nporo = 0.45, zr = 10, emax = 0.05):
    """
    calls soil_moisture_func to calculate soil moisture changes on all water retentive nodes.
    """
    s = soil_moisture_func(s = old_s, dt = dt, depth = depth, nporo = nporo, zr = zr, emax = emax)
    soil_array = np.ones(len(soil_nodes))*s
    soil_moisture = dict(zip(soil_nodes, soil_array))
    soil_moisture = fill_numbers(soil_moisture, gph.nodes)
    nx.set_node_attributes(gph, soil_moisture, 'soil_moisture')
    return s

def outlet_evaporation_func(gph, dt, outlet_node = 0, evap_rate = 0.01, level_name = 'level'):
    """
    This accounts for water evaporation at outlet. The default value is 0.01 ft per day, 
    which is nearly 3 mm per day. This is a linear function. 
    """
    new_level = gph.nodes[outlet_node].get(level_name) - evap_rate*dt
    gph.nodes[outlet_node][level_name] = new_level

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
    """
    Manning's equation to calculate open channel flow in pipes.
    """
    # initialize
    edge_list = []
    node_list = gph.nodes
    node_area_dict = nx.get_node_attributes(gph, 'node_area')
    node_area = np.array([node_area_dict[k] for k in gph.nodes])
    h_list = []
    t_list = []
    u_list = []
    h_radius_list = []
    dq_list = []
    area_list = []
    nx.topological_sort(gph)

    # calculate for each edge
    for m in gph.edges:
        us_node = m[0]
        ds_node = m[1]
        # h = 0.5*(gph.nodes[us_node].get(level) + gph.nodes[ds_node].get(level))
        h = gph.nodes[ds_node].get(level)
        elevdiff = gph.nodes[us_node].get(elev) + gph.nodes[us_node].get(level) - gph.nodes[ds_node].get(elev) - gph.nodes[ds_node].get(level)
        d = gph.edges[m].get(width)
        n = gph.edges[m].get(n_name)
        l = gph.edges[m].get(l_name)
        s = abs(elevdiff)/l
        if shape == 'circular':
            if h > d: 
                h = d
            if h <= 0: 
                h = 0
                R = 0
                A = 0
            else: 
                theta = 2*np.arccos(1-2*abs(h)/d)
                #print("h", h, "d", d, "theta", theta, sep = "\t")
               # R = 1/4*(1 - ignore_zero_div(np.sin(theta),theta))*d
                R = 1/4*(1 - (np.sin(theta)/theta))*d
                A = 1/8*(theta - np.sin(theta))*d**2
                #print("edge", m, "h", h, "d", d, sep = "\t")
        u = 1.49/n*R**(2/3)*s**(1/2)
        dq = np.sign(elevdiff)*u*A # Manning's Equation (Imperial Unit) for edges
        if dq >= gph.nodes[us_node].get("node_area")*gph.nodes[us_node].get(level) or gph.nodes[us_node].get(elev) > (gph.nodes[ds_node].get(elev) + gph.nodes[ds_node].get(level)):
            dq = gph.nodes[us_node].get("node_area")*gph.nodes[us_node].get(level)
            u = abs(ignore_zero_div(dq, A))
            # print(m, 'dq has been capped.', 'dq', dq)
        if dq < 1e-4:
            dq = 0
            u = 0
        t = ignore_zero_div(l,u)
        # if np.sign(elevdiff) < 0: 
        #     print("us node", us_node, "ds node", ds_node, "us flow", gph.nodes[us_node].get(elev) + gph.nodes[us_node].get(level), 
        #     "ds flow", gph.nodes[ds_node].get(elev) + gph.nodes[ds_node].get(level), "edge", m, "hydraulic radius", R, "area", A, 
        #     "slope", s, "edge_dqdt", dq, sep = '\t')
        dq_list.append(dq)
        edge_list.append(m)
        h_list.append(h)
        h_radius_list.append(R)
        t_list.append(t)
        u_list.append(u)
        area_list.append(A)

    # # calculate ACAt
    Q = dq_list*np.eye(len(dq_list))
    A = (nx.incidence_matrix(gph, oriented=True)).toarray()
    edge_cnt = len(gph.edges)
    J = np.ones(edge_cnt)
    dhdt_list = np.divide(A@Q@J,node_area) # this is at the nodes
    
    # record results
    dict_dq = dict(zip(edge_list, dq_list))         # flow through edge
    dict_h0 = dict(zip(edge_list, h_list))          # water level 
    dict_rad = dict(zip(edge_list, h_radius_list))
    dict_t = dict(zip(edge_list, t_list))
    dict_u = dict(zip(edge_list, u_list))
    dict_dhdt = dict(zip(node_list, dhdt_list))

    # assign results in map
    nx.set_edge_attributes(gph, dict_dq, 'edge_dq')
    nx.set_edge_attributes(gph, dict_h0, 'edge_h')
    nx.set_edge_attributes(gph, dict_rad, 'h_rad')
    nx.set_edge_attributes(gph, dict_t, 'edge_time')
    nx.set_edge_attributes(gph, dict_u, 'edge_velocity')
    nx.set_node_attributes(gph, dict_dhdt, 'dhdt')

    plt.show()
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
    calculate_flow_path(gph = gph)
    calculate_flow_path(gph = gph, accum_attr='edge_time')

def Darcy_func(gph, elev = 'elev', level = 'level', width = 'diam', l_name = 'length'):
    """
    Darcy's equation to calculate subsurface flow. Need to figure out how to calculate travel time within
    this Darcy framework. 
    """
    # initialize
    nx.topological_sort(gph)
    A = (nx.incidence_matrix(gph, oriented=True)).toarray()
    At = np.transpose(A)
    C0 = np.array([gph.edges[edge]['conductivity']/gph.edges[edge][l_name] for edge in gph.edges])
    C = C0 * np.eye(np.size(C0))
    
    # calculate water table changes
    h_current = np.array([gph.nodes[node].get(elev) + gph.nodes[node].get(level) for node in gph.nodes])
    dhdt_list = A@C@At@h_current
    dict_dhdt = dict(zip(gph.nodes, dhdt_list))

    #### how to calculate flow velocity within each edge in a subsurface network?
    #### need to account for porosity (nporo) in the soil. dhdt is water level changes
    #### at a node. I think we can utilize the incidence matrix to get back from node to edge. 

    # assign results in map
    # nx.set_edge_attributes(gph, dict_t, 'edge_time')
    # nx.set_edge_attributes(gph, dict_u, 'edge_velocity')
    nx.set_node_attributes(gph, dict_dhdt, 'dhdt')

    # calculate_flow_path(gph = gph)
    # calculate_flow_path(gph = gph, accum_attr='edge_time')

def calculate_flow_path(gph, accum_attr='length', path_attr_name=None):
    """
    calculate flow paths lengths and travel time
    """
    path_dict = {}
    if path_attr_name is None:
        path_attr_name = 'path_{}'.format(accum_attr)
    for node in gph.nodes:
        shortest_path_set = list(nx.shortest_path(gph, source = node, target = 0))
        path_attr = 0
        i = 0
        keep_running = True
        while (keep_running):
            for i in shortest_path_set:
                out_edge_set = gph.edges(i)
                if not out_edge_set: 
                    break
                for out_edge in out_edge_set:
                    if gph.edges[out_edge].get('edge_velocity') > 0: 
                        path_attr += gph.edges[out_edge].get(accum_attr)
                        # print('succeed:', node, 'i_node', i, out_edge, gph.edges[out_edge].get('edge_velocity'), gph.edges[out_edge].get(accum_attr), path_attr)
                    else: 
                        # print('fail:', node, 'i_node', i, out_edge, gph.edges[out_edge].get('edge_velocity'), gph.edges[out_edge].get(accum_attr), path_attr)
                        path_attr = 0
                        keep_running = False
                        break
            keep_running = False
        path_dict[node] = path_attr
    nx.set_node_attributes(gph, path_dict, path_attr_name)
    return path_dict

def dispersion_func(gph, l_name = 'length', t_name = 'edge_time'):
    """
    calculate dispersion coefficients. this doesn't work with Darcy yet.
    """
    path_length = 'path_{}'.format(l_name)
    path_time = 'path_{}'.format(t_name)

    network_path_length_dict = nx.get_node_attributes(gph, path_length)
    network_path_length_set = [network_path_length_dict[k] for k in network_path_length_dict]
    network_path_time_dict = nx.get_node_attributes(gph, path_time)
    network_path_time_set = [network_path_time_dict[k] for k in network_path_time_dict]
    print('network_path_length_dict', network_path_length_dict)
    # print('network_path_dq_dict', nx.get_edge_attributes(gph, 'edge_dq'))
    # print('network_path_time_dict', network_path_time_dict)

    try: 
        mean_path_length = statistics.mean(network_path_length_set)
        mean_path_time = statistics.mean(network_path_time_set)
        var_path_length = statistics.variance(network_path_length_set)
        disp_g = 0.5*ignore_zero_div(var_path_length,mean_path_time)
        #print(disp_g)
        network_celerity = ignore_zero_div(mean_path_length,mean_path_time)
        print('network_celerity', network_celerity)
        flowpath_celerity_dict = {k: ignore_zero_div(network_path_length_dict[k],network_path_time_dict[k]) for k in network_path_time_dict}
        print('flowpath_celerity_dict', flowpath_celerity_dict)
        stretched_path_length_set = [ignore_zero_div(network_path_length_dict[k],flowpath_celerity_dict[k])*network_celerity for k in network_path_length_dict]
        print('stretched_path_length_set', stretched_path_length_set)
        disp_kg = 0.5*ignore_zero_div(statistics.variance(stretched_path_length_set),mean_path_length)*network_celerity
    except statistics.StatisticsError:
        var_path_length = 0
        disp_g = 0
        disp_kg = 0

    return var_path_length, disp_g, disp_kg


def rainfall_nodes_func(gph, s, dt, depth, soil_nodes, rain_nodes, nporo = 0.45, zr = 10, emax = 0.05): 
    """  
    This function calculates the runoff on nodes after a result of bioretention activities. 
    Runoff is the amount of rainfall that is unable to be absorbed by the soil, and needs to
    be received by the stormwater pipes. 
    """
    soil_moisture_new = bioretention(gph = gph, dt = dt, depth = depth, soil_nodes = soil_nodes, old_s = s)
    #bioretention(gph, soil_nodes = [])
    rain_nodes_count = len(rain_nodes)
    precip = fill_numbers(dict(zip(rain_nodes, np.ones(rain_nodes_count)*depth)), gph.nodes) # precipitation at the highest node, exponential distribution like in the project
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
    outlet_evaporation_func(gph, dt, evap_rate = 0.01)
    return h_new, soil_moisture_new

def random_sample_soil_nodes(nodes_num, count_to_sample = None, range_min = 1, range_max = 20, range_count = 10 ):
    """
    randomly generate water retentive nodes in the network
    """
    if range_max >= nodes_num:
        range_max = nodes_num - 1
    if range_min >= nodes_num:
        range_min = nodes_num - 1
    range_len = range_max - range_min + 1
    if range_count > range_len:
        range_count = range_len 

    soil_nodes_combo_all = []
    combo_iter_list = np.linspace(range_min, range_max, num = range_count, dtype = int) # numbers of combinations to iterate from
    for combo in combo_iter_list:
        count_all_possible_combination = float(comb(nodes_num - 1, combo))
        if not count_to_sample:
            count_to_sample = np.ceil(np.log10(count_all_possible_combination) + 1).astype(int)
        for _ in range(count_to_sample):
            soil_nodes_combo_to_add = tuple(sample(range(1, nodes_num), combo))
            soil_nodes_combo_all.append(soil_nodes_combo_to_add)     
        # print("How many nodes? ", combo, "How many combos?", len(soil_nodes_combo_to_add))
        # print(soil_nodes_combo_all)
    soil_nodes_combo = pd.Series(soil_nodes_combo_all, dtype = object)
    soil_nodes_combo_count = len(soil_nodes_combo)
    print("Soil nodes count:", soil_nodes_combo_count)
    return soil_nodes_combo, soil_nodes_combo_count

def ignore_zero_div(x,y):
    if not x*y:
        try:
            return x/y
        except ZeroDivisionError:
            return 0
    else:
        return 0


def print_time(earlier_time):
    now_time = time.time()
    print("--- %s seconds ---" % round((time.time() - earlier_time),5))
    return now_time


if __name__ == '__main__':
    a = np.nan
    b = 0
    c = 5
    print('ignore_zero_div(b, a)', ignore_zero_div(b, a))
    print('ignore_zero_div(a, b)', ignore_zero_div(a, b))
    print('ignore_zero_div(c, a)', ignore_zero_div(c, a))
    print('ignore_zero_div(c, b)', ignore_zero_div(c, b))
    print('ignore_zero_div(a, c)', ignore_zero_div(a, c))