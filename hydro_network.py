import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
import itertools
import datetime as date
import datetime
import time
from random import sample
from math import comb
from math import log10
from scipy.special import factorial
import statistics
import os
import sys

## Functions
def diameter_change(gph):
    accumulate_downstream(gph)
    for k in nx.topological_sort(gph):
        print('node', k)
        # print('node cumulative area (ft2)', gph.nodes[k]['cumulative_node_drainage_area'])
        print('node cumulative area (acre)', gph.nodes[k]['cumulative_node_drainage_area']*2.3e-5)
        acre = gph.nodes[k]['cumulative_node_drainage_area']*2.3e-5
        Qd = 0.8*6*acre
        print('rational Q (c=0.8, i=5.6):', Qd)
        Dr = (2.16*Qd*0.01/np.sqrt(0.008))**(3/8)
        print('diameter for outflowing pipe', gph.out_edges(k,data='diam'))
        print('diameter calculated', Dr)

        
    # outlet_flow_rate = sum(data['edge_dq'] for u, v, data in H.in_edges(0, data = True))

def create_networks(g_type = 'gn', nodes_num = 10, n = 0.01, diam = 1, changing_diam = True, diam_increment = 0.5, soil_depth = 0, 
slope = 0.008, elev_min = 90, elev_max = 100, level = 0.5, node_drainage_area = 1.5, node_manhole_area = 50, conductivity = 0.5,
outlet_elev = None, outlet_level = None, outlet_node_drainage_area = None, seed = None, kernel = None):
    """
    create a random network with different properties. the slope has been defaulted to be the same in
    the entire network, and the diameter is defaulted to go up as the network is further away from the
    outlet.
    conductivity should somehow link to the porosity of soil. 
    node drainage area set in acre.
    """
    gph = nx.gn_graph(nodes_num, seed = seed, kernel = kernel)
    nx.topological_sort(gph)
    max_path_order = max(len(nx.shortest_path(gph, source = k, target = 0)) for k in gph.nodes)
    elev_range = np.linspace(elev_min, elev_max, num=max_path_order)

    for k in nx.topological_sort(gph):
        downstream_degree_to_outlet = len(nx.shortest_path(gph, source = k, target = 0))-1
        elev = elev_range[downstream_degree_to_outlet]
        a = dict(zip(["elev", "level", "node_drainage_area", "node_manhole_area", "soil_depth"], [elev, level, node_drainage_area, node_manhole_area, soil_depth]))
        b = dict(zip([k], [a]))
        # print(k, 'elevation',elev)
        nx.set_node_attributes(gph, b)
    accumulate_downstream(gph)
    for k in gph.edges:
        length = abs(k[0] - k[1])/slope
        downstream_degree_to_outlet = len(nx.shortest_path(gph, source = k[0], target = 0))
        diam0 = ((max_path_order - downstream_degree_to_outlet) * diam * diam_increment)*changing_diam + diam
        # print(k,'diameter in create network',diam0)
        a = dict(zip(["n", "length", "diam",'conductivity'], [n, length, diam0, conductivity]))
        b = dict(zip([k], [a]))
        nx.set_edge_attributes(gph, b)
    if changing_diam:
        for k in nx.topological_sort(gph):
            if k == 0:
                pass
            else:
                # print('node', k)
                # print('node cumulative area (ft2)', gph.nodes[k]['cumulative_node_drainage_area'])
                # print('node cumulative area (acre)', gph.nodes[k]['cumulative_node_drainage_area']*2.3e-5)
                acre = gph.nodes[k]['cumulative_node_drainage_area']
                Qd = 0.8*6*acre         # cfs
                # print('rational Q (c=0.8, i=5.6):', Qd)
                # print('diameter for outflowing pipe', gph.out_edges(k,data='diam'))
                Dr = (2.16*Qd*n/np.sqrt(slope))**(3/8)
                ds_node = [u for k, u in gph.out_edges(k,data=False)]
                # print('ds node',ds_node[0])
                gph[k][ds_node[0]]['diam']=round(Dr,1)
                # print('node',k,'diameter for outflowing pipe', gph.out_edges(k,data='diam'))
                # print('diameter calculated', Dr)
    if outlet_level: 
        nx.set_node_attributes(gph, outlet_level, "level")
    else:
        outlet_level = gph.nodes[0]['level']
    if outlet_node_drainage_area: 
        nx.set_node_attributes(gph, outlet_node_drainage_area, "node_drainage_area")
        nx.set_node_attributes(gph, outlet_node_drainage_area, "node_manhole_area")
    if outlet_elev: 
        nx.set_node_attributes(gph, outlet_elev, 'elev')
    else:
        outlet_elev = elev_min - outlet_level
        nx.set_node_attributes(gph, outlet_elev, 'elev')
   
    Manning_func(gph) 
    return gph

def fill_numbers(dictionary, full_list, number = 0):
    """
    a function to pad a dictionary with one number. this is used in the soil moisture calculation, 
    where non-water-retentive nodes automatically get s = 1 (saturated). 
    """
    my_dict = dict.fromkeys(full_list, number)
    my_dict.update(dictionary)
    return my_dict

def draw_varying_size(gph, ax = None, attribute = 'level', edge_attribute = None, node_drawing_ratio = 20):
    """
    a function to draw the network. not being used.
    """
    node_sizes = []
    node_colors = []
    labels = {}
    edge_labels = {}
    pos = graphviz_layout(gph, prog = 'dot')
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

def draw_network_timestamp(gph, ax = None, edge_attribute = 'edge_dq', soil_nodes = None, label_on = False, flood_level = 10, title = None):
    """
    draw the network flow and the dispersion coefficients at a single timestep. 
    """
    # pos = nx.spring_layout(gph)
    # pos = nx.planar_layout(gph, scale = 100)
    # print(edge_attribute)
    pos = graphviz_layout(gph, prog = 'dot')
    edge_color = {m: gph.edges[m].get(edge_attribute)/3600 for m in gph.edges}
    node_color = []
    node_size_og = 10
    node_size = []
    for node in gph: 
        if node == 0:
            node_color.append('C1')
            node_size.append(node_size_og)
        elif gph.nodes[node]['overflow']>0:
            node_color.append('C3')
            node_size.append(node_size_og*4)
        elif node in set(soil_nodes):
            node_color.append('C2')
            node_size.append(node_size_og*2)
        else:
            node_color.append('C0')
            node_size.append(node_size_og)
    node_label = {n: str(n) + ':' + str(round(gph.nodes[n].get('level') + gph.nodes[n].get('elev') ,1)) for n in gph.nodes if gph.nodes[n]['overflow']>0}
    edge_label = {m: str(round(gph.edges[m].get(edge_attribute), 2)) for m in gph.edges}
    # if edge_attribute 'edge_dq':
    #     edge_label = {m: str(round(gph.edges[m].get(edge_attribute), 2)) for m in gph.edges}

    cmap = plt.cm.RdBu
    vmin = min(min(edge_color.values()),-max(edge_color.values()))
    vcenter = 0
    vmax = max(max(edge_color.values()),-min(edge_color.values()))
    # print(vmin, vcenter, vmax)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    fig0, ax0 = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 5]})
    nx.draw(gph, pos, ax0[1], node_color = node_color, node_size = node_size, edge_color = [sm.to_rgba(i) for i in edge_color.values()], labels = node_label, font_size=6,with_labels = label_on, edge_cmap = cmap)
    # if label_on:
    #     nx.draw_networkx_edge_labels(gph, pos, edge_labels=edge_label)
    #fig = plt.gcf()
    #cbar_ax = fig0.add_axes([0.1, 0.25, 0.05, 0.35])
    plt.colorbar(sm, cax = ax0[0])
    ax0[0].set_xlabel('Edge dQ (cfs)')
    ax0[0].set_ylim(bottom = vmin,top=vmax)
    if title:
        plt.suptitle(title)

def graph_histogram(gph, kernel=None, ax=None):
    degrees = dict(gph.degree())
    degrees_list = degrees.values()
    mean_degrees = sum(degrees_list)/len(degrees)
    bin_list = np.linspace(min(degrees_list)-1, max(degrees_list), max(degrees_list) - min(degrees_list) + 2)
    if not ax:
        plt.figure()
        ax = plt.gca()
    ax.hist(degrees_list, bins=bin_list, label='Histogram',density=True)
    plt.xticks(bin_list)
    if kernel:
        # kernel = lambda x: mean_degrees**x*np.exp(-mean_degrees)/factorial(x)
    # x = np.linspace(1, 20, 20, dtype=int)
    # x = np.asarray([2, 3])
    # print(x)
    # print(factorial(x))
        ax1 = ax.twinx()
        # ax1.set_ylims(bottom = 0)    
        ax1.plot(bin_list, kernel(bin_list), 'k-o', label='Poisson Distribution')


def accumulate_downstream(gph, accum_attr='node_drainage_area', cumu_attr_name=None, soil_nodes=None):
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

def rainfall_func(dt, rain_duration = 2, size = 10, freq= 0.1, meanDepth_inch = 2, is_pulse = False):
    """
    the function to generate stochastic rainfalls or pulse rain event.
    """
    # mean depth in foot
    meanDepth_ft = meanDepth_inch/12
    depth = np.zeros(size)
    number_timesteps_with_rain = int(rain_duration/dt)
    meanDepth = meanDepth_ft/number_timesteps_with_rain
    
    # generate one single pulse rainfall event
    if is_pulse:
        start_time=2
        depth[start_time:(number_timesteps_with_rain+start_time)] = meanDepth 
    else: 
        # generate uniform and exponentially distributed random variables 
        depthExponential = np.random.exponential( meanDepth * np.ones(size) ) 
        freqUniform = np.random.random( size=size )
        # the occurence of rainfall in any independent interval is lambda*dt 
        yesrain = freqUniform<np.tile(freq,size)*dt
        # rain falls according to prob within an increment
        yesrain_index = [i for i, x in enumerate(yesrain) if x]
        for i in yesrain_index:
            depth[i:min(number_timesteps_with_rain+i,size)] += depthExponential[i]
    return depth

def soil_moisture_func(s, depth, dt, nporo = 0.45, zr = 10, emax = 0.005):
    """
    this function calculated soil moisture decays overtime on one node. this is assuming a linear decay.
    eta = emax/nporo/zr    # decay function: rho(s) = eta * s = emax/n zr * s
    gamma_soil = (nporo*zr)/meanDepth 
    dt is in hours
    """
    dt = dt/24          # dt converted to day
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
    dt = dt/24          # dt converted to day
    s = soil_moisture_func(s = old_s, dt = dt, depth = depth, nporo = nporo, zr = zr, emax = emax)
    soil_array = np.ones(len(soil_nodes))*s
    soil_moisture = dict(zip(soil_nodes, soil_array))
    soil_moisture = fill_numbers(soil_moisture, gph.nodes, number = 1) # Soil moisture is assumed "saturated" if impervious
    # print("soil moisture", soil_moisture)
    nx.set_node_attributes(gph, soil_moisture, 'soil_moisture')
    return s

# def runoff_func(gph, s, dt, depth, soil_nodes, rain_nodes, nporo = 0.45, zr = 10, emax = 0.05, flood_level = 10): 
#     """  
#     This function calculates the runoff on nodes after a result of bioretention activities. 
#     Runoff is the amount of rainfall that is unable to be absorbed by the soil, and needs to
#     be received by the stormwater pipes. (Saturation excess)
#     Overflow is not being returned to the system.
#     """
#     node_drainage_area = nx.get_node_attributes(gph,'node_drainage_area')
#     node_manhole_area = nx.get_node_attributes(gph,'node_manhole_area')
#     rain_nodes_count = len(rain_nodes)
#     precip = fill_numbers(dict(zip(rain_nodes, np.ones(rain_nodes_count)*depth)), gph.nodes) # precipitation at the highest node, exponential distribution like in the project
#     # print("precip", precip)
#     nx.set_node_attributes(gph, precip, 'precip')
#     soil_depth = nx.get_node_attributes(gph,'soil_depth')
#     soil_moisture = nx.get_node_attributes(gph, 'soil_moisture')
#     runoff = {k: (precip[k]>(1-soil_moisture[k])*soil_depth[k])*(precip[k]-(1-soil_moisture[k])*soil_depth[k]) for k in precip} # check this part
#     # print("runoff", runoff)
#     # nx.set_node_attributes(gph, runoff, 'runoff')
#     # h0 = attr_array_func('level', gph = gph, elem = 'node', ignore_outlet = True, ignore_attr=outlet_level)
#     h0 = nx.get_node_attributes(gph,'level')
#     # runoff0 = attr_array_func('runoff', gph = gph, elem = 'node')
#     # dh = attr_array_func('dhdt', gph = gph, elem = 'node') * dt
#     dhdt = nx.get_node_attributes(gph, 'dhdt')
#     #what happens if h0 + dh < 0?
#     # h_new = dict(zip(gph.nodes, h0 + dh + runoff0))
#     # print('h0',h0)
#     # print('dhdt*dt',[dhdt[k]*dt for k in gph.nodes])
#     h_new = {k: (h0[k] + dhdt[k]*dt + runoff[k]*node_drainage_area[k]/node_manhole_area[k]) for k in gph.nodes}
#     # if min(gph.edges[m]['edge_dq'] for m in gph.edges)<0:
#     #     # print(nx.get_node_attributes(gph, 'dhdt'))
#     #     print(nx.get_edge_attributes(gph, 'edge_dq'))
#     #     draw_network_timestamp(gph,soil_nodes=soil_nodes, title = 'backflowing')
#     #     plt.show()
#     # print("h_new before of:",h_new)
#     overflow = {k: (h_new[k]>flood_level)*(h_new[k]-flood_level) for k in gph.nodes}
#     h_new_in_manhole = {k: 10 for k in overflow if overflow[k] > 0}
#     h_new.update(h_new_in_manhole)
#     nx.set_node_attributes(gph, h_new, 'level')
#     nx.set_node_attributes(gph, overflow, 'overflow')
#     # print('overflow:',overflow)
#     # print("h_new after of:",h_new)
#     outlet_evaporation_func(gph, dt, evap_rate = 0.01)
    
#     if soil_nodes:
#         soil_moisture_new = bioretention(gph = gph, dt = dt, depth = depth, soil_nodes = soil_nodes, old_s = s)
#     else:
#         soil_moisture_new = s
#     # print('s_new:', soil_moisture_new)

#     return h_new, soil_moisture_new

def runoff_func(gph, s, dt, depth, soil_nodes, rain_nodes, flood_level = 10): 
    """  
    This function calculates the runoff on nodes after a result of bioretention activities. 
    Runoff is the amount of rainfall that is unable to be absorbed by the soil, and needs to
    be received by the stormwater pipes. (Saturation excess)
    Overflow is not being returned to the system.
    """
    # depth_ft_hr = depth/(dt*24)
    node_drainage_area = nx.get_node_attributes(gph,'node_drainage_area')
    node_manhole_area = nx.get_node_attributes(gph,'node_manhole_area')
    rain_nodes_count = len(rain_nodes)
    precip = fill_numbers(dict(zip(rain_nodes, np.ones(rain_nodes_count)*depth)), gph.nodes) # [ft/hr]precipitation at the highest node, exponential distribution like in the project
    # print("precip", precip)
    nx.set_node_attributes(gph, precip, 'precip')
    soil_depth = nx.get_node_attributes(gph,'soil_depth')
    soil_moisture = nx.get_node_attributes(gph, 'soil_moisture')
    # print('soil moisture', soil_moisture)
    # print('rainfall (ft/dt)', depth)
    runoff = {k: (precip[k]>((1-soil_moisture[k])*soil_depth[k]))*(precip[k]-(1-soil_moisture[k])*soil_depth[k])*node_drainage_area[k]*43560 for k in precip} # ft3 during time dt
    # print("runoff (ft3)", runoff)
    # nx.set_node_attributes(gph, runoff, 'runoff')
    # h0 = attr_array_func('level', gph = gph, elem = 'node', ignore_outlet = True, ignore_attr=outlet_level)
    h0 = nx.get_node_attributes(gph,'level')
    s0 = {k: h0[k]*node_manhole_area[k] for k in gph.nodes}     # cf
    # runoff0 = attr_array_func('runoff', gph = gph, elem = 'node')
    # dh = attr_array_func('dhdt', gph = gph, elem = 'node') * dt
    node_dq = nx.get_node_attributes(gph, 'node_dq')
    #what happens if h0 + dh < 0?
    # h_new = dict(zip(gph.nodes, h0 + dh + runoff0))
    # print('h0',h0)
    # print('dq at nodes',{k:node_dq[k]*dt for k in gph.nodes})
    h_new = {k: max(0,(s0[k] + (node_dq[k] + runoff[k])*dt)/node_manhole_area[k]) for k in gph.nodes}
    # if min(gph.edges[m]['edge_dq'] for m in gph.edges)<0:
    #     # print(nx.get_node_attributes(gph, 'dhdt'))
    #     print(nx.get_edge_attributes(gph, 'edge_dq'))
    #     draw_network_timestamp(gph,soil_nodes=soil_nodes, title = 'backflowing')
        # plt.show()
    # print("h_new before overflow:",h_new)
    overflow = {k: (h_new[k]>flood_level)*(h_new[k]-flood_level) for k in gph.nodes}
    h_new_in_manhole = {k: flood_level for k in overflow if overflow[k] > 0}
    h_new.update(h_new_in_manhole)
    nx.set_node_attributes(gph, h_new, 'level')
    nx.set_node_attributes(gph, overflow, 'overflow')
    # print('overflow:',overflow)
    # print("h_new after overflow:",h_new)
    outlet_evaporation_func(gph, dt, evap_rate = 0.01)
    
    if soil_nodes:
        soil_moisture_new = bioretention(gph = gph, dt = dt, depth = depth, soil_nodes = soil_nodes, old_s = s)
    else:
        soil_moisture_new = s
    # print('s_new:', soil_moisture_new)

    return h_new, soil_moisture_new

def outlet_evaporation_func(gph, dt, outlet_node = 0, evap_rate = 0.01, level_name = 'level'):
    """
    This accounts for water evaporation at outlet. The default value is 0.01 ft per day, 
    which is nearly 3 mm per day. This is a linear function. 
    """
    dt = dt/24
    new_level = gph.nodes[outlet_node].get(level_name) - evap_rate*dt
    # new_level = (1-evap_rate)*gph.nodes[outlet_node].get(level_name)
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

def Manning_func(gph, elev = 'elev', level = 'level', width = 'diam', n_name = 'n', l_name = 'length', shape = 'circular', flood_level = 10):
    """
    Manning's equation to calculate open channel flow in pipes.
    """
    # initialize
    edge_list = []
    node_list = gph.nodes
    node_manhole_area_dict = nx.get_node_attributes(gph, 'node_manhole_area')
    node_manhole_area = np.array([node_manhole_area_dict[k] for k in gph.nodes])
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
        h = gph.nodes[us_node].get(level)
        # h = gph.nodes[ds_node].get(level)
        elevdiff = gph.nodes[us_node].get(elev) + gph.nodes[us_node].get(level) - gph.nodes[ds_node].get(elev) - gph.nodes[ds_node].get(level)
        d = gph.edges[m].get(width)
        n = gph.edges[m].get(n_name)
        l = gph.edges[m].get(l_name)
        s = abs(elevdiff)/l
        # s = 0.008
        if h > d:   # This should be where pressured pipes are included
            print(m,"h=d","elev diff", elevdiff,'h',h,'d',d)
            # h = d
            return True
            break
        if shape == 'circular':
            if h <= 0: 
                # print(m, 'h', h,'h<=0', (h <= 0))
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
        u = 1.49/n*R**(2/3)*s**(1/2)*3600    # from ft/s to ft/hour
        dq = np.sign(elevdiff)*u*A # Manning's Equation (Imperial Unit) for edges
        # print('edge', m, 'dq', dq, 'elevdiff', elevdiff)

        # if dq > gph.nodes[us_node].get("node_manhole_area")*gph.nodes[us_node].get(level):
        #     # This is to compare volumetric flow rate in pipe with volume of water in upstream manhole
        #     dq = gph.nodes[us_node].get("node_manhole_area")*gph.nodes[us_node].get(level)
        #     u = abs(ignore_zero_div(dq, A))
        #     print(m,'capped at upstream')
        # elif dq < - gph.nodes[ds_node].get("node_manhole_area")*gph.nodes[ds_node].get(level):
        #     dq = - gph.nodes[ds_node].get("node_manhole_area")*gph.nodes[ds_node].get(level)
        #     u = abs(ignore_zero_div(dq, A))
        #     print(m,'capped at downstream')

            # print('dq after cap', dq)
            # print('term 1', gph.nodes[us_node].get("node_drainage_area")*gph.nodes[us_node].get(level))
            # print('term 2 us elev', gph.nodes[us_node].get(elev) )
            # print('term 2 us dh', gph.nodes[us_node].get(level) )
            # print('term 2 ds elev + dh', ((gph.nodes[ds_node].get(elev) + gph.nodes[ds_node].get(level))))
        
        # if (gph.nodes[us_node].get(elev) + gph.nodes[us_node].get(level)) < (gph.nodes[ds_node].get(elev) + gph.nodes[ds_node].get(level)):
        #     draw_network_timestamp(gph,soil_nodes=())
        #     plt.show()
        #     print(m, 'dq has been floored', dq)
        #     print('term 2 us elev', gph.nodes[us_node].get(elev) )
        #     print('term 2 us dh', gph.nodes[us_node].get(level) )
        #     print('term 2 ds elev + dh', ((gph.nodes[ds_node].get(elev) + gph.nodes[ds_node].get(level))))
        #     print('upstream is lower than downstream')

        if (abs(dq) < 1e-4):
            # print('abs(dq) < 1e-4')
            # print('m', m, 'u (ft/s)', u)
            dq = 0
            u = 0
        # print('edge', m, 'dq', dq)
        t = ignore_zero_div(l,abs(u))
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
    else: 
        # # calculate ACAt
        # print('dq_list',dq_list)
        Q = dq_list*np.eye(len(dq_list))
        A = (nx.incidence_matrix(gph, oriented=True)).toarray()
        edge_cnt = len(gph.edges)
        J = np.ones(edge_cnt)
        dq_node_list = A@Q@J
        # dhdt_list = np.divide(A@Q@J,node_manhole_area) # this is at the nodes
        # print('Q',dq_list)
        # print(Q)
        # print('dhdt_list',dhdt_list)
        
        # record results
        dict_dq = dict(zip(edge_list, dq_list))         # flow through edge
        dict_h0 = dict(zip(edge_list, h_list))          # water level in pipes
        dict_rad = dict(zip(edge_list, h_radius_list))  # hydraulic radius
        dict_t = dict(zip(edge_list, t_list))
        dict_u = dict(zip(edge_list, u_list))
        dict_node_dq = dict(zip(node_list, dq_node_list))
        # dict_dhdt = dict(zip(node_list, dhdt_list))

        # assign results in map
        nx.set_edge_attributes(gph, dict_dq, 'edge_dq')
        nx.set_edge_attributes(gph, dict_h0, 'edge_h')
        nx.set_edge_attributes(gph, dict_rad, 'h_rad')
        nx.set_edge_attributes(gph, dict_t, 'edge_time')
        nx.set_edge_attributes(gph, dict_u, 'edge_velocity')
        # nx.set_node_attributes(gph, dict_dhdt, 'dhdt')
        nx.set_node_attributes(gph, dict_node_dq, 'node_dq')

        # plt.show()
        # # # translate to increase at nodes (CALCULATIONS CHECK)
        # dhdt1_list = []
        # node_list = []
        # for n in gph.nodes:
        #     inflow = 0
        #     outflow = 0
        #     n_area = gph.nodes[n].get('node_drainage_area')
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
        return False

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
    # print('network_path_length_dict', network_path_length_dict)
    # print('network_path_dq_dict', nx.get_edge_attributes(gph, 'edge_dq'))
    # print('network_path_time_dict', network_path_time_dict)

    try: 
        mean_path_length = statistics.mean(network_path_length_set)
        mean_path_time = statistics.mean(network_path_time_set)
        var_path_length = statistics.variance(network_path_length_set)
        disp_g = 0.5*ignore_zero_div(var_path_length,mean_path_time)
        #print(disp_g)
        network_celerity = ignore_zero_div(mean_path_length,mean_path_time)
        # print('network_celerity', network_celerity)
        flowpath_celerity_dict = {k: ignore_zero_div(network_path_length_dict[k],network_path_time_dict[k]) for k in network_path_time_dict}
        # print('flowpath_celerity_dict', flowpath_celerity_dict)
        stretched_path_length_set = [ignore_zero_div(network_path_length_dict[k],flowpath_celerity_dict[k])*network_celerity for k in network_path_length_dict]
        # print('stretched_path_length_set', stretched_path_length_set)
        disp_kg = 0.5*ignore_zero_div(statistics.variance(stretched_path_length_set),mean_path_length)*network_celerity
    except statistics.StatisticsError:
        var_path_length = 0
        disp_g = 0
        disp_kg = 0

    return var_path_length, disp_g, disp_kg

def neighbor_index_calc():
    pass

def random_sample_soil_nodes(nodes_num, count_to_sample = None, range_min = 1, range_max = 20, range_count = 10):
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
    return soil_nodes_combo, soil_nodes_combo_count

def ignore_zero_div(x,y):
    np.seterr(all='ignore')
    if np.isnan(x*y):
        return 0
    else:
        try:
            return x/y
        except ZeroDivisionError or FloatingPointError:
            return 0

def print_time(earlier_time):
    now_time = time.time()
    print("--- %s seconds ---" % round((time.time() - earlier_time),5))
    return now_time


if __name__ == '__main__':
    kernel = lambda x: np.exp(-2)*2**x/factorial(x)
    G = create_networks(nodes_num=10,kernel=kernel,node_drainage_area=87120)
    depth = rainfall_func(dt=0.25,freq=0.8,is_pulse=False)
    print(depth)
    # draw_varying_size(G,edge_attribute='diam')
    # plt.show()