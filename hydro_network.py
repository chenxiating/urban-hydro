from networkx.algorithms.cluster import clustering
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
from math import floor
from scipy.special import factorial
import statistics
import os
import sys

class Storm_network:

    def __init__(self, beta = 0.5, nodes_num = 10, n = 0.01, diam = 1, changing_diam = True, diam_increment = 0.5, soil_depth = 0, 
slope = 0.008, elev_min = 90, elev_max = 100, level = 0.5, node_drainage_area = 1.5, node_manhole_area = 50, conductivity = 0.5,
outlet_elev = 85, outlet_level = 1, outlet_node_drainage_area = None, seed = None, soil_nodes = None, count = 0):
        """ create a random network with different properties. the slope has been defaulted to be the same in
        the entire network, and the diameter is defaulted to go up as the network is further away from the
        outlet. conductivity should somehow link to the porosity of soil. node drainage area set in acre.
        """
        self.beta = beta
        self.nodes_num = nodes_num
        # initialize graph
        # gph = my_grid_graph(m=int(np.sqrt(nodes_num)),n=int(np.sqrt(nodes_num)),beta=beta)
        self.matrix = pickle.load(open(r'../gibbs_grid/10-grid_0.pickle','rb'))
        self.gph = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        self.nodes_num = len(self.gph.nodes)
        # initialize topological order and elevation
        nx.topological_sort(self.gph)
        nodes_in_order = list(nx.topological_sort(self.gph))
        self.outlet_node = nodes_in_order[len(nodes_in_order)-1]
        max_path_order = max(len(nx.shortest_path(self.gph, source = k, target = self.outlet_node)) for k in self.gph.nodes)
        elev_range = np.linspace(elev_min, elev_max, num=max_path_order)
        
        for k in nx.topological_sort(self.gph):
            downstream_degree_to_outlet = len(nx.shortest_path(self.gph, source = k, target = self.outlet_node))-1
            elev = elev_range[downstream_degree_to_outlet]
            a = dict(zip(["elev", "level", "node_drainage_area", "node_manhole_area", "soil_depth"], 
            [elev, level, node_drainage_area, node_manhole_area, soil_depth]))
            b = dict(zip([k], [a]))
            # print(k, 'elevation',elev)
            nx.set_node_attributes(self.gph, b)
        self.max_path_order = max_path_order
        self.downstream_degree_to_outlet = {k: len(nx.shortest_path(self.gph, source = k, target = self.outlet_node))-1 for k in self.gph.nodes}
        self.accumulate_downstream()
        self.get_coordinates()
        self.flood_nodes = None
        self.soil_nodes = ()

        if soil_nodes:
            self.soil_nodes = soil_nodes
        self.random_sample_soil_nodes(count=count)
        
        if outlet_elev: 
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_elev}, 'elev')
        else:
            outlet_elev = elev_min - outlet_level
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_elev}, 'elev')
        for k in self.gph.edges:
            elev_us = self.gph.nodes[k[0]].get('elev')
            elev_ds = self.gph.nodes[k[1]].get('elev')
            length = abs(elev_us-elev_ds)/slope
            downstream_degree_to_outlet = len(nx.shortest_path(self.gph, source = k[0], target = self.outlet_node))
            a = dict(zip(["n", "length",'conductivity'], [n, length, conductivity]))
            b = dict(zip([k], [a]))
            nx.set_edge_attributes(self.gph, b)
        if changing_diam:
            for k in nx.topological_sort(self.gph):
                if k == self.outlet_node:
                    pass
                else:
                    acre = self.gph.nodes[k]['cumulative_node_drainage_area']
                    Qd = 0.96*(4.18/24)*acre         # cfs, 
                    # i should be in in/hr. Design storm
                    # 100-year 24-hr storm is 7.4 inch, 100-yr 2-hr storm is 4.55 inch.
                    # 10-year 24-hr storm is 4.18 inch.
                    Dr = (2.16*Qd*n/np.sqrt(slope))**(3/8) # Mays 15.2.7, page 621
                    ds_node = [ds_node for k, ds_node in self.gph.out_edges(k,data=False)][0]
                    # print('ds node',ds_node[0])
                    self.gph[k][ds_node]['diam']=round_pipe_diam(Dr)
                    # print('node',k,'diameter for outflowing pipe', gph.out_edges(k,data='diam'))
                    # print('diameter calculated', Dr)
        if outlet_level: 
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_level}, "level")
        else:
            outlet_level = self.gph.nodes[self.outlet_node]['level']
        if outlet_node_drainage_area: 
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_node_drainage_area}, "node_drainage_area")
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_node_drainage_area}, "node_manhole_area")
        return

    def get_coordinates(self):
        # convert matrix index to (i,j)  coordinates
        n = np.sqrt(self.nodes_num)
        coordinates = graphviz_layout(self.gph, prog = 'dot')
        nx.set_node_attributes(self.gph, coordinates, 'coordinates')
        return

    def accumulate_downstream(self, accum_attr='node_drainage_area', cumu_attr_name=None):
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

        for topo_node in nx.topological_sort(self.gph):
            # grab value in current node
            attrib_val = self.gph.nodes[topo_node].get(accum_attr, 0)

            # sum with cumulative values in upstream nodes and edges
            for p in self.gph.predecessors(topo_node):
                # add cumulative attribute val in upstream node, apply flow split fraction
                attrib_val += self.gph.nodes[p][cumu_attr_name]

                # store cumulative value in upstream edge
                self.gph[p][topo_node][cumu_attr_name] = attrib_val

                # add area routed directly to upstream edge/sewer
                attrib_val += self.gph[p][topo_node].get(accum_attr, 0)
            # store cumulative attribute value in current node
            self.gph.nodes[topo_node][cumu_attr_name] = attrib_val

        return 

    def random_sample_soil_nodes(self, count):
        us_nodes_to_sample = list(self.gph.nodes).copy()
        us_nodes_to_sample.remove(self.outlet_node)

        if self.soil_nodes and (count - len(self.soil_nodes))>0: 
            set_to_sample_from = set(us_nodes_to_sample) - set(self.soil_nodes)
            new_soil_nodes = sample(set_to_sample_from, count - len(self.soil_nodes))
            total_soil_nodes = tuple(set(self.soil_nodes) + set(new_soil_nodes))
            self.soil_nodes = total_soil_nodes
        else: 
            self.soil_nodes = tuple(sample(us_nodes_to_sample, count))
    
    # def random_sample_soil_nodes(self, count_to_sample = None, range_min = 1, range_max = 20, range_count = 10):
    #     """
    #     randomly generate water retentive nodes in the network
    #     """
    #     if range_max >= self.nodes_num:
    #         range_max = self.nodes_num - 1
    #     if range_min >= self.nodes_num:
    #         range_min = self.nodes_num - 1
    #     range_len = range_max - range_min + 1
    #     if range_count > range_len:
    #         range_count = range_len 
        
    #     us_nodes_to_sample = self.nodes.copy()
    #     us_nodes_to_sample.remove(self.outlet_node)
    #     soil_nodes_combo_all = []
    #     combo_iter_list = np.linspace(range_min, range_max, num = range_count, dtype = int) # numbers of combinations to iterate from
    #     for combo in combo_iter_list:
    #         count_all_possible_combination = float(comb(self.nodes_num - 1, combo))
    #         if not count_to_sample:
    #             count_to_sample = np.ceil(np.log10(count_all_possible_combination) + 1).astype(int)
    #         for _ in range(count_to_sample):
    #             # soil_nodes_combo_to_add = tuple(sample(range(1, nodes_num), combo))
    #             soil_nodes_combo_to_add = tuple(sample(us_nodes_to_sample, combo))
    #             soil_nodes_combo_all.append(soil_nodes_combo_to_add)     
    #         # print("How many nodes? ", combo, "How many combos?", len(soil_nodes_combo_to_add))
    #         # print(soil_nodes_combo_all)
    #     soil_nodes_combo = pd.Series(soil_nodes_combo_all, dtype = object)
    #     soil_nodes_combo_count = len(soil_nodes_combo)
    #     return soil_nodes_combo, soil_nodes_combo_count
    
    def calc_node_distance(self,type = 'soil'):
        if type == 'flood':
            nodes = self.flood_nodes
        else: 
            nodes = self.soil_nodes
        nodes_length = len(nodes)
        if nodes_length == 0:
            return 0
        else:
            total_path = 0
            for k in nodes:
                each_path = len(nx.shortest_path(self.gph, source=k, target = self.outlet_node)) - 1
                total_path = total_path + each_path
            node_elev = total_path/nodes_length
            return node_elev
    
    def calc_node_degree(self,type = 'soil'):
        if type == 'flood':
            nodes = self.flood_nodes
        else: 
            nodes = self.soil_nodes
        nodes_length = len(nodes)
        if nodes_length == 0:
            return 0
        else:
            degrees = dict(self.gph.degree())
            # soil_node_degree = ignore_zero_div(sum(degrees.get(k,0) for k in soil_nodes),soil_nodes_length)
            node_degree_sum = sum(degrees[k]*degrees[k] for k in nodes)
            # print(soil_node_degree_sum)
            node_degree = ignore_zero_div(node_degree_sum,nodes_length)
            return node_degree
    
    def calc_flow_path(self, accum_attr='length', path_attr_name=None):
        """calculate flow paths lengths and travel time"""
        path_dict = {}
        if path_attr_name is None:
            path_attr_name = 'path_{}'.format(accum_attr)
        for node in self.gph.nodes:
            shortest_path_set = list(nx.shortest_path(self.gph, source = node, target = self.outlet_node))
            path_attr = 0
            i = 0
            keep_running = True
            while (keep_running):
                for i in shortest_path_set:
                    out_edge_set = self.gph.edges(i)
                    if not out_edge_set: 
                        break
                    for out_edge in out_edge_set:
                        if self.gph.edges[out_edge].get('edge_velocity') > 0: 
                            path_attr += self.gph.edges[out_edge].get(accum_attr)
                            # print('succeed:', node, 'i_node', i, out_edge, gph.edges[out_edge].get('edge_velocity'), gph.edges[out_edge].get(accum_attr), path_attr)
                        else: 
                            # print('fail:', node, 'i_node', i, out_edge, gph.edges[out_edge].get('edge_velocity'), gph.edges[out_edge].get(accum_attr), path_attr)
                            path_attr = 0
                            keep_running = False
                            break
                keep_running = False
            path_dict[node] = path_attr
        nx.set_node_attributes(self.gph, path_dict, path_attr_name)
        return path_dict

    def calc_node_clustering(self,type = 'soil'):
        clustering_coef = 0
        all_nodes = {}
        all_ds_nodes = {}
        big_group = []

        if type == 'flood':
            us_search_nodes = self.flood_nodes
        else: 
            us_search_nodes = self.soil_nodes
        
        ds_search_nodes = us_search_nodes.copy()

        def iter_nodes(self, nodes, dir = 'ds'):
            
            def edge_dir(node, dir):
                if dir == 'ds':
                    return self.gph.out_edges(node)
                else: 
                    return self.gph.in_edges(node)
            
            def search_neighbor(node):
                edges = edge_dir(node,dir)
                all_nodes[node] = [edge[n] for edge in edges]
                gi_node = list(node for node in all_nodes[node] if node in nodes)
                return gi_node
            
            n = (dir == 'ds')
            for node in nodes:
                small_group = [node]
                gi_node = search_neighbor(node)                
                while len(gi_node) > 0:
                    small_group.append(node for node in gi_node)
                    nodes.remove(node for node in gi_node)
                    gi_node = search_neighbor(node)

        print(clustering_coef)
        return clustering_coef
    
    def calc_upstream_cumulative_area(self,accum_attr='node_drainage_area', cumu_attr_name=None):
        if cumu_attr_name is None:
            cumu_attr_name = 'cumulative_{}'.format(accum_attr)
                
        cumulative_attr_value = sum(self.gph.nodes[node][cumu_attr_name] for node in self.soil_nodes)
        # print('soil_nodes',self.soil_nodes,'cumu',cumulative_attr_value)
        return cumulative_attr_value

    def draw_network_init(self, ax = None, label_on = False, title = None):
        """ draw the network flow and the dispersion coefficients at a single timestep. """
        # pos = nx.spring_layout(gph)
        # pos = nx.planar_layout(gph, scale = 100)
        # print(edge_attribute)
        pos = graphviz_layout(self.gph, prog = 'dot')
        node_color = []
        node_size_og = 10
        node_size = []
        node_label = {node:str(node) for node in self.gph.nodes}
        for node in self.gph: 
            if node == self.outlet_node:
                node_color.append('C1')
                node_size.append(node_size_og)
            elif node in set(self.soil_nodes):
                node_color.append('C2')
                node_size.append(node_size_og*2)
                node_label[node]=str(node)
            elif self.flood_nodes: 
                node_color.append('C3')
                node_size.append(node_size_og*2)
                node_label[node]=str(node)
            else:
                node_color.append('C0')
                node_size.append(node_size_og)

        # _, ax0 = plt.subplots(1,2, gridspec_kw={'width_ratios': [2, 3]})
        _ = plt.figure()
        ax1 = plt.subplot(122)
        nx.draw(self.gph, pos, ax1, node_color = node_color, node_size = node_size,labels = node_label, 
        font_size=6,with_labels = label_on)
        ax1.set_title('Network')
        
        ax2 = plt.subplot(221)
        k = self.downstream_degree_to_outlet
        distance_dist = [k[j] for j in k]
        ax2.hist(distance_dist)
        ax2.set_xlabel('Dist. to Outlet')
        ax2.set_ylabel('Count')
        
        ax3 = plt.subplot(223)
        m = dict(self.gph.degree())
        degree_dist = [m[j] for j in m]
        ax3.hist(degree_dist)
        ax3.set_xlabel('Degrees')
        ax3.set_ylabel('Count')

        plt.tight_layout()
        
        if title:
            plt.suptitle(title)
            
    def draw_network_timestamp(self, ax = None, edge_attribute = 'edge_dq', label_on = False, flood_level = 10, title = None):
        """ draw the network flow and the dispersion coefficients at a single timestep. """
        # pos = nx.spring_layout(gph)
        # pos = nx.planar_layout(gph, scale = 100)
        # print(edge_attribute)
        pos = graphviz_layout(self.gph, prog = 'dot')
        edge_color = {m: self.gph.edges[m].get(edge_attribute)/3600 for m in self.gph.edges}
        node_color = []
        node_size_og = 10
        node_size = []
        for node in self.gph: 
            if node == 0:
                node_color.append('C1')
                node_size.append(node_size_og)
            elif self.gph.nodes[node]['overflow']>0:
                node_color.append('C3')
                node_size.append(node_size_og*4)
            elif node in set(self.soil_nodes):
                node_color.append('C2')
                node_size.append(node_size_og*2)
            else:
                node_color.append('C0')
                node_size.append(node_size_og)
        node_label = {n: str(n) + ':' + str(round(self.gph.nodes[n].get('level') + self.gph.nodes[n].get('elev') ,1)) 
        for n in self.gph.nodes if self.gph.nodes[n]['overflow']>0}
        edge_label = {m: str(round(self.gph.edges[m].get(edge_attribute), 2)) for m in self.gph.edges}
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
        nx.draw(self.gph, pos, ax0[1], node_color = node_color, node_size = node_size, 
        edge_color = [sm.to_rgba(i) for i in edge_color.values()], labels = node_label, 
        font_size=6,with_labels = label_on, edge_cmap = cmap)
        # if label_on:
        #     nx.draw_networkx_edge_labels(gph, pos, edge_labels=edge_label)
        #fig = plt.gcf()
        #cbar_ax = fig0.add_axes([0.1, 0.25, 0.05, 0.35])
        plt.colorbar(sm, cax = ax0[0])
        ax0[0].set_xlabel('Edge dQ (cfs)')
        ax0[0].set_ylim(bottom = vmin,top=vmax)
        if title:
            plt.suptitle(title)

    def graph_histogram(self, kernel=None, ax=None):
        degrees = dict(self.gph.degree())
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

    def dispersion_func(self, l_name = 'length', t_name = 'edge_time'):
        """
        calculate dispersion coefficients. this doesn't work with Darcy yet.
        """
        path_length = 'path_{}'.format(l_name)
        path_time = 'path_{}'.format(t_name)

        network_path_length_dict = nx.get_node_attributes(self.gph, path_length)
        network_path_length_set = [network_path_length_dict[k] for k in network_path_length_dict]
        network_path_time_dict = nx.get_node_attributes(self.gph, path_time)
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

    def draw_varying_size(self, ax = None, attribute = 'level', edge_attribute = None, node_drawing_ratio = 20):
        """
        a function to draw the network. not being used.
        """
        node_sizes = []
        node_colors = []
        labels = {}
        edge_labels = {}
        pos = graphviz_layout(self.gph, prog = 'dot')
        cnt = len(self.gph.nodes)
        for n in nx.topological_sort(self.gph):
            x = self.gph.nodes[n].get(attribute)
            node_sizes.append(node_drawing_ratio*x)
            node_colors.append(n/cnt)
            labels[n] = str(n) + ": " + str(round(x, 1))
        
        for m in self.gph.edges: 
            if edge_attribute is None:
                edge_labels[m] = str(m)
            else:
                edge_labels[m] = self.gph.edges[m].get(edge_attribute)
        
        nx.draw(self.gph, pos, ax, node_color = node_colors, node_size = node_sizes, labels=labels, with_labels=True, cmap = plt.cm.rainbow)
        nx.draw_networkx_edge_labels(self.gph, pos, edge_labels=edge_labels)
        #node_colors = np.pad(node_colors[0:(len(node_colors)-1)], (1, 0), constant_values = 0)
        #node_colors_array = [plt.cm.rainbow(x) for x in node_colors]node_colors_array = [plt.cm.rainbow(x) for x in node_colors]
        rainbow = plt.cm.get_cmap('rainbow',cnt)
        node_colors_array = [rainbow(x) for x in node_colors]
        #print(dict(zip(["node", "node_colors", "node_colors_array"],[gph.nodes, node_colors, node_colors_array])))
        return node_colors_array

# other utility codes
def round_pipe_diam(old_diam):
    old_inch = old_diam * 12
    if old_inch <= 12: 
        new_diam = 12
    elif old_inch <= 15:
        new_diam = 15
    elif old_inch <= 18:
        new_diam = 18
    elif old_inch <= 21:
        new_diam = 21
    elif old_inch <= 24:
        new_diam = 24
    elif old_inch <= 30:
        new_diam = 30
    elif old_inch <= 33:
        new_diam = 33
    elif old_inch <= 36:
        new_diam = 36
    elif old_inch <= 42:
        new_diam = 42
    elif old_inch <= 48: 
        new_diam = 48
    elif old_inch <= 54:
        new_diam = 54
    elif old_inch <= 60:
        new_diam = 60
    elif old_inch <= 66:
        new_diam = 66
    elif old_inch <= 72:
        new_diam = 72
    elif old_inch <= 78:
        new_diam = 78
    else:
        new_diam = round(old_inch)
    return new_diam/12

def fill_numbers(dictionary, full_list, number = 0):
    """
    a function to pad a dictionary with one number. this is used in the soil moisture calculation, 
    where non-water-retentive nodes automatically get s = 1 (saturated). 
    """
    my_dict = dict.fromkeys(full_list, number)
    my_dict.update(dictionary)
    return my_dict

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
    os.chdir(r'./gibbs_grid')
    # kernel = lambda x: np.exp(-2)*2**x/factorial(x)
    storm_web = Storm_network(beta=0.5,nodes_num=25,node_drainage_area=87120)
    # pos_grid = {node: (math.floor(node/self.nodes_num))) for node in storm_web.gph}
    pos = graphviz_layout(storm_web.gph, prog = 'dot')
    nx.draw(storm_web.gph,pos,node_size = 2)#,pos_grid,with_labels=True)
    plt.show()
