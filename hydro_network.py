"""
hydro_network.py 
@author: Xiating Chen
Last Edited: 2023/10/07


This code is to assign network attributes. 
    - Input: desired stormwater network attributes
    - Output: stormwater networks with attributes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
from collections import Counter
import time
from random import sample
from math import floor
import statistics
import Gibbs

class Storm_network:
    def __init__(self, nodes_num, beta = 0.5, n = 0.01, min_diam = 1, changing_diam = True, soil_depth = 0, slope = 0.008, elev_min = 90, 
pipe_length = 200, level = 0, node_drainage_area = 2, outlet_elev = 85, outlet_level = 1, outlet_node_drainage_area = None, 
soil_nodes = None, count = 0, fixing_graph = False, file_name = None, make_cluster = None, runoff_coef = 0.5):
        """ create a random network with different properties. the slope has been defaulted to be the same in
        the entire network, and the diameter is defaulted to go up as the network is further away from the
        outlet.

        If file_name or fixing_graph is not specified, new graphs will be generated. 

        nodes_num:      number of nodes in the graph
        beta:           Gibbs distribution parameter
        n:              Manning's roughness n
        min_diam:       minimum pipe diameter (ft), default 1
        changing_diam:  fixed pipe diameter or changing diameter based on drainage area, default sized to drainage
        soil_depth:     LID soil storage thickness (ft), default 0
        slope:          pipe slope (-), default 0.008
        elev_min:       lowest elevation point of the network, excluding the outlet (ft), default 90
        pipe_length:    pipe length (ft), default 200
        level:          initial water level in node (ft), default 0
        node_drainage_area:         drainage area per node/ catchment (acre), default 2
        outlet_elev:    outlet elevation point (ft), default 85
        outlet_level:   outlet water level (ft), default 1
        outlet_node_drainage_area:  drainage area to outelt (acre), default None
        soil_nodes:     the names of the green infrastructure nodes, default None
        count:          number of green infrastructure nodes, default 0
        fixing_graph:   run simulation on one single graph, default False
        file_name:      file path of the graph, default None
        make_cluster:   making green infrastructure nodes in close clusters, default False
        runoff_coef:    determine the runoff coefficient, default 0.5. We used 0.8 for "impervious", and 0.5 for "suburban"
        """

        if file_name:
            beta = file_name[file_name.find('beta-')+len('beta-'):file_name.rfind('_dist')]
        self.beta = float(beta)
        self.count = count
        self.n = int(np.sqrt(nodes_num))
        self.nodes_num = nodes_num
        if fixing_graph or file_name:
            if not file_name:
                file_name = f'{nodes_num}-node_graph_mp.pickle'
            try: 
                path = rf'{file_name}'
                input_matrix = np.array(pickle.load(open(path,'rb')))
                self.generate_graph(input_matrix=input_matrix)
            except FileNotFoundError: 
                self.generate_graph(file_name)
        else:
            self.generate_graph()
        self.matrix = self.network.matrix
        self.gph = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        self.set_attributes(elev_min, level, node_drainage_area, soil_depth, make_cluster, soil_nodes, 
        count, pipe_length, slope, changing_diam, min_diam, outlet_node_drainage_area, outlet_elev, outlet_level, n,
        runoff_coef)

    def set_attributes(self, elev_min, level, node_drainage_area, soil_depth, make_cluster, soil_nodes, 
    count,pipe_length,slope, changing_diam, min_diam, outlet_node_drainage_area, outlet_elev, outlet_level, n,
    runoff_coef):
        """ setting node and edge attributes to graphs """
        # initialize topological order and elevation
        nx.topological_sort(self.gph)
        nodes_in_order = list(nx.topological_sort(self.gph))
        self.gph.add_edge(nodes_in_order[len(nodes_in_order)-1],-1)
        self.outlet_node = -1
        max_path_order = max(len(nx.shortest_path(self.gph, source = k, target = self.outlet_node)) for k in self.gph.nodes)
        elev_range = np.arange(0,max_path_order)*pipe_length*slope + elev_min  
        
        # assigning edges attributes and topological orders
        for k in nx.topological_sort(self.gph):
            downstream_degree_to_outlet = len(nx.shortest_path(self.gph, source = k, target = self.outlet_node))-1
            elev = elev_range[downstream_degree_to_outlet]
            a = dict(zip(["elev", "level", "node_drainage_area", "soil_depth"], 
            [elev, level, node_drainage_area, soil_depth]))
            b = dict(zip([k], [a]))
            nx.set_node_attributes(self.gph, b)
        self.max_path_order = max_path_order
        self.downstream_degree_to_outlet = {k: len(nx.shortest_path(self.gph, source = k, target = self.outlet_node))-1 for k in self.gph.nodes}
        self.get_coordinates()
        self.accumulate_downstream()
        self.flood_nodes = None
        self.soil_nodes = ()
        self.pipe_cap = 0

        # make green infrastructure cluster
        if make_cluster:
            starting_dist = make_cluster
            self.make_dist_cluster(starting_dist)
        elif soil_nodes:
            self.soil_nodes = soil_nodes
        else:
            self.random_sample_soil_nodes(count=count)
        
        # changing outlet elevation
        if outlet_elev: 
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_elev}, 'elev')
        else:
            outlet_elev = elev_min - outlet_level
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_elev}, 'elev')
        for k in self.gph.edges:
            downstream_degree_to_outlet = len(nx.shortest_path(self.gph, source = k[0], target = self.outlet_node))
            a = dict(zip(["n", "length"], [n, pipe_length]))
            b = dict(zip([k], [a]))
            nx.set_edge_attributes(self.gph, b)
            
        # size edges to cumulative upstream drainage area using the Rational Method 
        if changing_diam:
            for k in nx.topological_sort(self.gph):
                if k == self.outlet_node:
                    pass
                else:
                    C = runoff_coef
                    acre = self.gph.nodes[k]['cumulative_node_drainage_area']
                    Qd = C*(4.18/24)*acre         # cfs, 
                    # Sized to 10-year 24-hr storm is 4.18 inch. i should be in in/hr.
                    Dr = (2.16*Qd*n/np.sqrt(slope))**(3/8) # Mays 15.2.7, page 621
                    ds_node = [ds_node for k, ds_node in self.gph.out_edges(k,data=False)][0]
                    self.gph[k][ds_node]['diam']=round_pipe_diam(Dr)
                    self.pipe_cap += (self.gph[k][ds_node]['diam']/2)**2*(np.pi)*self.gph[k][ds_node]['length']
        else:
            nx.set_edge_attributes(self.gph, {edge: {'diam':min_diam} for edge in self.gph.edges})
            self.pipe_cap = sum(e['length'] for e in dict(self.gph.edges).values()) * (min_diam/2)**2*(np.pi)
        
        if outlet_level: 
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_level}, "level")
        else:
            outlet_level = self.gph.nodes[self.outlet_node]['level']
        
        if outlet_node_drainage_area: 
            nx.set_node_attributes(self.gph, {self.outlet_node: outlet_node_drainage_area}, "node_drainage_area")
        self.min_diam = min(e['diam'] for e in dict(self.gph.edges).values())
        self.max_diam = max(e['diam'] for e in dict(self.gph.edges).values())
        return

    def generate_graph(self, input_matrix = None):
        """ generate graph using Gibbs.py """
        if input_matrix is not None:
            outlet_point = None
            self.n = int(np.sqrt(input_matrix.shape[0]))
        self.network = Gibbs.main(size=self.n, beta = self.beta,outlet_point = outlet_point,input_matrix=input_matrix)
        if input_matrix is None: 
            self.network.export_tree()
            print("Exported tree in hydro_network.generate_graph")

    def accumulate_downstream(self, accum_attr='node_drainage_area', cumu_attr_name=None):
        """
        pass through the graph from upstream to downstream and accumulate the value
        an attribute found in nodes and edges, and assign the accumulated value
        as a new attribute in each node and edge.
        Where there's a flow split, apply an optional split fraction to
        coded in the upstream edge. (This is from Adam Erispaha's Sewergraph Package).
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
    
    def get_coordinates(self):
        """ convert matrix index to (i,j)  coordinates"""
        coordinates = {node: self.network.convert_index(node) for node in self.gph.nodes}
        nx.set_node_attributes(self.gph, coordinates, 'coordinates')
        return

    def random_sample_soil_nodes(self, count):
        """ randomly assign green infrastructure nodes """
        us_nodes_to_sample = list(self.gph.nodes).copy()
        us_nodes_to_sample.remove(self.outlet_node)

        if self.soil_nodes and (count - len(self.soil_nodes))>0: 
            set_to_sample_from = set(us_nodes_to_sample) - set(self.soil_nodes)
            new_soil_nodes = sample(set_to_sample_from, count - len(self.soil_nodes))
            total_soil_nodes = tuple(set(self.soil_nodes) + set(new_soil_nodes))
            self.soil_nodes = total_soil_nodes
        else: 
            self.soil_nodes = tuple(sample(us_nodes_to_sample, count))
    
    def make_dist_dict(self):
        """ create dictionary that shows each node's distance to the outlet """
        dist_dict = {k:(len(nx.shortest_path(self.gph, source=k, target = self.outlet_node)) - 1) for k in self.gph.nodes}
        dist_dict = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
        return dist_dict

    def calc_node_distance(self,type = 'soil'):
        """ calculate the distance to outlet of a particular node group (e.g. flood, green
        infrastructure) """
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
    
    def calc_hydrograph_example(self, to_node = None):
        """ example to build a hydrograph """
        if not to_node:
            to_node = self.outlet_node
        nodes = [n for n in nx.traversal.bfs_tree(self.gph, to_node, reverse=True) if n != to_node]
        each_path = {k: (len(nx.shortest_path(self.gph, source=k, target = to_node)) - 1) for k in nodes}
        unique_vals = set(each_path.values())
        a = [sum(value == v for value in each_path.values()) for v in unique_vals]
        return np.array(a)
    
    def calc_flow_path(self, accum_attr='length', path_attr_name=None):
        """ calculate flow paths lengths and travel time """
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
                        else: 
                            path_attr = 0
                            keep_running = False
                            break
                keep_running = False
            path_dict[node] = path_attr
        nx.set_node_attributes(self.gph, path_dict, path_attr_name)
        return path_dict
    
    def iter_nodes(self, nodes, nodes_to_search = None, dir = 'ds'):
        """ a method used to look for nodes interatively """
        big_group = {}
        us_nodes = {}
        ds_nodes = {}
        
        def edge_dir(node, dir):
            """Look for upstream incoming or downstream outgoing edge"""
            if dir == 'ds':
                return self.gph.out_edges(node)
            else: 
                return self.gph.in_edges(node)
        
        def search_ds_neighbor(leaf):
            n = 1
            edges = edge_dir(leaf,dir) # Look for downstream edge
            ds_nodes[leaf] = [edge[n] for edge in edges]
            gi_node = list(root for root in ds_nodes[leaf] if root in nodes)
            if len(gi_node) > 0:
                search_node = gi_node[0]
                root = search_ds_neighbor(search_node)
            else:
                root = leaf
                big_group[root] = 1
                return root
            return root

        def search_us_neighbor(root, to_add = 0):
            n = 0
            edges = edge_dir(root,'us') # Look for upstream edge
            us_nodes[root] = [edge[n] for edge in edges]
            gi_nodes = list(node for node in us_nodes[root] if node in nodes)
            to_add += len(gi_nodes)
            if len(gi_nodes) > 0:
                for gi_node in gi_nodes:
                    # nodes.remove(gi_node)
                    to_add = search_us_neighbor(gi_node, to_add=to_add)
            return to_add
        
        if nodes_to_search is None:
            nodes_to_search = nodes
        
        for node in nodes_to_search:
            if (node in us_nodes.values()) or (node in big_group.keys()):
                pass
            root = search_ds_neighbor(node)
            to_add = search_us_neighbor(root)
            big_group[root] += to_add
        return big_group

    def calc_node_clustering(self,type = 'soil'):
        """ calculate nodes degree of clustering (not used in paper) """
        clustering_coef = 0
        self.soil_node_cluster = []
        if type == 'flood':
            nodes = list(self.flood_nodes).copy()
        else: 
            nodes = list(self.soil_nodes).copy()
        if len(self.soil_nodes) > 0:
            big_group=self.iter_nodes(nodes=nodes,dir='ds')
            self.soil_node_cluster = big_group.values()
            cluster_hist = Counter(self.soil_node_cluster)
            norm_hist = {i: floor(len(self.soil_nodes)/i) for i in cluster_hist.keys()}
            clustering_coef = 1/len(self.soil_nodes) * sum(i*cluster_hist[i]/norm_hist[i] for i in cluster_hist.keys())
        return clustering_coef
    
    def make_dist_cluster(self, starting_dist):
        """ make clusters of green infrastructure nodes from a given distance """
        dist_dict = self.make_dist_dict()
        count = self.count

        def make_list(starting_dist):
            sub = [i for i in dist_dict if dist_dict[i] == starting_dist]
            if sub == []:
                return
            soil_nodes_list = sub
            return soil_nodes_list
        
        soil_nodes_list = make_list(starting_dist)
        new_starting_dist = starting_dist
        if count:
            while len(soil_nodes_list) < count:
                deficit = count - len(soil_nodes_list)
                new_starting_dist += 1
                try: 
                    soil_nodes_list = soil_nodes_list + list(set(make_list(new_starting_dist))-set(soil_nodes_list))[:deficit]

                except TypeError:
                    new_starting_dist = starting_dist
                    while len(soil_nodes_list) < count:
                        new_starting_dist = new_starting_dist - 1
                        soil_nodes_list = soil_nodes_list + list(set(make_list(new_starting_dist))-set(soil_nodes_list))[:deficit]
            soil_nodes_list = soil_nodes_list[:count]
        self.soil_nodes = soil_nodes_list
                
    def draw_network_init(self, ax = None, label_on = False, title = None):
        """ draw the network flow and the dispersion coefficients at a single timestep. """
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
                node_color.append('#a5d1f0')
                node_size.append(node_size_og)

        fig = plt.figure()
        gs = fig.add_gridspec(3,2)
        plt.suptitle(rf'{self.n} by {self.n} network with $\beta$ = {self.beta}')

        ax0 = fig.add_subplot(gs[:-1,0])
        pos_grid = {k: (floor(k/self.n), k%self.n) for k in self.gph.nodes()}
        nx.draw(self.gph, pos_grid, ax0, node_color=node_color, node_size=node_size, edge_color='lightgrey')
        
        ax1 = fig.add_subplot(gs[:-1,1])
        pos = graphviz_layout(self.gph, prog = 'dot')
        nx.draw(self.gph, pos, ax1, node_color = node_color, node_size = node_size,labels = node_label, 
        font_size=6,edge_color='lightgrey',with_labels = label_on)
        
        ax2 = fig.add_subplot(gs[2,:])
        k = self.downstream_degree_to_outlet
        distance_dist = [k[j] for j in k]
        ax2.grid(alpha=0.1)
        bin_spacing_dist = list(np.linspace(0,max(distance_dist),max(distance_dist)+1, endpoint=True,dtype=int))
        ax2.hist(distance_dist, bins = bin_spacing_dist, align='left')
        ax2.set_xlabel('Dist. to Outlet')
        ax2.set_ylabel('Count')
        
        plt.subplots_adjust(wspace=0)

        if title:
            plt.suptitle(title)
            
            
    def draw_network_timestamp(self, edge_attribute = 'edge_dq', label_on = False, flood_level = 10, title = None):
        """ draw the network flow and the dispersion coefficients at a single timestep. """
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

        cmap = plt.cm.RdBu
        vmin = min(min(edge_color.values()),-max(edge_color.values()))
        vcenter = 0
        vmax = max(max(edge_color.values()),-min(edge_color.values()))
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        fig0, ax0 = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 5]})
        nx.draw(self.gph, pos, ax0[1], node_color = node_color, node_size = node_size, 
        edge_color = [sm.to_rgba(i) for i in edge_color.values()], labels = node_label, 
        font_size=6,with_labels = label_on, edge_cmap = cmap)
        plt.colorbar(sm, cax = ax0[0])
        ax0[0].set_xlabel('Edge dQ (cfs)')
        ax0[0].set_ylim(bottom = vmin,top=vmax)
        if title:
            plt.suptitle(title)

    def graph_histogram(self, kernel=None, ax=None):
        """ plot degree distribution of the graph with option for kernal density plot """
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
            ax1 = ax.twinx()
            ax1.plot(bin_list, kernel(bin_list), 'k-o', label='Poisson Distribution')

    
# other utility codes
def round_pipe_diam(old_diam):
    """ this is to round up pipes to common size """
    common_sizes = [12, 15, 18, 21, 24, 30, 33, 36, 42, 48, 54, 60, 66, 72, 78]
    old_inch = old_diam * 12
    
    for size in common_sizes:
        if old_inch <= size:
            return size / 12
    raise ValueError('Pipe size is larger than common size.')

if __name__ == '__main__':
    
    file_name = r'./example/10-grid_beta-0.5_dist-128_ID-0x10b3f48e0>.pickle' # example
    a = Storm_network(file_name = file_name, nodes_num=100,count=0,changing_diam=True)
    a.draw_network_init(label_on=True)
    plt.show()
