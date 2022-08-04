#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:49:28 2021

@author: xuefeng, chenxiating
"""
import random
from tracemalloc import start
import numpy as np
import glob
import networkx as nx
import matplotlib.pyplot as plt
import math
import time
import datetime as date
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
import pandas as pd
import os
from multiprocessing import current_process, cpu_count


def handler(signum, frame):
    print("Breaking while loop now!")
    raise Exception("End of time")

class Uniform_network:
    def __init__(self, m, n, beta, deltaH = None, mode='Gibbs',outlet_point = None, input_matrix = None): 
        """ m:      number of rows (indexed by i)
            n:      number of columns (indexed by j)
            (i,j)   coordinates gives item (1+j)+i*n in adjacency matrix"""
        
        self.m = m
        self.n = n
        self.beta = beta
        self.deltaH = deltaH
        if mode == 'Uniform':
            self.beta = 0
        
        # initialize adjacency matrix
        self.matrix = np.zeros((m*n, m*n))
        
        # initialize all the nodes laid out on a grid 
        grid = [(i,j) for i in range(m) for j in range(n)]
        self.grid_nodes = grid
        
        # label for "open" points that have not been visited before
        grid_copy = self.grid_nodes.copy()
        self.open_nodes = grid_copy 

        # label for "open" edges that have not been visited before
        possible_edges = []
        for vi, vj in grid_copy:
            possible_next_pts = [(vi+1, vj), (vi-1, vj), (vi, vj+1), (vi, vj-1)]
            pts = [pt for pt in possible_next_pts if (pt in set(self.grid_nodes))] # eliminate points outside of grid
            possible_edges.append(list((self.convert_ij(vi, vj), self.convert_ij(*k)) for k in pts))
        list_set = set(edge for sublist in possible_edges for edge in sublist)
        
        if input_matrix is not None:
            self.matrix = input_matrix
            self.path_diff = self.calculate_path_diff(self.matrix)
        else: 
            self.possible_edges = list(list_set)
            self.outlet_point=outlet_point
            self.generate_tree()
            # k = 5*self.n**3 
        self.path_diff = self.calculate_path_diff(self.matrix)
        self.find_outlet_point()
#       # number of adjacency nodes 
#        self.n_adjacent = 4 * np.ones((m,n))
#        self.n_adjacent[0,:], self.n_adjacent[-1,:], self.n_adjacent[:,0], self.n_adjacent[:,-1] = 3,3,3,3
#        self.n_adjacent[0,0], self.n_adjacent[-1,-1], self.n_adjacent[0,-1], self.n_adjacent[-1,0] = 2,2,2,2
    
    def convert_ij(self, i,j): 
        """converts (i,j) coordinates on a matrix to corresponding index on adjacency matrix
        add 1 if starting index at 1 instead of 0"""
        return j+i*self.n
    
    def convert_index(self, k):
        """convert matrix index to (i,j)  coordinates"""
        return (math.floor(k/self.n), k%self.n)

    def random_next_point(self, vi, vj):
        """generates the next point within the grid via random walk 
        determine possible destination nodes and assign probabilities"""
        possible_next_pts = [(vi+1, vj), (vi-1, vj), (vi, vj+1), (vi, vj-1)]

        # need to check that pt are in grid
        pts = [pt for pt in possible_next_pts if (pt in set(self.grid_nodes))] # eliminate points outside of grid
        rv = len(pts) # total number of adjacent nodes
        # simulate direction of movement
        q = random.random() # random number from 0 to 1
        # upper bounds of quantiles. For example, for rv = 3, this generates (0.33, 0.66, 1)
        upper_bounds = [i/rv for i in range(1,rv+1)] 
        ipt = np.where(q < np.array(upper_bounds))[0][0] # first exceedance instance for upper bound
    
        # find next point 
        next_point = pts[ipt]
        return next_point

    def open_edges(self):
        """Finds open edges in the network"""
        # possible edges on a grid
        possible_edges = self.possible_edges

        # matrix to show existing edges, regardless of direction
        current_edges_matrix = self.matrix + np.transpose(self.matrix)
        current_edge_arr = np.transpose(np.where(current_edges_matrix == 1))
        current_edges = list(map(tuple,current_edge_arr))

        # removing edges that start from the outlet point]
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        outlet_point_k = self.convert_ij(self.outlet_point[0],self.outlet_point[1])
        impossible_edges = {(v_outlet, v_ng) for (v_outlet, v_ng) in possible_edges if v_outlet == outlet_point_k}
        # print(impossible_edges) # DEBUG!

        # edges where a new edge can be added to create loop
        open_edges = list(set(possible_edges) - set(current_edges) - set(impossible_edges))

        return open_edges 

    def random_next_edge(self):
        """Generates an edge within the grid randomly, after an edge is removed
        from the grid in the process of generating network approcahing Gibbs dist. 
        Generate new tree s2 from s1 and check whether s2 is a tree"""

        # edges where a new edge can be added to create loop, return a list of edges
        open_edges = self.open_edges()
        random.shuffle(open_edges)
        outlet_point_k = self.convert_ij(self.outlet_point[0],self.outlet_point[1])
        # add an edge to create loop
        for v_start, v_end in open_edges:
            # two_edges_to_try = [[v_start, v_end], [v_end, v_start]]
        # for v0, v1 in two_edges_to_try:
            # self.draw_tree(dist_label=False) # DEBUG!
            self.matrix[v_start, v_end] = 1
            # check where the loop is in graph
            G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
            # loops = nx.find_cycle(G, orientation="ignore")
            # print(f'edge to try: {v_start}, {v_end}') # DEBUG!
            
            
            [(v_from, v_to)] = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_from == v_start) and (v_to != v_end)]
            ### DEBUG! 
            # if test_list == []:
                # self.draw_tree(dist_label=False) # DEBUG!
                # plt.show() # DEBUG!
            # try:
            #     [(v_from, v_to)] = test_list
            # except ValueError:
            #     print(f'added: {v_start}, {v_end}')
            #     print(test_list) 
            ### DEBUG! 
            # try: 
            #     test_list = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_from == v0) and (v_to != v1)]
            #     print(test_list)
            #     [(v_from, v_to)] = test_list
            # except (IndexError, ValueError, UnboundLocalError):
            #     in_outlet = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_to == v0)]
            #     random.shuffle(in_outlet)  
            #     print(f'v0: {v0}, v1: {v1}, v_from: {v_from}, v_to: {v_to}')
            #     (v_from, v_to) = random.choice(in_outlet)
                # print((v_from, v_to))
            ### END DEBUG!                
            
            # random.shuffle(loops)
            # select an old edge within the loop to be removed
            # for v_from, v_to in loops:
            self.matrix[v_from, v_to] = 0 # remove an edge in the loop
            # break                
            # if nx.find_cycle(G, orientation="ignore"):
            if self.is_tree_structure():
                # self.draw_tree(title = f'Success! ({v0}, {v1}) is added, ({v_from}, {v_to}) is removed')
                # self.draw_tree(title=f'Added ({v0}, {v1}), deleted ({v_from}, {v_to})')
                # plt.show()
                # print('is tree!') # DEBUG!
                r = 0
                vi, vj = self.convert_index(v_from)
                possible_next_pts = [(vi+1, vj), (vi-1, vj), (vi, vj+1), (vi, vj-1)]
                pts = [pt for pt in possible_next_pts if (pt in set(self.grid_nodes))]
                # # draw tree here
                # self.draw_tree(node_to_color=v_from)
                # plt.show()
                for x, y in pts:
                    new_v_to = self.convert_ij(x, y)
                    # print(f'From ({vi}, {vj}) to ({x}, {y}), the matrix value is {self.matrix[v_from, new_v_to]  + self.matrix[new_v_to, v_from]}')
                    r = r + self.matrix[v_from, new_v_to] + self.matrix[new_v_to, v_from]
                r = len(pts) - r
                # print(f'r is {r}')

                return r
            else:
                # if the new tree doesn't retain old structure, add the edge back in
                # self.draw_tree(dist_label=False) # DEBUG!
                # print('Redo!') # DEBUG!
                self.matrix[v_from, v_to] = 1 
                self.matrix[v_start, v_end] = 0
                # self.draw_tree(dist_label=False) # DEBUG!

                
        self.random_next_edge()
        
    def generate_branch(self, current_point):
        """recursive function for branch generation
        generate next point from given point"""
        next_point = self.random_next_point(*current_point)
        # print('Next point:', *next_point, (next_point in self.open_nodes),'Number of open nodes:',len(self.open_nodes))
        
        # return points in matrix order & update adjacency matrix
        v0 = self.convert_ij(*current_point) # in matrix order
        v1 = self.convert_ij(*next_point)
        
        try:
            # test to see if next_point is open
            self.open_nodes.remove(next_point)
        except ValueError:  
            # this means that it hit an existing tree - this will be ignored
            pass
        else: # have not hit an existing tree
            # update adjacency matrix 
            self.matrix[v1,v0] = 1
        return next_point

    def find_outlet_point(self):
        G = nx.from_numpy_matrix(self.matrix,create_using=nx.DiGraph)
        outlet_point_k = [n for n, d in G.out_degree() if d == 0][0]
        self.outlet_point = self.convert_index(outlet_point_k)
    
    def calculate_path_diff(self, input_matrix):
        """calculate the difference between the mapped path length and shortest path length"""
        G = nx.from_numpy_matrix(input_matrix,create_using=nx.DiGraph)
        outlet_point_k = [n for n, d in G.out_degree() if d == 0][0]
        self.outlet_point = self.convert_index(outlet_point_k)
        total_path = sum([len(nx.shortest_path(G, source = k, target = outlet_point_k)) - 1 for k in G.nodes])
        # outlet_x, outlet_y = self.convert_index(self.outlet_point)
        grid_nodes = self.grid_nodes
        shortest_path = sum((abs(x - self.outlet_point[0]) + abs(y - self.outlet_point[1])) for x, y in grid_nodes)
        # # check path calculations
        # for x, y in grid_nodes:
        #     print(f'Outlet node: ({outlet_x}, {outlet_y}). Grid node: ({x}, {y})')
        #     print('Path distance: ',len(nx.shortest_path(G, source = self.convert_ij(x, y), target = outlet)) - 1)
        #     print('Shortest distance: ',(abs(x - outlet_x) + abs(y - outlet_y)))
        diff = total_path - shortest_path
        if diff < 0:
            raise Exception("Diff is negative!")
        return diff

    def generate_Gibbs(self):
        """reiterate steps to make one Gibbs graph"""
        """k: number of trees"""
        """deltaH_list: records all the H iterated while generating graph"""
        deltaH_list = []
        # burntime=3*k/4
        burntime = 0.5*self.n**3 
        i = 0
        # if k < burntime: 
        #     raise ValueError('Iteration number is less than burntime.')
        #for i in range(k):
        while i <= burntime:
            i +=1
            s1_matrix = self.matrix.copy()
            r = self.random_next_edge()
            s2_matrix = self.matrix.copy()
            H_diff = float(self.calculate_path_diff(s2_matrix) - self.calculate_path_diff(s1_matrix))
            threshold = 1/r*min(1, np.exp(-self.beta*H_diff))

            # x is a unif var to decide whether to take s2_matrix as the new network
            x = random.random()
            if x > threshold:
                self.matrix = s1_matrix
                if self.deltaH:
                    # deltaH_threshold = self.n * self.m/10
                    deltaH_threshold = 1
                    path_diff = self.calculate_path_diff(self.matrix)
                    if (path_diff > (self.deltaH - deltaH_threshold)) & (path_diff < (self.deltaH + deltaH_threshold)):
                        deltaH_list.append(self.calculate_path_diff(self.matrix))
                        print(f"We are done here! Delta H = {self.deltaH}")
                        break
            # if i > burntime: 
            deltaH_list.append(self.calculate_path_diff(self.matrix))
        return np.array(deltaH_list)

    def is_tree_structure(self):
        """check if the tree retains original structure after 'minimal change' process """
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        return nx.is_tree(G) and max(d for n, d in G.out_degree()) <= 1
    

    def generate_tree(self, export=False):
        """do random walk from given point until boundary is hit, or until all nodes have been visited
        initialize first point, generate tree until hitting dead end, initialize next point"""
        start = time.perf_counter()
        grid_copy = self.grid_nodes.copy()
        self.open_nodes = grid_copy

        #initialize first point - this will be the outlet point
        if not self.outlet_point:
            self.outlet_point = random.choice(self.open_nodes)
        
        # update open_nodes 
        # print(self.open_nodes, self.outlet_point)
        self.open_nodes.remove(self.outlet_point)
        
        # build the first branch first 
        first_point = self.outlet_point
        
        while len(self.open_nodes)>0: 
            # print('###### Open nodes left: ###### ', len(self.open_nodes))
            # generate branch
            next_point = self.generate_branch(first_point)
            first_point = next_point
        
        finish = time.perf_counter()
        # print(f'{self.m} by {self.n} uniform graph, finished in {round(finish-start,2)} seconds(s)')
        self.deltaH_list = self.generate_Gibbs()
            # self.generate_Gibbs(k=k)
        self.path_diff = self.calculate_path_diff(self.matrix)
        finish = time.perf_counter()
        cp = str(current_process())
        cp_name = cp[cp.find('name=')+6:cp.find(' parent=')-1]
        print(f'beta = {self.beta} Gibbs graph with path diff = {self.path_diff}, finished in {round(finish-start,2)} seconds(s) at {cp_name}')
        if self.deltaH: 
            deltaH_threshold = self.n * self.m/10
            if (self.calculate_path_diff(self.matrix) > (self.deltaH - deltaH_threshold)) & (self.calculate_path_diff(self.matrix) < (self.deltaH + deltaH_threshold)):
                # print(f'{k}-trees beta = {self.beta} Gibbs graph with path diff = {self.path_diff}, finished in {round(finish-start,2)} seconds(s)')
                print('We are exporting trees here!') # DEBUG!
                self.export_tree()
            else: 
                print(f'We did not reach the threshold of {self.deltaH} +/- {deltaH_threshold}. We are generating trees here!') # DEBUG!
                # self.generate_tree()
        elif export:
            print('Exporting tree')
            self.export_tree()
        print(f'Path difference is {self.path_diff}')

        # ### DEBUG!!
        # try:
        #     self.deltaH_list = self.generate_Gibbs(k=k)
        #     # self.generate_Gibbs(k=k)
        #     self.path_diff = self.calculate_path_diff(self.matrix)
        #     finish = time.perf_counter()
        #     cp = str(current_process())
        #     cp_name = cp[cp.find('name=')+6:cp.find(' parent=')-1]
        #     print(f'{k}-trees beta = {self.beta} Gibbs graph with path diff = {self.path_diff}, finished in {round(finish-start,2)} seconds(s) at {cp_name}')
        #     if self.deltaH: 
        #         deltaH_threshold = self.n * self.m/10
        #         if (self.calculate_path_diff(self.matrix) > (self.deltaH - deltaH_threshold)) & (self.calculate_path_diff(self.matrix) < (self.deltaH + deltaH_threshold)):
        #             # print(f'{k}-trees beta = {self.beta} Gibbs graph with path diff = {self.path_diff}, finished in {round(finish-start,2)} seconds(s)')
        #             self.export_tree()
        #         else: 
        #             self.generate_tree(k=k)
        #     elif export:
        #         print('Exporting tree')
        #         self.export_tree()
        #     print(f'Path difference is {self.path_diff}')
        # except RecursionError:
        #     self.open_nodes = grid_copy
        #     self.matrix = np.zeros((self.m*self.n, self.m*self.n))
        #     print('Generating new tree')
        #     self.generate_tree(k=k) 
        # ### END DEBUG!!
        return self.path_diff

    def draw_tree(self,title=None,dist_label=True,save=False,starting_coord=None):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis('off')
        small_node_size = 50
        outlet_point_k = self.convert_ij(self.outlet_point[0],self.outlet_point[1])
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        node_color_dict = {node: 'C0' for node in G.nodes}
        node_size_dict = {node: small_node_size for node in G.nodes}
        node_color_dict[outlet_point_k] = 'C1'
        node_size_dict[outlet_point_k] = small_node_size*5
        node_color = list(node_color_dict.values())
        node_size = list(node_size_dict.values())
        pos_grid = {k: self.convert_index(k) for k in G.nodes()}
        path_to_highlight = []
        nx.draw_networkx_nodes(G, pos=pos_grid, node_color=node_color, node_size=node_size)
        if dist_label:
            node_labels = {}
            for node in G.nodes:
                node_xy = self.convert_index(node)
                actual = len(nx.shortest_path(G, source = node, target = outlet_point_k)) - 1
                shortest = abs(node_xy[1] - self.outlet_point[1]) + abs(node_xy[0] - self.outlet_point[0])
                # node_labels[node] = f'{actual} - {shortest} \n = {actual-shortest}'
                if actual-shortest == 0: 
                    label = ''
                else:
                    label = actual-shortest
                    # print(str(self.convert_index(node)),actual,shortest)
                node_labels[node] = f'{label}'
            label_grid = {node: (pos_grid[node][0]+0.1, pos_grid[node][1]+0.1) for node in G.nodes}
            nx.draw_networkx_labels(G,pos=label_grid,labels=node_labels,font_size = 20)
        
        if starting_coord:
            if type(starting_coord) is tuple:
                starting_coord = self.convert_ij(starting_coord[0],starting_coord[1])
            elif type(starting_coord) is int:
                pass
            else:
                TypeError('Starting coordinate is neither tuple nor integer.')
            passing_nodes = nx.shortest_path(G, source = starting_coord, target = outlet_point_k)
            path_to_highlight = [(passing_nodes[x], passing_nodes[x+1]) for x in range(len(passing_nodes)-1)]
            nx.draw_networkx_edges(G,pos=pos_grid,edgelist=path_to_highlight,width=4,edge_color='r')
            nx.draw_networkx_nodes(G, pos=pos_grid, nodelist=[starting_coord], node_color='r',edgecolors='w', node_size=small_node_size*5)
        
        nx.draw_networkx_edges(G, pos=pos_grid,edgelist=G.edges - path_to_highlight,width=1,edge_color='grey')

         #, with_labels = True)
        # plt.subplot(122)
        # pos_gv = graphviz_layout(G, prog = 'dot')
        # #nx.draw(G, pos=nx.planar_layout(G), node_size=10, edge_color='lightgrey')
        # nx.draw(G, pos=pos_gv, node_color=node_color, node_size=node_size, edge_color='grey') #, with_labels=True)
        # if dist_label:
        #     label_grid_gv = {node: (pos_gv[node][0], pos_gv[node][1]+1) for node in G.nodes}
        #     nx.draw_networkx_labels(G,pos=label_grid_gv,labels=node_labels,font_size = 20)
        plt.title(title)
        if save:
            plt.savefig(f'./tree_size{self.n}by{self.m}_beta{self.beta}.png')
    
    def draw_tree_with_distance(self,title=None,dist_label=True,save=False):
        fig = plt.figure(figsize=(12,6))
        # gs = fig.add_gridspec(3,6)
        small_node_size = 50
        arrowsize = int(30)
        outlet_point_k = self.convert_ij(self.outlet_point[0],self.outlet_point[1])
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        node_color_dict = {node: 'C0' for node in G.nodes}
        node_size_dict = {node: small_node_size for node in G.nodes}
        # node_color_dict.update({node: 'C2' for node in [1, 3, 5, 10, 16, 20, 22]})
        # print(node_color_dict)
        # node_size_dict.update({node: small_node_size*5 for node in [1, 3, 5, 10, 16, 20, 22]})
        node_color_dict[outlet_point_k] = 'C1'
        node_size_dict[outlet_point_k] = small_node_size*5
        node_color = list(node_color_dict.values())
        node_size = list(node_size_dict.values())
        # plt.subplot(121)
        # ax1 = fig.add_subplot(gs[:3,:3])
        ax1 = plt.subplot(121)
        pos_grid = {k: self.convert_index(k) for k in G.nodes()}
        nx.draw(G, pos=pos_grid, node_color=node_color, node_size=node_size, edge_color='grey', arrowsize=arrowsize) #, with_labels = True)
        if dist_label:
            node_labels = {}
            for node in G.nodes:
                node_xy = self.convert_index(node)
                actual = len(nx.shortest_path(G, source = node, target = outlet_point_k)) - 1
                shortest = abs(node_xy[1] - self.outlet_point[1]) + abs(node_xy[0] - self.outlet_point[0])
                # node_labels[node] = f'{actual} - {shortest} \n = {actual-shortest}'
                # if actual-shortest == 0: 
                #     label = ''
                # else:
                label = actual-shortest
                    # print(str(self.convert_index(node)),actual,shortest)
                node_labels[node] = f'{label}'
            label_grid = {node: (pos_grid[node][0]+0.2, pos_grid[node][1]+0.2) for node in G.nodes}
            nx.draw_networkx_labels(G,pos=label_grid,labels=node_labels,font_size = 30)
        
        # # plt.subplot(122)
        # ax2 = fig.add_subplot(gs[:3,3:])
        # pos_gv = graphviz_layout(G, prog = 'dot')
        # #nx.draw(G, pos=nx.planar_layout(G), node_size=10, edge_color='lightgrey')
        # nx.draw(G, pos=pos_gv, node_color=node_color, node_size=node_size, edge_color='grey') #, with_labels=True)
        # # if dist_label:
        # #     label_grid_gv = {node: (pos_gv[node][0], pos_gv[node][1]+1) for node in G.nodes}
        # #     nx.draw_networkx_labels(G,pos=label_grid_gv,labels=node_labels,font_size = 20)
        
        # ax3 = fig.add_subplot(gs[:4,-5:])
        # distance_dist = [len(nx.shortest_path(G, source = node, target = outlet_point_k)) - 1 for node in G.nodes]
        # ax3.grid(alpha=0.1)
        # bin_spacing_dist = list(np.linspace(0,max(distance_dist),max(distance_dist)+1, endpoint=True,dtype=int))
        # ax3.hist(distance_dist, bins = bin_spacing_dist, align='left')
        # ax3.set_xlabel('Node distance to outlet')
        # ax3.set_ylabel('Count')

        # ax3 = fig.add_subplot(gs[:3,-3:])
        ax3 = plt.subplot(122)
        max_length = max(len(nx.shortest_path(G, source = node, target = outlet_point_k)) - 1 for node in G.nodes)
        # plt.subplot(121)
        pos_grid = {k: self.convert_index(k) for k in G.nodes()}
        
        if dist_label:
            node_labels = {}
            for node in G.nodes:
                actual = len(nx.shortest_path(G, source = node, target = outlet_point_k)) - 1
                if actual == max_length:
                    node_labels[node]=max_length
                    node_color_dict[node] = 'red'
                    node_size_dict[node] = small_node_size*5   
            label_grid = {node: (pos_grid[node][0]+0.2, pos_grid[node][1]+0.2) for node in G.nodes}
            nx.draw_networkx_labels(G,ax=ax3, pos=label_grid,labels=node_labels,font_size = 30)            
            
        node_color = list(node_color_dict.values())
        node_size = list(node_size_dict.values())
        nx.draw(G, pos=pos_grid, node_color=node_color, node_size=node_size, edge_color='grey', arrowsize=arrowsize) 
        
        if save:
            plt.savefig(f'./tree_size{self.n}by{self.m}_beta{self.beta}.png')
    
    def draw_tree_with_distance_label(self,title=None,dist_label=True,save=False):
        fig = plt.figure(figsize=(6,6))
        small_node_size = 50
        outlet_point_k = self.convert_ij(self.outlet_point[0],self.outlet_point[1])
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        node_color_dict = {node: 'C0' for node in G.nodes}
        node_size_dict = {node: small_node_size for node in G.nodes}
        node_color_dict[outlet_point_k] = 'C1'
        node_size_dict[outlet_point_k] = small_node_size*5
    
        max_length = max(len(nx.shortest_path(G, source = node, target = outlet_point_k)) - 1 for node in G.nodes)
        # plt.subplot(121)
        pos_grid = {k: self.convert_index(k) for k in G.nodes()}
        
        if dist_label:
            node_labels = {}
            for node in G.nodes:
                actual = len(nx.shortest_path(G, source = node, target = outlet_point_k)) - 1
                if actual == max_length:
                    node_labels[node]=max_length
                    node_color_dict[node] = 'red'
                    node_size_dict[node] = small_node_size*5   
            label_grid = {node: (pos_grid[node][0]+0.2, pos_grid[node][1]+0.2) for node in G.nodes}
            nx.draw_networkx_labels(G,pos=label_grid,labels=node_labels,font_size = 20)            
        
        node_color = list(node_color_dict.values())
        node_size = list(node_size_dict.values())
        nx.draw(G, pos=pos_grid, node_color=node_color, node_size=node_size, edge_color='grey') #, with_labels = True)

        if save:
            plt.savefig(f'./tree_size{self.n}by{self.m}_beta{self.beta}.png')
    
    def export_tree(self, i=0, name = None):
        if not name:
            self_name = str(self)
            ID = self_name[self_name.find('x')-1:]
            name = f'{self.n}-grid_beta-{self.beta}_dist-{self.path_diff}_ID-{ID}.pickle'
        f = open(name,'wb')
        pickle.dump(self.matrix,f)
        f.close()
        # self.draw_tree(save=True)

    def compile_datasets(self,folder_name = None):
        if folder_name:
            os.chdir(folder_name)
        deltaH_list = []
        for one_file in glob.glob('*.pickle'):
            grid = pickle.load(open(one_file, 'rb'))
            deltaH = self.calculate_path_diff(grid)
            deltaH_list.append(deltaH)
        plt.hist(np.array(deltaH_list))
        plt.xlabel('$\Delta$ H')
        plt.ylabel('Count')
        plt.show()
        return np.array(deltaH_list)

def gibbs_pdf(beta,all_deltaH_list):
    fig = plt.figure()
    plt.suptitle(rf'Distribution of {int(len(all_deltaH_list))} random Gibbs trees with $\beta$ = {beta}')
    gs = fig.add_gridspec(1,5)
    ax1 = fig.add_subplot(gs[0,:-1])
    ax2 = fig.add_subplot(gs[0,-1])
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    for deltaH_list in all_deltaH_list:
        ax1.plot(deltaH_list,alpha = max(0.1, 1/len(all_deltaH_list)), color = 'C0')
    ax1.set_ylabel('$\Delta H$')
    ax1.set_xlabel('Iteration')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    new_list = [elem for a0 in all_deltaH_list for elem in a0]
    ax2.hist(new_list,density=True,orientation='horizontal')#,bins=30,)
    ax2.axis('off')
    # ax2.spines['bottom'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.get_yaxis().set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.xaxis.set_label_position("top")
    # ax2.xaxis.tick_top()
    # ax2.set_xlabel('Frequency')
    # ax2.grid()
    # ax2.set_ylabel('$\Delta H$')
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position("right")
    # ax2.set_xlabel('Frequency')
    plt.subplots_adjust(wspace=0)
    plt.savefig(f'./dist_beta{beta}.png')

def test(size, beta=0.5, tree_num = 1000):
    start = time.perf_counter()    
    all_deltaH_list = []
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    for i in range(tree_num):
        uni = Uniform_network(size, size, beta=beta,mode="Gibbs")
        # ax1.plot(deltaH_list,alpha = 0.1, color = 'C0')
        all_deltaH_list.append(uni.deltaH_list)
    uni.export_tree(i = i)
    gibbs_pdf(uni.beta,all_deltaH_list)
    name = f'deltaH_beta{uni.beta}.pickle'
    f = open(name,'wb')
    pickle.dump(all_deltaH_list,f)
    f.close()
    finish = time.perf_counter()
    print(f'{size} by {size} final graph, finished in {round(finish-start,2)} seconds')
    return uni

def main(size, beta, outlet_point, mode = "Gibbs",input_matrix = None):
    gibbs = Uniform_network(m=size, n=size, beta=beta, outlet_point=outlet_point, mode = mode, input_matrix=input_matrix)
    return gibbs

#%%
if __name__ == '__main__':
    random.seed(5)
    # path_diff_list = []
    for beta_val in [0.01,0.9]:
        gibbs = Uniform_network(5, 5, beta=beta_val, outlet_point = (0,0), mode='Gibbs')
        gibbs.draw_tree(dist_label=False, starting_coord=(3,3))
        plt.savefig(f'/Volumes/GoogleDrive/My Drive/urban-stormwater-analysis/writing/GI_network/figures/demo_beta_{beta_val}.png')
    plt.show()