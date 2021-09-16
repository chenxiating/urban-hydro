#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:49:28 2021

@author: xuefeng
"""

from multiprocessing import Value
from os import error
import random
from networkx.classes.digraph import DiGraph
import numpy as np
import glob
import networkx as nx
import matplotlib.pyplot as plt
import math
import time
import datetime as date
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
import signal
import pandas as pd
import os

from numpy.lib.function_base import delete

def handler(signum, frame):
    print("Breaking while loop now!")
    raise Exception("End of time")

class Uniform_network:
    
    def __init__(self, m, n, beta): 
        # m: number of rows (indexed by i)
        # n: number of columns (indexed by j)
        # (i,j) coordinates gives item (1+j)+i*n in adjacency matrix
        self.m = m
        self.n = n
        self.beta = beta
        
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
        self.possible_edges = list(list_set)

#        # number of adjacency nodes 
#        self.n_adjacent = 4 * np.ones((m,n))
#        self.n_adjacent[0,:], self.n_adjacent[-1,:], self.n_adjacent[:,0], self.n_adjacent[:,-1] = 3,3,3,3
#        self.n_adjacent[0,0], self.n_adjacent[-1,-1], self.n_adjacent[0,-1], self.n_adjacent[-1,0] = 2,2,2,2

    def convert_ij(self, i,j): 
        """converts (i,j) coordinates on a matrix to corresponding index on adjacency matrix
        add 1 if starting index at 1 instead of 0"""
        return j+i*self.n
    
    def convert_index(self, k):
        # convert matrix index to (i,j)  coordinates
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
        # possible edges on a grid
        possible_edges = self.possible_edges
        # matrix to show existing edges, regardless of direction
        current_edges_matrix = self.matrix + np.transpose(self.matrix)
        current_edge_arr = np.transpose(np.where(current_edges_matrix == 1))
        current_edges = list(map(tuple,current_edge_arr))
        # edges where a new edge can be added to create loop
        open_edges = list(set(possible_edges) - set(current_edges))
        return open_edges 

    def random_next_edge(self):
        """generates an edge within the grid randomly, after an edge is removed
        from the grid in the process of generating network approcahing Gibbs. 
        Generate new tree s2 from s1 and check whether s2 is a tree"""

        # edges where a new edge can be added to create loop, return a list of edges
        open_edges = self.open_edges()
        random.shuffle(open_edges)
        # add an edge to create loop
        for v_start, v_end in open_edges:
            two_edges_to_try = [[v_start, v_end], [v_end, v_start]] 
            for v0, v1 in two_edges_to_try:
                self.matrix[v0, v1] = 1
                # check where the loop is in graph
                G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
                # loops = nx.find_cycle(G, orientation="ignore")
                try: 
                    [(v_from, v_to)] = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_from == v0) and (v_to != v1)]
                except ValueError:
                    in_outlet = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_to == v0)]
                    random.shuffle(in_outlet)  
                    (v_from, v_to) = random.choice(in_outlet)
                # random.shuffle(loops)
                # select an old edge within the loop to be removed
                # for v_from, v_to in loops:
                self.matrix[v_from, v_to] = 0 # remove an edge in the loop
                # if nx.find_cycle(G, orientation="ignore"):

                if self.tree_structure_check():
                    # self.draw_tree(title = f'Success! ({v0}, {v1}) is added, ({v_from}, {v_to}) is removed')
                    # self.draw_tree(title=f'Added ({v0}, {v1}), deleted ({v_from}, {v_to})')
                    # plt.show()
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
                    self.matrix[v_from, v_to] = 1 
                    self.matrix[v0, v1] = 0
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

    def calculate_path_diff(self, input_matrix):
        """calculate the difference between the mapped path length and shortest path length"""
        G = nx.from_numpy_matrix(input_matrix,create_using=nx.DiGraph)
        outlet = [n for n, d in G.out_degree() if d == 0][0]
        total_path = sum([len(nx.shortest_path(G, source = k, target = outlet)) - 1 for k in G.nodes])
        outlet_x, outlet_y = self.convert_index(outlet)
        grid_nodes = self.grid_nodes
        shortest_path = sum((abs(x - outlet_x) + abs(y - outlet_y)) for x, y in grid_nodes)
        # # check path calculations
        # for x, y in grid_nodes:
        #     print(f'Outlet node: ({outlet_x}, {outlet_y}). Grid node: ({x}, {y})')
        #     print('Path distance: ',len(nx.shortest_path(G, source = self.convert_ij(x, y), target = outlet)) - 1)
        #     print('Shortest distance: ',(abs(x - outlet_x) + abs(y - outlet_y)))
        diff = total_path - shortest_path
        if diff < 0:
            raise Exception("Diff is negative!")
        return diff

    def generate_Gibbs(self, k, burntime = 3000):
        """reiterate steps to make one Gibbs graph"""
        j = 0
        deltaH_list = []
        if k < burntime: 
            raise ValueError('Iteration number is less than burntime.')
        for i in range(k):
            s1_matrix = self.matrix.copy()
            r = self.random_next_edge()
            s2_matrix = self.matrix.copy()
            H_diff = self.calculate_path_diff(s2_matrix) - self.calculate_path_diff(s1_matrix)
            threshold = 1/r*min(1, np.exp(-self.beta*H_diff))

            # decide whether to take x as the new network
            x = random.random()
            if x > threshold:
                self.matrix = s1_matrix                
            if i > burntime: 
                deltaH_list.append(self.calculate_path_diff(self.matrix))
            j = j + 1
        return np.array(deltaH_list)

    def tree_structure_check(self):
        """check if the tree retains original structure after 'minimal change' process """
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        return nx.is_tree(G) and max(d for n, d in G.out_degree()) <= 1
    
    # def calculate_norm_coef(self):
    #     """calculate the normalization constant (norm_coef) and all possible degrees (r) for all adjacent graphs in a tree"""
    #     """8/4: actually wrong. we are not only interested in adjacent graphs, but also all other graphs."""
    #     # edges where a new edge can be added to create loop
    #     s1_matrix = self.matrix.copy()
    #     open_edges = self.open_edges()
    #     norm_coef = 0
    #     H_diff = self.calculate_path_diff(self.matrix)
    #     exp_coef = np.exp(-self.beta*(H_diff))
    #     norm_coef = norm_coef + exp_coef
    #     # print(f'Delta H is {round(H_diff,2)}, and normalization coef now is {round(norm_coef,2)}.')
    #     # self.draw_tree2(s1_matrix=s1_matrix)
    #     # plt.show()
    #     r = 1
    #     for v0, v1 in open_edges:
    #         self.matrix[v0, v1] = 1
    #         # check where the loop is in graph
    #         G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
    #         # loops = nx.find_cycle(G, orientation="ignore")
    #         try: 
    #             [(v_from, v_to)] = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_from == v0) and (v_to != v1)]
    #         except ValueError:
    #             in_outlet = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_to == v0)]
    #             random.shuffle(in_outlet)  
    #             (v_from, v_to) = random.choice(in_outlet)
    #         # select an old edge within the loop to be removed
    #         self.matrix[v_from, v_to] = 0 # remove an edge in the loop

    #         if self.tree_structure_check():
    #             r = r + 1
    #             H_diff = self.calculate_path_diff(self.matrix)
    #             exp_coef = np.exp(-self.beta*(H_diff))
    #             norm_coef = norm_coef + exp_coef
    #             # print(f'Delta H is {round(H_diff,2)}, and normalization coef now is {round(norm_coef,2)}.')
    #             # self.draw_tree2(s1_matrix=s1_matrix, title=f'iteration {r}')
    #             # plt.show()
    #             self.matrix = s1_matrix.copy()      
    #         else:
    #             # if the new tree doesn't retain old structure, add the edge back in
    #             # self.draw_tree2(s1_matrix=s1_matrix)
    #             self.matrix[v_from, v_to] = 1 
    #             self.matrix[v0, v1] = 0
    #     return norm_coef, r

    def generate_tree(self, mode = 'uniform', k=1000):
        """do random walk from given point until boundary is hit, or until all nodes have been visited
        initialize first point, generate tree until hitting dead end, initialize next point"""
        start = time.perf_counter()
        #initialize first point - this will be the outlet point
        outlet_point = random.choice(self.open_nodes)
        print('Outlet point:', self.convert_ij(*outlet_point))
        
        # update open_nodes 
        self.open_nodes.remove(outlet_point) 
        
        # build the first branch first 
        first_point = outlet_point
        
        while len(self.open_nodes)>0: 
            # print('###### Open nodes left: ###### ', len(self.open_nodes))
            # generate branch
            next_point = self.generate_branch(first_point)
            first_point = next_point
        
        # self.draw_tree(title = "Uniform before Gibbs")
        # norm_coef, _ = self.calculate_norm_coef(self.beta)
        # print(f'Normalization coefficient for this graph is {norm_coef}')

        finish = time.perf_counter()
        print(f'{self.m} by {self.n} uniform graph, finished in {round(finish-start,2)} seconds(s)')
        
        if mode == 'Gibbs':
            i = 0
            try:
                deltaH_list = self.generate_Gibbs(k=k)
                finish = time.perf_counter()
                print(f'A {k}-iteration Gibbs graph, finished in {round(finish-start,2)} seconds(s)')
                return deltaH_list
            except RecursionError:
                self.open_nodes = self.grid_nodes.copy()
                self.matrix = np.zeros((self.m*self.n, self.m*self.n))
                i += 1
                print(f'Generating new tree, attempt # {i}')
                self.generate_tree(mode=mode)     

    def draw_tree(self,title=None,node_to_color = None):
        _ = plt.figure(figsize=(10,4.5))
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        node_color_dict = {node: 'C0' for node in G.nodes}
        if node_to_color:
            node_color_dict[node_to_color] = 'C1'
        node_color = list(node_color_dict.values())
        plt.subplot(121)
        pos_grid = {k: self.convert_index(k) for k in G.nodes()}
        nx.draw(G, pos=pos_grid, node_color=node_color, node_size=2, edge_color='grey') #, with_labels = True)
        plt.subplot(122)
        pos_gv = graphviz_layout(G, prog = 'dot')
        #nx.draw(G, pos=nx.planar_layout(G), node_size=10, edge_color='lightgrey')
        nx.draw(G, pos=pos_gv, node_color=node_color, node_size=2, edge_color='lightgrey') #, with_labels=True)
        plt.title(title)
    
    # def draw_tree2(self,s1_matrix, title=None):
    #     _ = plt.figure(figsize=(10,4.5))
    #     G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
    #     plt.subplot(122)
    #     pos_grid = {k: self.convert_index(k) for k in G.nodes()}
    #     nx.draw(G, pos=pos_grid, node_size=2, edge_color='black') #, with_labels = True)
    #     plt.subplot(121)
    #     H = nx.from_numpy_matrix(s1_matrix, create_using=nx.DiGraph)
    #     pos_grid = {k: self.convert_index(k) for k in H.nodes()}
    #     nx.draw(H, pos=pos_grid, node_size=2, edge_color='grey') #, with_labels = True)
    #     plt.title(title)
    
    def export_tree(self, i, name = None):
        if not name:
            name = f'{self.n}-grid_{i}.pickle'
        f = open(name,'wb')
        pickle.dump(self.matrix,f)
        f.close()

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

def gibbs_pdf(uni, all_deltaH_list):
    _ = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for deltaH_list in all_deltaH_list:
        ax1.plot(deltaH_list,alpha = 0.1, color = 'C0')
    new_list = [elem for a0 in all_deltaH_list for elem in a0]
    ax2.hist(new_list,density=True,bins=30,orientation='horizontal')
    ax2.set_ylabel('$\Delta H$')
    ax2.set_xlabel('Frequency')
    ax1.set_ylabel('$\Delta H$')
    plt.title(f'beta = {uni.beta}')
    plt.savefig(f'./dist_beta{uni.beta}.png')

def main(size, beta=0.5):
    start = time.perf_counter()    
    all_deltaH_list = []
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    for _ in range(1000):
        uni = Uniform_network(size, size, beta=beta)
        deltaH_list = uni.generate_tree(mode="Gibbs", k=4000)
        # ax1.plot(deltaH_list,alpha = 0.1, color = 'C0')
        all_deltaH_list.append(deltaH_list)
        print(len(deltaH_list))
    uni.export_tree(i = beta)
    gibbs_pdf(uni,all_deltaH_list)
    name = f'deltaH_beta{uni.beta}.pickle'
    f = open(name,'wb')
    pickle.dump(all_deltaH_list,f)
    f.close()
    finish = time.perf_counter()
    print(f'{size} by {size} final graph, finished in {round(finish-start,2)} seconds(s)')
    # print(uni.matrix)

#%%
if __name__ == '__main__':
    #%%
    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")

    # signal.signal(signal.SIGALRM, handler)
    # signal.alarm(1800)

    size = 10
    dir_name =  f'gibbs{size}_{dt_str}'
    try: 
        os.chdir(dir_name)
    except FileNotFoundError:
        os.makedirs(dir_name)
        os.chdir(dir_name)
    main(size)