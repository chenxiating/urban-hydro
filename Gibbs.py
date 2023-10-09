"""
Gibbs.py 
@author: Xiating Chen, Xue Feng
Last Edited: 2023/10/05

This code is to generate spanning trees according to Gibbs distribution. 
    - Input: size of the lattice grid, parameter for flow path meandering, parameter for Gibbs distribution
    - Output: network without attributes
"""
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import time
import pickle
from multiprocessing import current_process

class Uniform_network:
    def __init__(self, m, n, beta, deltaH = None, outlet_point = (0,0), input_matrix = None, export = True): 
        """ m:      number of rows (indexed by i)
            n:      number of columns (indexed by j)
            beta:   beta for the Gibbs distribution
            deltaH: if passed, generate a tree with a specific deltaH value
            outlet_point:   if passed, coordinates of the outlet, default at (0,0)
            input_matrix:   if passed, adjacency matrix of the network without generating new tree
            export: to export tree as a pickle file, default export tree"""
        
        self.m = m
        self.n = n
        self.beta = beta
        self.deltaH = deltaH
        
        self.matrix = np.zeros((m*n, m*n))       # initialize adjacency matrix
        grid = [(i,j) for i in range(m) for j in range(n)]  # initialize all the nodes laid out on a grid 
        self.grid_nodes = grid
        
        grid_copy = self.grid_nodes.copy()      # label for "open" points that have not been visited before
        self.open_nodes = grid_copy 

        possible_edges = []                     # label for "open" edges that have not been visited before
        for vi, vj in grid_copy:
            possible_next_pts = [(vi+1, vj), (vi-1, vj), (vi, vj+1), (vi, vj-1)]
            pts = [pt for pt in possible_next_pts if (pt in set(self.grid_nodes))] # eliminate points outside of grid
            possible_edges.append(list((self.convert_ij(vi, vj), self.convert_ij(*k)) for k in pts))
        list_set = set(edge for sublist in possible_edges for edge in sublist)
        
        if input_matrix is not None:
            self.matrix = input_matrix
        else: 
            self.possible_edges = list(list_set)
            self.outlet_point=outlet_point
            self.generate_tree(export = export)
        self.path_diff = self.calculate_path_diff(self.matrix)
        self.path_diff_prime = self.calculate_path_diff_prime(self.matrix)
        self.find_outlet_point()
    
    def generate_tree(self, export):
        """do random walk from given point until boundary is hit, or until all nodes have been visited
        initialize first point, generate tree until hitting dead end, initialize next point"""
        start = time.perf_counter()
        grid_copy = self.grid_nodes.copy()
        self.open_nodes = grid_copy

        if not self.outlet_point:               # initialize first point - this will be the outlet point
            self.outlet_point = random.choice(self.open_nodes)
        self.open_nodes.remove(self.outlet_point)   # update open_nodes    
        first_point = self.outlet_point         # build the first branch first 
        while len(self.open_nodes) > 0:         # initialize the branches
            next_point = self.generate_branch(first_point)
            first_point = next_point
        
        self.deltaH_list = self.generate_Gibbs()    # Metropolis-Hastings sampling according to the Gibbs distribution

        self.path_diff = self.calculate_path_diff(self.matrix)  
        self.path_diff_prime = self.calculate_path_diff_prime(self.matrix)
        finish = time.perf_counter()
        
        cp = str(current_process())
        cp_name = cp[cp.find('name=')+6:cp.find(' parent=')-1]
        print(f'beta = {self.beta} Gibbs graph with H = {self.path_diff}, Hp = {self.path_diff_prime}, finished in {round(finish-start,2)} seconds(s) at {cp_name}')
        if export:
                self.export_tree()
        return self.path_diff

    def convert_ij(self, i,j): 
        """converts (i,j) coordinates on a matrix to corresponding index on adjacency matrix
        add 1 if starting index at 1 instead of 0"""
        return j+i*self.n
    
    def convert_index(self, k):
        """convert matrix index to (i,j) coordinates"""
        return (math.floor(k/self.n), k%self.n)

    def random_next_point(self, vi, vj):
        """generates the next point within the grid via random walk 
        determine possible destination nodes and assign probabilities"""
        possible_next_pts = [(vi+1, vj), (vi-1, vj), (vi, vj+1), (vi, vj-1)]

        pts = [pt for pt in possible_next_pts if (pt in set(self.grid_nodes))] # eliminate points outside of grid
        rv = len(pts)           # total number of adjacent nodes
        q = random.random()     # random number from 0 to 1 to simulate direction of movement
        upper_bounds = [i/rv for i in range(1,rv+1)]        # upper bounds of quantiles. For example, for rv = 3, this generates (0.33, 0.66, 1)
        ipt = np.where(q < np.array(upper_bounds))[0][0]    # first exceedance instance for upper bound
        next_point = pts[ipt]                               # find next point 
        return next_point

    def open_edges(self):
        """Finds open edges in the network"""
        possible_edges = self.possible_edges        # set of edges on the grid

        # matrix to show existing edges, regardless of direction
        current_edges_matrix = self.matrix + np.transpose(self.matrix)
        current_edge_arr = np.transpose(np.where(current_edges_matrix == 1))
        current_edges = list(map(tuple,current_edge_arr))

        # removing edges that start from the outlet point
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        self.find_outlet_point()
        impossible_edges = {(v_outlet, v_ng) for (v_outlet, v_ng) in possible_edges if v_outlet == self.outlet_point_k}

        # edges where a new edge can be added to create loop
        open_edges = list(set(possible_edges) - set(current_edges) - set(impossible_edges))

        return open_edges 

    def random_next_edge(self):
        """Aldous's 'minimal change' process. Adds an edge within the grid randomly, after an edge is removed
        from the grid in the process of generating network approcahing Gibbs dist. Generate new tree s2 from s1 and check whether s2 is a tree"""
        
        open_edges = self.open_edges()      # edges where a new edge can be added to create loop, return a list of edges
        random.shuffle(open_edges)          # randomly shuffle the list of open edges
        
        for v_start, v_end in open_edges:   # randomly select one open edge from tree s1
            self.matrix[v_start, v_end] = 1 # add an edge in "open edges" to create loop
            G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph) # create a tree with the new matrix
            [(v_from, v_to)] = [(v_from, v_to) for (v_from, v_to) in G.edges if (v_from == v_start) and (v_to != v_end)]    # check all the edges we can remove to disconnect the loop
            self.matrix[v_from, v_to] = 0   # remove such edge in the loop
            if self.is_tree_structure():    # check if the modified matrix s2 maintains a tree structure
                r = 0                       # initialize r, parameter that counts all possible neighboring tree
                vi, vj = self.convert_index(v_from)
                possible_next_pts = [(vi+1, vj), (vi-1, vj), (vi, vj+1), (vi, vj-1)]     
                pts = [pt for pt in possible_next_pts if (pt in set(self.grid_nodes))]      # list all neighboring nodes to the starting node of the newly added edge
                for x, y in pts:
                    new_v_to = self.convert_ij(x, y)    # find the matrix location of the new ending node
                    r = r + self.matrix[v_from, new_v_to] + self.matrix[new_v_to, v_from]   # count the occupied neighboring edges of s1
                r = len(pts) - r        # N(s1), count the occupied neighboring edges of s1
                return r
            else:
                # if the new tree s2 doesn't retain old structure, add the edge back in and the resulting tree is s1
                self.matrix[v_from, v_to] = 1 
                self.matrix[v_start, v_end] = 0
        self.random_next_edge()
        
    def generate_branch(self, current_point):
        """recursive function for branch generation, generate next point from current point"""
        next_point = self.random_next_point(*current_point)
        
        # return points in matrix order & update adjacency matrix
        v0 = self.convert_ij(*current_point) # in matrix order
        v1 = self.convert_ij(*next_point)
        
        try:
            self.open_nodes.remove(next_point)  # test to see if next_point is open
        except ValueError:  
            pass    # this means that it hit an existing tree - this will be ignored
        else:       # have not hit an existing tree
            self.matrix[v1,v0] = 1   # update adjacency matrix 
        return next_point

    def find_outlet_point(self):
        """find outlet point of the network, if it's not (0,0)"""
        G = nx.from_numpy_matrix(self.matrix,create_using=nx.DiGraph)
        self.outlet_point_k = [n for n, d in G.out_degree() if d == 0][0]
        self.outlet_point = self.convert_index(self.outlet_point_k)
    
    def calculate_path_diff(self, input_matrix):
        """calculate H_T(s) the flow path variation between the actual and the shortest path length according to Troutman & Karlinger (1992)"""
        G = nx.from_numpy_matrix(input_matrix,create_using=nx.DiGraph)
        self.find_outlet_point()
        total_path = sum([len(nx.shortest_path(G, source = k, target = self.outlet_point_k)) - 1 for k in G.nodes])
        grid_nodes = self.grid_nodes
        shortest_path = sum((abs(x - self.outlet_point[0]) + abs(y - self.outlet_point[1])) for x, y in grid_nodes)
        diff = total_path - shortest_path
        if diff < 0:
            raise Exception("Diff is negative!")
        return diff
    
    def calculate_path_diff_prime(self, input_matrix):
        """calculate the size-normalized difference between the mapped path length and shortest path length proposed in Chen & Feng (2022)"""
        G = nx.from_numpy_matrix(input_matrix,create_using=nx.DiGraph)
        self.find_outlet_point()
        ratio_sum = sum([((len(nx.shortest_path(G, source = k, target = self.outlet_point_k)) - 1) -
        (abs(self.convert_index(k)[0] - self.outlet_point[0]) + abs(self.convert_index(k)[1] - self.outlet_point[1])))/ 
        (abs(self.convert_index(k)[0] - self.outlet_point[0]) + abs(self.convert_index(k)[1] - self.outlet_point[1])) for k in G.nodes if k != self.outlet_point_k])

        diff_prime = ratio_sum / len(G.nodes)
        if diff_prime < 0:
            raise Exception("Diff is negative!")
        return diff_prime

    def generate_Gibbs(self):
        """Metropolis-Hastings algorithm for sampling spanning trees
        return deltaH_list: records all the H iterated while generating graph"""
        deltaH_list = []
        burntime = 0.5*self.n**4 
        i = 0       # initialize burntime iteration
        while i <= burntime:
            i +=1
            s1_matrix = self.matrix.copy()
            r = self.random_next_edge()     # generate s2
            s2_matrix = self.matrix.copy()
            H_diff = float(self.calculate_path_diff(s2_matrix) - self.calculate_path_diff(s1_matrix))   # calculate the path difference between s1 and s2
            threshold = 1/r*min(1, np.exp(-self.beta*H_diff))   # acceptance ratio for switching to s2
            x = random.random()             # x, a unif var to decide whether to s2 is accepted
            if x > threshold:               # x > acceptance ratio, reject s2, return to s1
                self.matrix = s1_matrix
            if self.pass_deltaH_threshold():
                deltaH_list.append(self.calculate_path_diff(self.matrix))
                print(f"We passed delta H requirements and are done here! Delta H = {self.deltaH}")
                break
            deltaH_list.append(self.calculate_path_diff(self.matrix))
        return np.array(deltaH_list)

    def pass_deltaH_threshold(self, deltaH_threshold = None, msg = None):
        """check if self.deltaH is within a +/- desired deltaH threshold, the default threshold is sqrt of the grid size"""
        if self.deltaH:
            if not deltaH_threshold:
                deltaH_threshold = self.n
            path_diff = self.calculate_path_diff(self.matrix)
            if msg:
                print(msg)
            return (path_diff > (self.deltaH - deltaH_threshold)) & (path_diff < (self.deltaH + deltaH_threshold))
        else: 
            return False

    def is_tree_structure(self):
        """check if the tree retains original structure after 'minimal change' process """
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        return nx.is_tree(G) and max(d for n, d in G.out_degree()) <= 1

    def draw_tree(self,title=None,dist_label=True,save=False,starting_coord=None):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis('off')
        small_node_size = 50
        self.find_outlet_point()
        G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        node_color_dict = {node: 'C0' for node in G.nodes}
        node_size_dict = {node: small_node_size for node in G.nodes}
        node_color_dict[self.outlet_point_k] = 'C1'
        node_size_dict[self.outlet_point_k] = small_node_size*5
        node_color = list(node_color_dict.values())
        node_size = list(node_size_dict.values())
        pos_grid = {k: self.convert_index(k) for k in G.nodes()}
        path_to_highlight = []
        nx.draw_networkx_nodes(G, pos=pos_grid, node_color=node_color, node_size=node_size)
        if dist_label:
            node_labels = {}
            for node in G.nodes:
                node_xy = self.convert_index(node)
                actual = len(nx.shortest_path(G, source = node, target = self.outlet_point_k)) - 1
                shortest = abs(node_xy[1] - self.outlet_point[1]) + abs(node_xy[0] - self.outlet_point[0])
                if actual-shortest == 0: 
                    label = ''
                else:
                    label = actual-shortest
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
            passing_nodes = nx.shortest_path(G, source = starting_coord, target = self.outlet_point_k)
            path_to_highlight = [(passing_nodes[x], passing_nodes[x+1]) for x in range(len(passing_nodes)-1)]
            nx.draw_networkx_edges(G,pos=pos_grid,edgelist=path_to_highlight,width=4,edge_color='r')
            nx.draw_networkx_nodes(G, pos=pos_grid, nodelist=[starting_coord], node_color='r',edgecolors='w', node_size=small_node_size*5)
        
        nx.draw_networkx_edges(G, pos=pos_grid,edgelist=G.edges - path_to_highlight,width=1,edge_color='grey')
        plt.title(title)
        if save:
            plt.savefig(f'./tree_size{self.n}by{self.m}_beta{self.beta}.png')
    
    def export_tree(self, name = None):
        """exporting tree to a pickle file"""
        if not name:
            self_name = str(self)
            ID = self_name[self_name.find('x')-1:]
            name = f'{self.n}-grid_beta-{self.beta}_dist-{self.path_diff}_ID-{ID}.pickle'
        f = open(name,'wb')
        pickle.dump(self.matrix,f)
        f.close()

def gibbs_pdf(beta,all_deltaH_list):
    """ plot the frequency distribution of all H generate with a given beta """
    fig = plt.figure()
    plt.suptitle(rf'Distribution of {int(len(all_deltaH_list))} random Gibbs trees with $\beta$ = {beta}')
    gs = fig.add_gridspec(1,5)
    ax1 = fig.add_subplot(gs[0,:-1])
    ax2 = fig.add_subplot(gs[0,-1])
    for deltaH_list in all_deltaH_list:
        ax1.plot(deltaH_list,alpha = max(0.05, 1/len(all_deltaH_list)), color = 'C0')
    ax1.set_ylabel('$\Delta H$')
    ax1.set_xlabel('Iteration')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    new_list = [elem for a0 in all_deltaH_list for elem in a0]
    ax2.hist(new_list,density=True,orientation='horizontal')
    ax2.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.savefig(f'./dist_beta{beta}.png')

def generate_many_trees(size, beta=0.5, tree_num = 1000):
    """ this function can be used to test out generate multiple square-trees with a given beta, and 
    the delta H (path difference) will be saved as an output. 

    It will also create a figure that shows the delta H distribution of the graphs generated.
    size:   number of rows and columns in the square matrix (square graph)
    beta:   parameter for Gibbs distribtuion 
    tree_num: number of trees to be generated """

    start = time.perf_counter()    
    all_deltaH_list = []
    for _ in range(tree_num):
        uni = Uniform_network(size, size, beta=beta)
        all_deltaH_list.append(uni.deltaH_list)
    gibbs_pdf(uni.beta,all_deltaH_list)
    name = f'deltaH_beta{uni.beta}.pickle'
    f = open(name,'wb')
    pickle.dump(all_deltaH_list,f)
    f.close()
    finish = time.perf_counter()
    print(f'{size} by {size} final graph, finished in {round(finish-start,2)} seconds')

def main(size, beta, outlet_point,input_matrix = None):
    """ generate one single tree.
    size:   number of rows and columns in the square matrix (square graph)
    beta:   parameter for Gibbs distribtuion 
    outlet_point: specify a fixed outlet point in the square graph
    input_matrix: the incidence matrix of the graph.
    
    When "input_matrix" is specified, it overrides the other inputs 
    (e.g., size, beta, outlet_point). """

    gibbs = Uniform_network(m=size, n=size, beta=beta, outlet_point=outlet_point, input_matrix=input_matrix)
    return gibbs

if __name__ == '__main__':

    ## To generate 1000 trees (3x3 grid) with different beta values
    for beta_val in [0.01, 1]:
        generate_many_trees(size=3, beta=beta_val, tree_num=1000)
    
    ## To generate one single tree
    main(size = 3, beta = 0.01, outlet_point = (0,0))