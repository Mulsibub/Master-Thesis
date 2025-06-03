import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid


class sink_to_source_class:
    def __init__(self,adjacency,F_in,F_out,F_in_total,P_minus,P_plus):
        self.adjacency = adjacency
        self.F_in = F_in
        self.F_out = F_out
        self.F_in_total = F_in_total
        self.P_minus = P_minus
        self.P_plus = P_plus
        self.l, self.n = F_in.shape

    #Network structure when walking from source to sink
    def walk(self): #source to sink.. will be reversable to sink to source
    
        #Initialize
        n_link = 0
        n_node = 0
        sub_step_intervals = []
        
        walk_link = np.full((self.l, 3), -1, dtype=int)
        WalkNode = namedtuple('WalkNode', ['links', 'sources', 'sinks'])
        walk_node = []
        link_n = []
        source_node_n = []
        sink_node_n = []
        
        
        #Determine initial state of unknowns (How many of sinks exporting neighbors knows their own composition?)
        Unknowns = np.sum(self.F_out>0,axis=0).A1
        initial_sink_nodes = np.where(self.F_out.sum(axis=0) < 1e-6)[1]
        neighbors = self.adjacency[:, initial_sink_nodes].nonzero()[0]
        
        for neighbor in neighbors:
            Unknowns[neighbor] -= 1
        Unknowns[initial_sink_nodes] = -1 #Already determined, they are now all -1
        
        while Unknowns.sum() != -self.n:
            
            source_nodes = np.where(Unknowns == 0)[0]
            
            for node in source_nodes:
                link_from_sink = self.F_out[:, node].nonzero()[0]
                sink_node = np.nonzero(self.F_in[link_from_sink])[1]
                
                #Node-wise step through
                link_n.append(link_from_sink)
                source_node_n.append(node)
                sink_node_n.append(sink_node)
                
                #Link-wise step through
                for i in range(len(link_from_sink)):
                    # Store link information (link index, source, sink)
                    walk_link[n_link, 0] = link_from_sink[i]
                    walk_link[n_link, 1] = node #source
                    walk_link[n_link, 2] = sink_node[i] #sink
                    n_link += 1
            
            sub_step_intervals.append(n_node) #THIS COULD BE USED TO VECTORIZE ALL SUB STEPS LATER! un-used for now
            
            neighbors = self.adjacency[:, source_nodes].nonzero()[0]
            for neighbor in neighbors:
                if Unknowns[neighbor] != -1:
                    Unknowns[neighbor] -= 1
            Unknowns[source_nodes] -= 1
        
        #append nodeal information into one data-structure
        walk_node.append(WalkNode(link_n, source_node_n, sink_node_n))
        
        walk_sink_to_source = {
            'link_wise' : walk_link,
            'node_wise': [row for row in walk_node if not np.any([x is None for x in row])], #Filter out NonType.. because len does not equal self.init.n but len+initial = self.init.n
            'substep_intervals':sub_step_intervals,
            'absolute_sink_nodes':initial_sink_nodes
            }
        return walk_sink_to_source
    
    
    def q_nn(self,walk): #importer point of veiew.. resposnibility of a sinks import
    
        node_walk = walk['node_wise']
        sources = node_walk[0].sources
        q = np.zeros((self.n,self.n))
        q_plus = np.zeros((self.n,self.n))
        np.fill_diagonal(q_plus,1)
        
        absolute_sinks = walk['absolute_sink_nodes']
        q[absolute_sinks,absolute_sinks] = 1
        
        P_out_i = q_plus * self.P_minus
        
        P_in = self.P_plus + self.F_in_total
        
        for i, source in enumerate(sources):
            links_from_sink = node_walk[0].links[i]
            sinks = node_walk[0].sinks[i]
            
            P_out_j = (q[:,sinks]*self.F_out[links_from_sink,:]).sum(axis=1)
            q[:,source] = (P_out_i[:,source] + P_out_j) / P_in[source]
        return q
    
    
    def q_ln(self,walk,q_nn):
        link_counter = 0
        node_walk = walk['node_wise']
        link_walk = walk['link_wise']
        q_ln = np.zeros((self.l,self.n))
    
        
        #initial_sink_nodes = walk['initial_sink_nodes']
        #initial_sink_link_wise = link_walk[np.where(np.isin(link_walk[:, 2], initial_sink_nodes))[0]]
        #q_ln[initial_sink_link_wise[:,0],initial_sink_link_wise[:,2]] = 1
        
        
        #CAN I DOT THIS?
        #q_ln[link_walk[:,0],:] += q_nn[:,link_walk[:,2]]
        
        
        #INSTEAD OF THIS??
        all_sources = node_walk[0].sources
        for i,source in enumerate(all_sources):
            
            out_flow_links = node_walk[0].links[i]
    
            for link in out_flow_links:
                sink = link_walk[:,2][link_counter]
                q_ln[link,:] = q_nn[:,sink]
                link_counter += 1
        
        return q_ln


class link_capacity:
    def __init__(self, q_ln, F, P):
        self.q_ln = q_ln  # Shape (956, 512)
        self.F = F
        self.P = P
        self.n = P.shape[0]
        self.f = abs(F.todense()).sum(axis=1).A1  # Shape (956,)
        self.l = self.q_ln.shape[0]

        # Precompute KDE and cumulative probability
        self.precompute_probabilities()
    
    def precompute_probabilities(self):
        """Precompute KDE and cumulative probability distribution"""
        # KDE for P(f_l)
        self.kde_f = gaussian_kde(self.f, bw_method='silverman')
        
        # Define range of f_l values
        self.x_values = np.linspace(np.percentile(self.f, 0.5), np.percentile(self.f, 99.5), 500)
        
        # Compute KDE estimated probability density P(f_l)
        self.prop_f = self.kde_f(self.x_values)

        # Precompute cumulative probability distribution P(f_l ≤ K)
        self.cum_dist = cumulative_trapezoid(self.prop_f, self.x_values, initial=0)
        self.cum_dist /= self.cum_dist[-1]  # Normalize

        # Precompute P(f_l) at actual f values (to avoid recomputation)
        self.P_f_values = self.kde_f(self.f)

    def cum_flow_distribution(self, K):
        """Compute cumulative probability P(f_l ≤ K) for array K"""
        K = np.asarray(K)
        K_indices = np.searchsorted(self.x_values, K)  
        K_indices = np.clip(K_indices, 0, len(self.cum_dist) - 1)
        return self.cum_dist[K_indices]



    def weights(self, K):
        """Compute conditional expectation ⟨q_ln | f_l⟩ and plot validation figures"""
    
        """First part of w (P_l|f_l>K)"""
        ##########################################################################
        # Get cumulative probability for each K
        prop_to_K = self.cum_flow_distribution(K)
        
        # Compute conditional probability density function P(f_l | f_l > K)
        
        cond_prop = self.prop_f[:, None] / (1 - prop_to_K)  # Broadcasting to match dimensions
        ##########################################################################
    
        """Second part of w: Compute ⟨q_ln | f_l⟩ using KDE-based probability calculations"""
        ##########################################################################
        # Initialize storage for P_f_given_q
        P_f_given_q = np.zeros((self.l, self.n))  # Shape (956, 512)
    
        for j in range(self.q_ln.shape[1]):  # Iterate over columns of q_ln
            weights_j = self.q_ln[:, j]
    
            # If all weights are zero, set P_f_given_q[:, j] = 0 (avoid KDE failure)
            if np.all(weights_j == 0):
                continue  # Skip KDE computation and leave P_f_given_q[:, j] as zeros
    
            # Compute KDE for each column of q_ln separately
            kde_f_given_q_j = gaussian_kde(self.f, weights=weights_j, bw_method='silverman')
            P_f_given_q[:, j] = kde_f_given_q_j(self.f)
    
        # Compute true conditional expectation ⟨q_ln | f_l⟩ = sum( q_ln * P(q_{ln} | f_l) )
        conditional_avg = np.sum(self.q_ln * P_f_given_q, axis=0)  # Shape (512,)
        ##########################################################################
    
        # Compute final weights
        #w = cond_prop * conditional_avg  # Ensure proper broadcasting
    
        return cond_prop,conditional_avg  # Return the final computed weights
    
        
    
        

        
        
        
        
    

        
        
        
    











