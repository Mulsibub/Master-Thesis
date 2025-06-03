import numpy as np
from collections import namedtuple

class source_to_sink_class:
    def __init__(self,adjacency,F_in,F_out,F_out_total,P_minus,P_plus): #TODO fix so it only needs need adj, F and P
    
        self.adjacency = adjacency
        self.F_in = F_in
        self.F_out = F_out
        self.F_out_total = F_out_total
        self.P_minus = P_minus
        self.P_plus = P_plus
        self.l, self.n = F_in.shape
        

    def walk(self): #source to sink
    
        n_link = 0
        n_node = 0
        sub_step_intervals = []
        
        walk_link = np.full((self.l, 3), -1, dtype=int)
        walk_node = []
        link_n = []
        source_node_n = []
        sink_node_n = []
        WalkNode = namedtuple('WalkNode', ['links', 'sources', 'sinks']) #A lil custom datatype
        
        
        
        Unknowns = np.sum(self.F_in>0,axis=0).A1
        initial_source_nodes = np.where(self.F_in.sum(axis=0) < 1e-6)[1]
        neighbors = self.adjacency[:, initial_source_nodes].nonzero()[0]
        
        for neighbor in neighbors:
            Unknowns[neighbor] -= 1
        Unknowns[initial_source_nodes] = -1 #Already determined, they are now all -1
        
            
        max_iter = 10000
        iter_count = 0
        while Unknowns.sum() != -self.n and iter_count < max_iter:
            sink_nodes = np.where(Unknowns == 0)[0]
            
            for sink in sink_nodes:
                link_from_source = self.F_in[:, sink].nonzero()[0]
                source = self.F_out[link_from_source].nonzero()[1]
                #Node-wise step through
                
                link_n.append(link_from_source)
                source_node_n.append(source)
                sink_node_n.append(sink)
                
                for i in range(len(link_from_source)):
                    # Store link information (link index, source, sink)
                    walk_link[n_link, 0] = link_from_source[i]
                    walk_link[n_link, 1] = source[i] #source
                    walk_link[n_link, 2] = sink
                    n_link += 1
            sub_step_intervals.append(n_node) #THIS COULD BE USED TO VECTORIZE ALL SUB STEPS LATER!
            neighbors = self.adjacency[:, sink_nodes].nonzero()[0]
            for neighbor in neighbors:
                if Unknowns[neighbor] != -1:
                    Unknowns[neighbor] -= 1
            Unknowns[sink_nodes] -= 1
            iter_count += 1
        if iter_count >= max_iter:
            return 'NoFlow'
            # raise RuntimeError("Exceeded max iterations in walk(). Likely stuck.")    
        else:
            walk_node.append(WalkNode(link_n, source_node_n, sink_node_n))
            walk_source_to_sink = {
                'link_wise' : walk_link,
                'node_wise': [row for row in walk_node if not np.any([x is None for x in row])],
                'substep_intervals':sub_step_intervals,
                'absolute_source_nodes':initial_source_nodes # len(initial_source_nodes) + len(node_wise) = self.init.n
                }
            return walk_source_to_sink


    def q_nn(self,walk): #exporter point of veiew.. responsibility of a sources export

        node_walk = walk['node_wise']
        sinks = node_walk[0].sinks
        q = np.zeros((self.n,self.n))
        q_plus = np.zeros((self.n,self.n))
        np.fill_diagonal(q_plus,1)
        
        absolute_sources = walk['absolute_source_nodes']
        q[absolute_sources,absolute_sources] = 1#P_in_j needs to be initialized for absolute source
        
        
        #The power the node provides itself
        P_in_i = q_plus * self.P_plus
        #Power not provided by the node itself 
        P_out = self.P_minus + self.F_out_total
        
        for i, sink in enumerate(sinks): #DONT TRY TO VECTORIZE THIS, it is dependent of itself! could use substeps far part vectorization
            links_from_sources = node_walk[0].links[i]
            sources = node_walk[0].sources[i]
            
            #The power provided to the node from other nodes (not just neighbor sources)
            P_in_j = (q[:,sources]*self.F_in[links_from_sources,:]).sum(axis=1)
            q[:,sink] = (P_in_i[:,sink] + P_in_j) / P_out[sink] #Responsibility of sink
        return q
    

    def partition_q(self,t,q,layout,load):
        wind_gen =    layout['Generation']['wind'][t,:]
        solar_gen =   layout['Generation']['solar'][t,:]
        backup_gen =  layout['Generation']['backup'][t,:]
        curtailment = layout['Generation']['curtailment'][t,:]
        ren_gen =     layout['Generation']['ren'][t,:]
        
        # Define cases for all time steps
        case_A = (ren_gen > load) & (curtailment > 0)
        case_B = (ren_gen < load) & (backup_gen > 0)
        case_C = (ren_gen > load) & (backup_gen > 0)
        
        #case_D = (ren_gen>load) & (curtailment>0)
        #case_E = (ren_gen>load) & (curtailment>0)
        #case_F = (ren_gen<load) & (curtailment>0)
        #print((case_A+case_B+case_C+case_D+case_E+case_F).sum())
        
        q_wind = np.zeros_like(q)  # Initialize q_wind
        q_wind[case_A, :] = q[case_A, :] * (wind_gen[case_A, None] / ren_gen[case_A, None])
        q_wind[case_C, :] = q[case_C, :] * (wind_gen[case_C, None] / ren_gen[case_C, None]) * (
            (ren_gen[case_C, None] - load[case_C, None]) / (backup_gen[case_C, None]+ren_gen[case_C,None]-load[case_C,None]))
        
        q_solar = np.zeros_like(q)  # Initialize q_solar
        q_solar[case_A, :] = q[case_A, :] * (solar_gen[case_A, None] / ren_gen[case_A, None])
        q_solar[case_C, :] = q[case_C, :] * (solar_gen[case_C, None] / ren_gen[case_C, None]) * (
            (ren_gen[case_C, None] - load[case_C, None]) / (backup_gen[case_C, None]+ren_gen[case_C,None]-load[case_C,None]))
        
        q_backup = np.zeros_like(q)
        q_backup[case_B,:] = q[case_B,:]
        q_backup[case_C,:] = q[case_C,:] * (backup_gen[case_C,None] / (backup_gen[case_C,None] + ren_gen[case_C,None] - load[case_C,None]))
        
        return q_wind,q_solar,q_backup
   