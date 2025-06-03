#%%
import numpy as np
import scipy as sc
import scipy.interpolate
import scipy.optimize

#%%
import pandas as pd
import geopy.distance

#%%
 
class layout_class:
    def __init__(self, init):
        self.init = init #Call init class, self.init.whatever
        self.init.t2=8760
        

####################################################################
# METHODS
####################################################################

    def CF_prop(self, alpha, CF,Load_int):
        Mean_Load = self.init.Mean_Load[Load_int[0]:Load_int[1]]
        Total_Load = Mean_Load.sum()
        
        # Compute the mean capacity factor across time
        CF_mean = CF.mean(axis=0)
        
        Avg_Generation_needed  = alpha*self.init.gamma * Total_Load
        a = Avg_Generation_needed / np.sum(np.power(CF_mean,self.init.beta+1))
        
        kappa = np.multiply(a, np.power(CF_mean[:, np.newaxis], self.init.beta))
    
        
        Generation = np.multiply(kappa[:, np.newaxis, :], CF.T[:,:,np.newaxis]).astype(np.float32)
        Generation = np.nan_to_num(Generation.transpose(1, 0, 2).astype(np.float32))
        kappa = np.quantile(Generation, q=0.99, axis=0)
        return Generation, kappa
    
    def SynchBalancing(self, Gen_wind, Gen_solar,Load_int):
        Load = self.init.Load[:,Load_int[0]:Load_int[1]]
        Mean_Load = self.init.Mean_Load[Load_int[0]:Load_int[1]]
        Total_Load = Mean_Load.sum()
        
        
        # Renewable Generation
        Gen_renewable = np.add(Gen_wind, Gen_solar).astype(np.float32)
        # Sum of all nodal mismatches
        mismatch = np.subtract(Gen_renewable, Load[:,:,np.newaxis]).astype(np.float32)
        
        # Balancing
        Balancing = np.multiply(mismatch.sum(axis=1)[:,:,np.newaxis]  ,  np.divide(Mean_Load,Total_Load)).astype(np.float32)
        Balancing = Balancing.transpose(0,2,1).astype(np.float32)
        
        # return Balancing
        Injection = np.subtract(mismatch, Balancing).astype(np.float32)
        return Balancing, Injection

        
 
    
    def NodalBalancing(self, GW, GS, Load_int):
        # Converts DataFrames to numpy arrays for code effeciency
        Load = self.init.Load[:,Load_int[0]:Load_int[1]]
        # Total_Load = Load.sum(axis=1)
        GR = np.add(GW, GS).astype(np.float32)
        mismatch = np.subtract(GR, Load[:,:,np.newaxis]).astype(np.float32)
        
        
        # Boolean lists for M+ or M-
        PMmlist = mismatch > 0 # True When positive nodal mismatch    (m+)
        NMmlist = mismatch <= 0 # True When negative nodal mismatch   (m-)
        
        # Calculating sum of nodes with surplus and sum of nodes with deficit
        PDel = np.where(PMmlist, mismatch, 0).sum(axis=1) # Surplus Gen
        NDel = -np.where(NMmlist, mismatch, 0).sum(axis=1) # Deficit Gen
        
        # Boolean list for dispatch cases
        Caselist = PDel > NDel #True When DeltaP > DeltaN, i.e. Positive Global Mismatch)
        
        # M+
        PMm = np.where(PMmlist, mismatch, 0)
        # PMm_S = PMm.sum(axis=1)
        
        # M-
        NMm = np.where(NMmlist, mismatch, 0)
        # NMm_S = -NMm.sum(axis=1)
        
        # When Delta- < Delta+
        PBackup = np.zeros(np.shape(mismatch)).astype(np.float32) # For all cases >Delta- => B_n=0
        # handling m- nodes
        PInjection = NMm
        # Handling m+ nodes
        twf = np.divide(NDel, np.where(PDel == 0, np.nan, PDel))
        twf = np.nan_to_num(twf)
        temp = np.multiply(PMm,  twf[:, np.newaxis])
        PInjection = np.add(PInjection, temp)
        PCurtailment = np.subtract(PMm, temp)
        
        # When Delta- > Delta+
        NCurtailment = np.zeros(np.shape(mismatch)) # For all cases <Delta- => C_n=0
        # handling m- nodes
        twf = np.divide(PDel, np.where(NDel == 0, np.nan, NDel))
        twf = np.nan_to_num(twf)
        temp = np.multiply(NMm,  twf[:, np.newaxis])
        NBackup = np.subtract(NMm, temp)
        # Handling m+ nodes
        NInjection = np.add(PMm, temp)
        
        #Combining
        temp = Caselist[:, np.newaxis, :]
        Injection = np.where(temp, PInjection, NInjection)
        Backup = np.where(temp, PBackup, NBackup)
        Curtailment = np.where(temp, PCurtailment, NCurtailment)
        Balancing = np.add(Backup, Curtailment)
        
        # return Balancing
        return Balancing, Injection #, PowerInjected, mismatch
    
    #############################################################################
    #Optimal alpha definition and options in this section
    
    #Note, the function under simply forwards the otimal alpha calculation to the relevant function
    def Optimal_alpha(self,Balancing,kappa_wind,kappa_solar,Gen_wind,Gen_solar,alpha): #Optimize alpha depending on opt alpha choice
        if self.init.alpha_opt_parameter == 'min_backup_energy':
            alpha_opt = self.alpha_min_backup_energy(Balancing,alpha)
            
        elif self.init.alpha_opt_parameter == 'critical_parameter':
            VRES_gen = np.add(Gen_wind,Gen_solar)
            mismatch = np.subtract(VRES_gen,Balancing)
            
            alpha_opt = self.alpha_min_critical_parameter(Balancing,kappa_wind,kappa_solar,mismatch,alpha)
        
        return alpha_opt
    
    #This is the one that is almost always used
    def alpha_min_backup_energy(self,Balancing,alpha): 
        Backup_Gen = -np.minimum(Balancing,0).astype(np.float32)
        Backup_Energy = Backup_Gen.sum(axis=(0, 1))
        f = scipy.interpolate.interp1d(alpha, Backup_Energy, kind='quadratic')
        alpha_opt = sc.optimize.minimize_scalar(f, bounds=(0,1), method='Bounded').x
        return alpha_opt
    
    #TODO yet to be implemented
    def alpha_min_LCOE_eu(self):
        #Yet to be implimented
        n=2
        return 1
    


    def Capacity_Calc(self,Balancing):
        kappaBackup = np.quantile(np.abs(Balancing), q=0.99, axis=0)
        return kappaBackup
    
    def Trans_capacity(self,Injection, Balancing,Bus):
        try:
            alpha_n = Injection.shape[2]
        except IndexError:
            alpha_n = 1
            
        

        
        kappaTrans = np.zeros((self.init.l,alpha_n))
        length = np.zeros(self.init.l)
        
        incidence = self.init.link_data['incidence']
        PTDF = self.init.link_data['PTDF']

        
        for i in range(self.init.l):
            pos = Bus.iloc[np.where(incidence[:,i]!=0)[0],:][['x','y']].to_numpy()
            length[i] = geopy.distance.distance(pos[0,:], pos[1,:]).km
        
        
        for i in range(alpha_n):
            if alpha_n == 1:
                # If alpha_n == 1, Injection is 2D, no need to slice
                FlowPattern = Injection @ PTDF.T  # Multiply Injection (time, node) with PTDF
            else:
                # If alpha_n > 1, Injection is 3D, slice along third dimension
                FlowPattern = Injection[:, :, i] @ PTDF.T
            kappaTrans[:, i] = np.quantile(np.abs(FlowPattern), q=0.99, axis=0)
            
        return kappaTrans,length
    
    def Trans_capacity2(self, Injection, Balancing,Bus):
       
        # kappaTrans = np.zeros(self.init.l)
        length = np.zeros(self.init.l)
        incidence = self.init.link_data['incidence']
        PTDF = self.init.link_data['PTDF'].astype(np.float32)

        
        for i in range(self.init.l):
            pos = Bus.iloc[np.where(incidence[:,i]!=0)[0],:][['x','y']].to_numpy()
            length[i] = geopy.distance.distance(pos[0,:], pos[1,:]).km
        
        Injections = Injection.squeeze(-1).astype(np.float32)  # Removes the last dimension (1)
        # Compute FlowPattern
        FlowPattern = np.matmul(Injections, PTDF.T)  # or Injection @ PTDF.T
        kappaTrans = np.quantile(np.abs(FlowPattern), q=0.99, axis=0)
            
        return FlowPattern, kappaTrans, length    
 
       
####################################################################
# APPLICATION - Running the methods
####################################################################
    


    def run(self):
        # Check the layout scheme and delegate to the appropriate method
        if self.init.Layout_Scheme == 'Global':
           return self.GlobalLayout()
        elif self.init.Layout_Scheme == 'Local':
            return self.LocalLayout()
        elif self.init.Layout_Scheme == 'LocalGAS':
            return self.LocalLayoutGAS()
        
    
    def BalancingHandler(self, Gen_wind, Gen_solar, Load_int, alpha_n = 1):
        """
        Function for handling the different types of balancing.

        Parameters
        ----------
        Gen_wind : Array of float 32
            Wind generation for all nodes over all time
        Gen_solar : Array of float 32
            Solar generation for all nodes over all time
        Load_int : list
            The range of nodes.
        alpha_n : int, optional
            the number of alphas. The default is 1.

        Raises
        ------
        ValueError
            The balancing scheme options are 'Global', 'Local' (also called CB), 'Nodal', and 'NoT.

        Returns
        -------
        Balancing : Array of float32
            Nodal balancing.
        Injections : Array of float32
            Nodal power injections.

        """
        if self.init.Balancing_scheme == 'Global':
            Balancing, Injections = self.SynchBalancing(Gen_wind, Gen_solar, Load_int)
        elif self.init.Balancing_scheme == 'Local':
            Balancing = np.zeros((self.init.t2, self.init.n, alpha_n))
            Injections = np.zeros((self.init.t2, self.init.n, alpha_n))
            for i, (start, end) in enumerate(self.init.country_int):
                Balancing[:, start:end,], Injections[:, start:end,] = self.SynchBalancing(Gen_wind[:, start:end, :], Gen_solar[:, start:end, :], [start, end])
        elif self.init.Balancing_scheme == 'Nodal':
            Balancing, Injections = self.NodalBalancing(Gen_wind, Gen_solar, Load_int)
        elif self.init.Balancing_scheme == 'NoT':
            self.remove_international_links()
            Balancing = np.zeros((self.init.t2, self.init.n, alpha_n))
            Injections = np.zeros((self.init.t2, self.init.n, alpha_n))
            for i, (start, end) in enumerate(self.init.country_int):
                Balancing[:, start:end,], Injections[:, start:end,] = self.SynchBalancing(Gen_wind[:, start:end, :], Gen_solar[:, start:end, :], [start, end]) 
        else:
            raise ValueError("Invalid Balancing Scheme")
        
        return Balancing, Injections
        

    def GlobalLayout(self):
        alpha_n = self.init.n_alpha
        self.alpha_n = alpha_n #Sometimes i need this.. meh
        Load_int = [0, self.init.n]
        alpha = np.linspace(0, 1, alpha_n)
                
        # Generate wind and solar using alpha values
        Gen_wind_alpha, kappa_wind_alpha = self.CF_prop(alpha, self.init.CF_wind, Load_int)
        Gen_solar_alpha, kappa_solar_alpha = self.CF_prop(1 - alpha, self.init.CF_solar, Load_int)
        
        Balancing_alpha, _ = self.BalancingHandler(Gen_wind_alpha, Gen_solar_alpha, Load_int, alpha_n = alpha_n)

        # Optimize alpha
        alpha_opt = self.Optimal_alpha(Balancing_alpha, kappa_wind_alpha, kappa_solar_alpha, Gen_wind_alpha, Gen_solar_alpha, alpha)
        
        # Calculate final wind and solar generation
        Gen_Wind, kappa_wind = self.CF_prop(alpha_opt, self.init.CF_wind, Load_int)
        Gen_Solar, kappa_solar = self.CF_prop(1 - alpha_opt, self.init.CF_solar, Load_int)
        
        balancing, injections = self.BalancingHandler(Gen_Wind, Gen_Solar, Load_int) # Now for only a single alpha
        
        return self.layoutformatting(Gen_Wind, kappa_wind, Gen_Solar, kappa_solar, balancing, injections, alpha_opt)
    
  
    
    def LocalLayout(self):
        alpha_n = self.init.n_alpha
        self.alpha_n = alpha_n
        alpha = np.linspace(0, 1, alpha_n)
        
        # Preallocate arrays for wind and solar generation
        Gen_wind_alpha, Gen_solar_alpha = (np.zeros((self.init.t2, self.init.n, alpha_n)) for _ in range(2))
        kappa_wind_alpha, kappa_solar_alpha = (np.zeros((self.init.n,alpha_n)) for _ in range(2))
        Gen_Wind, Gen_Solar, balancing = (np.zeros((self.init.t2, self.init.n, 1)) for _ in range(3))
        kappa_solar, kappa_wind = (np.zeros((self.init.n, 1)) for _ in range(2))
        alpha_opt = np.zeros(self.init.n_country)
        Load_int = [0, self.init.n]
        ###################################################
        # Balancing_alpha = np.zeros((512,100))
        # Loop through country-specific sections and compute generation
        for i, (start, end) in enumerate(self.init.country_int):
            Gen_wind_alpha[:, start:end, :], kappa_wind_alpha[start:end, :] = self.CF_prop(alpha, self.init.CF_wind[:, start:end], [start, end])
            Gen_solar_alpha[:, start:end, :], kappa_solar_alpha[start:end, :] = self.CF_prop(1 - alpha, self.init.CF_solar[:, start:end], [start, end])
            Load = self.init.Load[:,start:end].sum(axis=1)
            MM = (Gen_wind_alpha[:, start:end, :] + Gen_solar_alpha[:, start:end, :]).sum(axis=1)-Load[:,np.newaxis]
            Backup_Gen = -np.minimum(MM,0).astype(np.float32).sum(axis=0)
            f = scipy.interpolate.interp1d(alpha, Backup_Gen, kind='quadratic')
            alpha_opt[i] = sc.optimize.minimize_scalar(f, bounds=(0,1), method='Bounded').x
            Gen_Wind[:, start:end,:], kappa_wind[start:end,:] = self.CF_prop(alpha_opt[i], self.init.CF_wind[:, start:end], [start, end])
            Gen_Solar[:, start:end,:], kappa_solar[start:end,:] = self.CF_prop(1 - alpha_opt[i], self.init.CF_solar[:, start:end], [start, end])
       
            
        Balancing, Injections = self.BalancingHandler(Gen_Wind, Gen_Solar, Load_int)
        return self.layoutformatting(Gen_Wind, kappa_wind, Gen_Solar, kappa_solar, Balancing, Injections, alpha_opt)
    
    def LocalLayoutGAS(self):
        alpha_n = self.init.n_alpha
        n_countries = self.init.n_country
        alpha_bounds = [(0, 1)] * n_countries  # bounds for each country's alpha
        
        def greedy_axial_search(objective, alpha_init, bounds, tol=1e-4, max_iter=100, resolution=10):
            n = len(alpha_init)
            alpha = alpha_init.copy()
            best_score = objective(alpha)
            
            for iteration in range(max_iter):
                improved = False
                for i in range(n):
                    best_val = alpha[i]
                    best_obj = best_score
        
                    test_vals = np.linspace(bounds[i][0], bounds[i][1], resolution)
                    for val in test_vals:
                        alpha_try = alpha.copy()
                        alpha_try[i] = val
                        obj = objective(alpha_try)
                        if obj < best_obj:
                            best_obj = obj
                            best_val = val
        
                    if best_val != alpha[i]:
                        alpha[i] = best_val
                        best_score = best_obj
                        improved = True
        
                if not improved:
                    break
            return alpha, best_score
        
        def greedy_axial_search_with_restarts(objective, bounds, num_restarts=10, **kwargs):
            n = len(bounds)
            best_alpha = None
            best_score = float('inf')
        
            for r in range(num_restarts):
                # Random initialization within bounds
                alpha_init = np.array([np.random.uniform(low, high) for (low, high) in bounds])
                
                alpha, score = greedy_axial_search(objective, alpha_init, bounds, **kwargs)
        
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
        
                print(f"Restart {r+1}/{num_restarts}: Score = {score:.4f}")
        
            print(f"\nBest overall score: {best_score:.4f}")
            return best_alpha, best_score
        
        def objective(alpha_opt):
            # Calculate wind and solar generation using the current alpha vector
            Gen_Wind = np.zeros((self.init.t2, self.init.n))
            Gen_Solar = np.zeros((self.init.t2, self.init.n))
            
            for i, (start, end) in enumerate(self.init.country_int):
                gen_wind_raw, _ = self.CF_prop(alpha_opt[i], self.init.CF_wind[:, start:end], [start, end])
                Gen_Wind[:, start:end] = np.squeeze(gen_wind_raw, axis=2)  # Squeeze before assigning

                gen_solar_raw, _ = self.CF_prop(1 - alpha_opt[i], self.init.CF_solar[:, start:end], [start, end])
                Gen_Solar[:, start:end] = np.squeeze(gen_solar_raw, axis=2)
                        
            Load = self.init.Load.sum(axis=1)
            Net = Gen_Wind.sum(axis=1) + Gen_Solar.sum(axis=1) - Load
            Backup_Gen = -np.minimum(Net, 0).astype(np.float32)
            # print(f"Objective({alpha_opt}) = {Backup_Gen.sum()}")
            return Backup_Gen.sum()  # Objective: total EU-wide backup generation
        
        # Initial guess: alpha = 0.5 for all countries
        alpha_init = np.full(n_countries, 0.7)
        import time
        start1 = time.time()
        alpha_opt, _ = greedy_axial_search_with_restarts(
            objective,
            bounds=alpha_bounds,
            num_restarts=2,       # Seems to be robust at res=11
            tol=1e-1,
            max_iter=100,
            resolution=alpha_n*2          # Higher resolution for finer search
        )
        end1 = time.time()
        length1 = end1 - start1
        self.timeelapsed = length1/60
        print(f"GAS elapsed time: {self.timeelapsed}")

    
        # Now compute final Gen_Wind, Gen_Solar with optimized alphas
        Gen_Wind, Gen_Solar, balancing = (np.zeros((self.init.t2, self.init.n, 1)) for _ in range(3))
        kappa_solar, kappa_wind = (np.zeros((self.init.n, 1)) for _ in range(2))
        
        for i, (start, end) in enumerate(self.init.country_int):
            Gen_Wind[:, start:end,:], kappa_wind[start:end,:] = self.CF_prop(alpha_opt[i], self.init.CF_wind[:, start:end], [start, end])
            Gen_Solar[:, start:end,:], kappa_solar[start:end,:] = self.CF_prop(1 - alpha_opt[i], self.init.CF_solar[:, start:end], [start, end])
        
        Balancing, Injections = self.BalancingHandler(Gen_Wind, Gen_Solar, [0, self.init.n])
        return self.layoutformatting(Gen_Wind, kappa_wind, Gen_Solar, kappa_solar, Balancing, Injections, alpha_opt)
    
    def layoutformatting(self, Gen_Wind, kappa_wind, Gen_Solar, kappa_solar, balancing, injection, alpha_opt):

        Gen_Backup = -np.minimum(balancing,0).astype(np.float32)
        Gen_Curtailment = np.maximum(balancing,0).astype(np.float32)
        backup_energy = Gen_Backup.sum(axis=(0, 1))
        backup_kappa = np.quantile(Gen_Backup, 0.99, axis=0)[:,0]

        mismatch = np.subtract(np.add(Gen_Wind[:,:,0], Gen_Solar[:,:,0]), self.init.Load)
        Flows, kappaTrans, length  = self.Trans_capacity2(injection, balancing, self.init.BusData)
        
        # Nodal transmission kappa
        Nodaltrans_Load = self.compute_nodal_transmission_capacity((kappaTrans, length), Method="Load")
        Nodaltrans_Half = self.compute_nodal_transmission_capacity((kappaTrans, length), Method="Half")
        
        # Country acumullations of transmission capacity
        kappa_C_T_Load  = np.zeros(len(self.init.country_list))
        kappa_C_T_Half  = np.zeros(len(self.init.country_list))
        for i, (start, end) in enumerate(self.init.country_int):
           kappa_C_T_Load[i] = (Nodaltrans_Load[start:end]).sum()
           kappa_C_T_Half[i] = (Nodaltrans_Half[start:end]).sum()
        
        kappa = {
            'wind':  kappa_wind[:,0].astype(np.float32),
            'solar': kappa_solar[:,0].astype(np.float32),
            'backup': backup_kappa.astype(np.float32),
            'trans': {'Power':kappaTrans.astype(np.float32), 'length':length.astype(np.float32)}, # Linkwise capacities. See misc for nodal/country wise
            'backup_energy': Gen_Backup.sum(axis=0)[:,0].astype(np.float32), # Sum of produced energy for each node ## Might be changed to avg
        }
        Generation = {
            'wind':Gen_Wind[:,:,0].astype(np.float32),
            'solar':Gen_Solar[:,:,0].astype(np.float32),
            'backup':Gen_Backup[:,:,0].astype(np.float32),
            'curtailment':Gen_Curtailment[:,:,0].astype(np.float32),
            'ren':np.add(Gen_Wind[:,:,0],Gen_Solar[:,:,0]).astype(np.float32)
            }
        misc = {
            'alpha_opt':alpha_opt,
            'balancing':balancing[:,:,0].astype(np.float32),
            'backup_energy':backup_energy,
            'mismatch':mismatch.astype(np.float32),
            'injection':injection[:,:,0].astype(np.float32),
            'Flows': Flows.astype(np.float32),
            'nodal_trans_Load': Nodaltrans_Load.astype(np.float32),
            'nodal_trans_Half': Nodaltrans_Half.astype(np.float32),
            'country_trans_Load': kappa_C_T_Load.astype(np.float32),
            'country_trans_Half': kappa_C_T_Half.astype(np.float32)
            }
        
        result = {
            'kappa':kappa,
            'Generation':Generation,
            'misc':misc
            }
        return result
    
    
    
 
    def remove_international_links(self):
        """
        Modifies the incidence matrix by removing international links.
        
        Parameters:
        - incidence_matrix (np.array): The (512, 956) incidence matrix.
        - country_of_n (np.array): An array of length 512 indicating the country of each node.
        
        Returns:
        - np.array: The modified incidence matrix with international links cut.
        """
        incidence_matrix = self.init.link_data['incidence']
        adjacency_matrix = self.init.link_data['adjacency']
        country_of_n = self.init.country_of_n
        num_nodes, num_links = incidence_matrix.shape

    
        # Identify columns to keep where abs(incidence_matrix).sum(axis=0) == 2
        valid_links = np.where(np.abs(incidence_matrix).sum(axis=0) == 2)[0]
        incidence_matrix = incidence_matrix[:, valid_links]
        
        # Filter out international links
        cols_to_keep = []
        for link_idx in range(incidence_matrix.shape[1]):
            nodes = np.where(incidence_matrix[:, link_idx] != 0)[0]
            if len(nodes) == 2 and country_of_n[nodes[0]] == country_of_n[nodes[1]]:
                cols_to_keep.append(link_idx)
            else:
                # Remove the edge from the adjacency matrix
                node1, node2 = nodes if len(nodes) == 2 else (None, None)
                if node1 is not None and node2 is not None:
                    adjacency_matrix[node1, node2] = 0
                    adjacency_matrix[node2, node1] = 0
        
        # Compute the degree matrix (diagonal matrix with node degrees)
        incidence_matrix = incidence_matrix[:, cols_to_keep]
        degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
        # Compute the graph Laplacian matrix: L = D - A
        laplacian_matrix = degree_matrix - adjacency_matrix
    
        LaplacianPseudoInverse = np.linalg.pinv(laplacian_matrix , hermitian=True)
        PTDFMatrix = incidence_matrix .T @ LaplacianPseudoInverse # PDFT = Power Transfer Distribution Factors
       
        self.init.link_data = {
            'adjacency':adjacency_matrix,
            'incidence': incidence_matrix,
            #'laplacian': laplacian.values,
            #'LaplacianPseudoInverse':LaplacianPseudoInverse.values ,
            'PTDF': PTDFMatrix,
            #'graph':graph
            }
        self.init.l = self.init.link_data['incidence'].shape[1]

    
    def compute_nodal_transmission_capacity(self, kappa_trans, Method = "Half"):
        """
        Computes nodal transmission capacity using the incidence matrix
    
        Parameters:
        - PowerFlow (np.array): Array of power flows along links (shape: (num_links,)).
        Returns:
        - np.array: Array of nodal transmission capacities (shape: (num_nodes,)).
        """
        (max_flow, length) = kappa_trans
        incidence_matrix = self.init.link_data['incidence']
        num_nodes, num_links = incidence_matrix.shape

        if Method == "Half":
            # Initialize nodal capacity array
            nodal_capacity = np.zeros(num_nodes)
            
            for link_idx in range(num_links):
                # Get nodes connected by this link
                nodes = np.where(incidence_matrix[:, link_idx] != 0)[0]
                if len(nodes) != 2:
                    continue
                node1, node2 = nodes
                # Compute transmission capacity contribution
                transmission_capacity = length[link_idx] * max_flow[link_idx]
                nodal_capacity[node1] += transmission_capacity / 2
                nodal_capacity[node2] += transmission_capacity / 2
        elif Method == 'Load':
            transmission_capacity = np.multiply(length, max_flow)
            sum_trans = transmission_capacity.sum()
            nodal_capacity = sum_trans*(self.init.Mean_Load/self.init.Total_Load)
        
        return nodal_capacity.astype(np.float32)


    
 
    
 
    
 
    
 
    
