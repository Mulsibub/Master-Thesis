
import numpy as np
import scipy.sparse as sparse
import time


from flowtracing.source_to_sink import source_to_sink_class
from flowtracing.sink_to_source import sink_to_source_class
from post_processing.PlotFunctions import FTPlotter
#%%




class flowtracing_class:
    """
    Initialization
    """
    def __init__(self,init,layout, n_bins=50, Import = False, path='FTPlots/', smalln=False):
        n_t = init.t
        for key, item in layout['Generation'].items():
            layout['Generation'][key] = item[0:n_t,:]
        for key in ['balancing', 'Flows', 'injection', 'mismatch']:
            layout['misc'][key] = layout['misc'][key][0:n_t,:]

        if Import:
            self.Injections = layout['misc']['injection'][0:n_t,:]
        else:
            self.Injections = -layout['misc']['injection'][0:n_t,:]
            
        if not smalln:    
            self.link_data = init.link_data
            init.Load2 = init.Load[0:n_t,:]
            self.Load2 = init.Load2
        else:
            self.link_data = layout['link_data']
            init.Load2 = layout['Load_C'][0:n_t,:]
        self.smalln = smalln    
  
        
        self.PTDF = self.link_data['PTDF'].astype(np.float32)
        self.incidence = sparse.csr_matrix(self.link_data['incidence']).astype(np.float32)
        self.adjacency = self.compute_adjacency_matrix(self.incidence).astype(np.float32)
        self.FlowPattern = self.Injections @ self.PTDF.T.astype(np.float32)
        self.t_total = n_t
        
        #Defining the clases
        self.init = init
        self.layout = layout
        
        #Init result matrices
        (L,N) = np.shape(self.PTDF)
        self.total_export = np.zeros((N, N)).astype(np.float32)
        self.total_export_wind = np.zeros_like(self.total_export).astype(np.float32)
        self.total_export_solar = np.zeros_like(self.total_export).astype(np.float32)
        self.total_export_backup = np.zeros_like(self.total_export).astype(np.float32)
        self.total_use_trans = np.zeros((L,N)).astype(np.float32)
        

        self.L, self.N = L, N
        self.n_bins=n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.F2 = np.abs(self.FlowPattern)
        self.kappa_T = np.quantile(self.F2, q=0.99, axis=0).astype(np.float32)
        norm_flow = np.divide(self.F2, self.kappa_T)
        self.norm_flow = np.clip(norm_flow, 0, 1.0 - 1e-8)
        bin_indices = np.floor(np.multiply(self.norm_flow, n_bins-1))
        self.bin_indices = (bin_indices).astype(np.int32)
        self.sum_q = np.zeros((N, L, n_bins), dtype=np.float32)  # accumulate sums
        self.count_q = np.zeros((L, n_bins), dtype=np.int32)     # accumulate counts
        self.path = path
        self.Import=Import
        
        
    def compute_adjacency_matrix(self,incidence):
        """Compute the adjacency matrix using incidence matrix multiplication."""
        adjacency_matrix = incidence @ incidence.T
        adjacency_matrix.setdiag(0)
        return adjacency_matrix.todense()
    
    
    def run_source_to_sink(self, t):
        
        model = source_to_sink_class(self.adjacency,self.F_in,self.F_out,self.F_out_total,self.P_minus,self.P_plus)
        
        walk = model.walk()
        if walk == 'NoFlow':
            q_nn = sparse.csr_matrix(np.diag(np.ones(len(self.F_out_total))))
        else:
            q_nn = model.q_nn(walk)


        q_wind, q_solar, q_backup = model.partition_q(t,q_nn,self.layout,self.init.Load2[t,:])
        return q_nn, q_wind, q_solar, q_backup
        

    def run_sink_to_source(self, t):
        model = sink_to_source_class(self.adjacency,self.F_in,self.F_out,self.F_in_total,self.P_minus,self.P_plus)
        walk = model.walk()
        q_nn = model.q_nn(walk)
        q_ln = model.q_ln(walk,q_nn)
        return q_ln#, q_nn, q_ln ,q_wind,q_solar,q_backup
    
    
    def Initialize_time_step(self, t):
        """Initializes time step variables and stores them as instance attributes."""
        
        # Compute flow matrix and its positive/negative components
        F = self.incidence.multiply(sparse.csr_matrix(self.FlowPattern[t, :].T)).T
        F_out, F_in = F.multiply(F > 0), -F.multiply(F < 0)
    
        # Compute injection values and their positive/negative components
        P = self.Injections[t, :]
        P_plus, P_minus = P * (P > 0), -P * (P < 0)
    
        # Compute total flow in and out
        F_out_total, F_in_total = np.asarray(F_out.sum(axis=0)).ravel(), np.asarray(F_in.sum(axis=0)).ravel()
    
        # Store all computed values as instance attributes
        for var_name in ["F", "F_out", "F_in", "P", "P_plus", "P_minus", "F_out_total", "F_in_total"]:
            setattr(self, var_name, locals()[var_name])
    
    """
    Runing the script
    """
        
    def run(self):
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""
        Import = self.Import
        t_total = self.t_total
        # condprop = np.zeros((956,512))
        # Prepare accumulation arrays for special plots
        if not self.smalln:
            Q_ln1 = np.zeros((t_total), dtype=np.float32)
            l1, n1 = 270, 281 # Viking link and London
            Q_ln2 = np.zeros((t_total), dtype=np.float32)
            l2, n2 = 280, 172 # balearic link and node
            Q_ln3 = np.zeros((t_total), dtype=np.float32)
            l3, n3 = 271, 129 # Great Belt link and Aarhus and Fyn
        # bins = np.linspace(0,1,100)
        # norm_flow = self.norm_flow
        bin_indices = self.bin_indices
        total_export = self.total_export
        total_export_wind = self.total_export_wind
        total_export_solar = self.total_export_solar
        total_export_backup = self.total_export_backup
        start = time.time()
        for t in range(t_total):
            self.Initialize_time_step(t) #Get F,P and their variations for t
            # P = self.P
            if Import:
                P = self.P_minus
                q_nn, q_wind, q_solar, q_backup = self.run_source_to_sink(t)
                total_export += q_nn * P
                total_export_wind += q_wind * P
                total_export_solar += q_solar * P
                total_export_backup += q_backup * P
    
            q_ln = self.run_sink_to_source(t)
            if not self.smalln:
                Q_ln1[t] = q_ln[l1, n1]
                Q_ln2[t] = q_ln[l2, n2]
                Q_ln3[t] = q_ln[l3, n3]
            
            Bin_ind =  bin_indices[t,:]
            self.CondAvg_acumulate(q_ln, Bin_ind)
        
        end = time.time()
        length = end - start 
        

        cond_avg = self.CondAvg_condition()
        # cond_avg = self.CondAvg_condition_Reduced(NewBinCount = 50)

        K_ln_T = self.compute_K_ln_T(cond_avg)
        Res = {
            'total' : total_export,
            'wind' : total_export_wind,
            'solar' : total_export_solar,
            'backup' : total_export_backup,
            'trans': K_ln_T
        }
        # Plotting
        if not self.smalln:
            plotter = FTPlotter(self.init, path=self.path)
            kappa_l_T = np.quantile(self.F2, q=0.99, axis=0).astype(np.float32)
    
            if self.Import:
                Tit = self.init.Layout_Scheme+'_'+self.init.Balancing_scheme+'_Import'
            else:
                Tit = self.init.Layout_Scheme+'_'+self.init.Balancing_scheme+'_Export'
            Tit2 = Tit.replace("_", " ")
            print(f'{length/t_total} seconds per timestep for '+Tit2)
            plotter.CountryUsage(K_ln_T, kappa_l_T, Title=Tit2, figname=Tit, subpath='Heatmap/')
            plotter.scattermean(self.norm_flow[:,l1], Q_ln1, cond_avg[n1,l1,:], l1, n1, Title=Tit2+' View, n = London, l = Viking Link',
                        figname='LondonVik_'+Tit, subpath='Scattermean/London/')
            plotter.scattermean(self.norm_flow[:,l2], Q_ln2, cond_avg[n2,l2,:], l2, n2, Title=Tit2+' View, Balearic Islands Link and Node.',
                        figname='balearic'+Tit, subpath='Scattermean/Balearic/')
            plotter.scattermean(self.norm_flow[:,l3], Q_ln3, cond_avg[n3,l3,:], l3, n3, Title=Tit2+' View, n = Aarhus and Fyn Node, l= Great Belt Link',
                        figname='GreatBelt'+Tit, subpath='Scattermean/GreatBelt/')
        if self.smalln:
            plotter = FTPlotter(self.init, path=self.path)
            if self.Import:
                Tit = self.init.Layout_Scheme+'_'+self.init.Balancing_scheme+'_Import'
            else:
                Tit = self.init.Layout_Scheme+'_'+self.init.Balancing_scheme+'_Export'
            Tit2 = Tit.replace("_", " ")
            print(f'{length/t_total} seconds per timestep for '+Tit2)
            plotter.smallngroups(self.layout, K_ln_T, Title=Tit2, figname=Tit, subpath='Heatmap/')
        return Res
       
        
       

    """ 
    Computing the cond_avg
    """    
    
    def CondAvg_acumulate(self, q_ln, Bin_ind):
        sum_q =  self.sum_q
        count_q = self.count_q
        L = self.L
        for l in range(L):
            b = Bin_ind[l]
            sum_q[:, l, b] += q_ln[l, :]
            count_q[l, b] += 1
        self.sum_q = sum_q
        self.count_q = count_q
        
    def CondAvg_condition_Reduced(self, NewBinCount = 50):
        n_dims = int(self.n_bins/NewBinCount)
        sum_q = self.sum_q.reshape(512, 956, NewBinCount, n_dims).sum(axis=3)
        count_q = self.count_q.reshape(956, NewBinCount, n_dims).sum(axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            cond_avg = sum_q / count_q[None, :, :]
        cond_avg = np.nan_to_num(cond_avg, nan=(0))
        if np.allclose(cond_avg.sum(axis=0), 1, rtol=1e-1):
            print('Number of bins is suitable')
            self.n_bins = NewBinCount
            return cond_avg.astype(np.float32)
        else:
            print('Smaller number of bins Req')
            
    def CondAvg_condition(self):

        with np.errstate(divide='ignore', invalid='ignore'):
            cond_avg = self.sum_q / self.count_q[None, :, :]
        # Replace NaNs from division by zero with 0
        cond_avg = np.nan_to_num(cond_avg, nan=(0))
        # cond_avg = np.nan_to_num(cond_avg, nan=0.0)
        # Convert to float32
        return cond_avg.astype(np.float32)        
    
    
    def compute_K_ln_T(self, cond_avg):
        """
        Compute K_ln^T based on the given equation using the conditional average.
        
        Parameters:
            cond_avg: np.ndarray, shape (956, 512, 100), conditional average <c_ln | f_l>
            
        Returns:
            K_ln_T: np.ndarray, shape (956, 512), effective capacity per node-link
        """

        
        
        # temp=np.sum(cond_avg,axis=0)
        n_bins = self.n_bins
        N, L = self.N, self.L
       
        
        norm_flow=self.norm_flow
        bin_edges = np.linspace(0,1,n_bins+1)
        bins=np.zeros(n_bins)
        for k in range(n_bins):
            bins[k] = (bin_edges[k]+bin_edges[k+1])/2
            
        # Propability distribution of F_l: P(F_l)
        P = np.zeros((n_bins, L))
        for l in range(L):
            hist, _ = np.histogram(norm_flow[:, l], bins=n_bins, range=(0,1))
            P[:, l] = hist / norm_flow.shape[0]  # normalize by number of time steps
            
        # # Cumulative propability distribution of F_l: P_c(F_l)
        P_c = np.zeros((n_bins, L))
        for k_idx in range(n_bins):
                P_c[k_idx, :] = P[:k_idx, :].sum(axis=0) #+P[k_idx, :]

        kappa_l_T = np.quantile(self.F2, q=0.99, axis=0).astype(np.float32)
        delta_kappa =  kappa_l_T/ n_bins  
        
        def compute_K_ln_T_vectorized(P_l, c_ln, P_l_c, dkappa, kappa_T):
            n_bins, L = P_l.shape
            N = c_ln.shape[0]
            K_ln_T = np.zeros((N, L))
            # bin_T = np.minimum(n_bins, np.floor(kappa_T / dkappa).astype(int))

            for l in range(L):
                denom = np.clip(1 - P_l_c[:, l], 1e-8, None)

                integrand = P_l[:, l] * c_ln[:, l, :]
                integrand_rev_cumsum = np.cumsum(integrand[:, ::-1], axis=1)[:, ::-1]

                weights = dkappa[l] / denom
                K_ln_T[:, l] = np.dot(integrand_rev_cumsum, weights)

            return K_ln_T

        K_ln_T = compute_K_ln_T_vectorized(P, cond_avg, P_c, delta_kappa, kappa_l_T)

        #Uncomment to verify validity
        # temp2=K_ln_T.sum(axis=0)/ kappa_l_T
        # np.allclose(K_ln_T.sum(axis=0), kappa_l_T, rtol=1e-7)
        return K_ln_T.astype(np.float32) 





