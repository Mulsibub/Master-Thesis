import numpy as np
import pandas as pd
import networkx as nx
import geopy.distance

class data_class:
    def __init__(self, data, input_parameters):
        self.input_parameters = input_parameters
        self.data = data
        
    def Initial_data(self):    
        # Input Parameters
        self.Layout_Scheme = self.input_parameters['Layout_Scheme']
        self.Balancing_scheme = self.input_parameters['Balancing_Scheme']
        self.gamma = np.float32(self.input_parameters['gamma'])
        self.beta = np.float32(self.input_parameters['beta'])
        self.alpha_opt_parameter = self.input_parameters['alpha_opt_parameter']
        self.n_alpha = self.input_parameters['n_alpha']

        
        if self.alpha_opt_parameter == 'critical_parameter':
            self.choice_of_critical_parameter = self.input_parameters['choice_of_critical_parameter']
        
        # Input Data
        self.t = self.input_parameters['n_t']
        # self.Load = self.data['Load'][0:self.t,:]
        # self.Mean_Load = self.Load.mean(axis=0)
        # self.Total_Load = self.Mean_Load.sum()
        # self.CF_wind = self.data['CF_wind'][0:self.t,:]
        # self.CF_solar = self.data['CF_solar'][0:self.t,:]
        self.BusData = self.data['BusData']
        # self.n = self.Mean_Load.shape[0]
        
        self.Load = self.data['Load']
        self.CF_wind = self.data['CF_wind']
        self.CF_solar = self.data['CF_solar']
        self.Mean_Load = self.Load.mean(axis=0)
        self.Total_Load = self.Mean_Load.sum()
        self.n = self.Mean_Load.shape[0]
        
        
        

        
    # def Country_data(self):
    #     #########################
    #     #Input data for countries
    #     #########################
        
        
    #     self.country_of_n = self.BusData['country'].to_numpy()  # Convert to NumPy array
    #     self.country_list = np.unique(self.country_of_n)
    #     self.country_int = []
    #     for country in self.country_list:
    #         start_index = np.where(self.country_of_n == country)[0][0]
    #         end_index = np.where(self.country_of_n == country)[0][-1]
            
    #         # Always append the tuple (start_index, end_index), even if they are the same
    #         self.country_int.append((start_index, end_index + 1))  # Add +1 to make inclusive slicing easier
            
    #     self.n_country = len(self.country_list)
    def Country_data(self):
        #########################
        #Input data for countries
        #########################
        self.country_of_n = self.BusData['country'].to_numpy()  # Convert to NumPy array
        self.country_list = np.unique(self.country_of_n)
        self.country_int = []
        for country in self.country_list:
            start_index = np.where(self.country_of_n == country)[0][0]
            end_index = np.where(self.country_of_n == country)[0][-1]
            
            # Always append the tuple (start_index, end_index), even if they are the same
            self.country_int.append((start_index, end_index + 1))  # Add +1 to make inclusive slicing easier
        self.n_country = len(self.country_list)
        
        
    def nodal_data(self):
        BusData = self.BusData
        BusNames = list(BusData['name'])
        BusPositions = {}
        for _, name in enumerate(BusNames):
            BusPositions[name] = (BusData['x'][_], BusData['y'][_])
        self.nodePos = BusPositions
        self.nodeName = BusNames
    

    def link_dataF(self):
        BusData = self.BusData
        # try:
        #     adjacency_data = self.adjacency
        # except AttributeError:
        #     self.PTDF_func()
        #     adjacency_data = self.link_data['adjacency']
        adjacency_data = self.data['adjacency']
        adjacency_data = adjacency_data.set_index('name')
        graph = nx.from_pandas_adjacency(adjacency_data, create_using=nx.Graph())
        for edge in graph.edges:
            graph.edges[edge]['weight']=1
        incidence = pd.DataFrame(nx.incidence_matrix(graph, oriented=True).todense(), index=graph.nodes, columns=list(graph.edges))
        l = incidence.shape[1]

        length = np.zeros(l)
        pos = np.zeros((l,2,2))
        inc= incidence.values
        for i in range(l):
            pos[i,:,:] = BusData.iloc[np.where(inc[:,i]!=0)[0],:][['x','y']].to_numpy()
            length[i] = geopy.distance.distance(pos[i,0,:], pos[i,1,:]).km
        length = length
        
        return {'length': length, 'positions': pos, 'incidence':incidence, 'graph':graph}


    def PTDF_func(self):
            
        
        adjacency_data = self.data['adjacency']
        adjacency_data = adjacency_data.set_index('name')
        graph = nx.from_pandas_adjacency(adjacency_data, create_using=nx.Graph())
        for edge in graph.edges:
            graph.edges[edge]['weight']=1
        self.link_data = {
            'adjecenty':adjacency_data,
            'graph':graph
            }
            
        # Graph matrices
        # NB: Random orientation of incidence matrix since graph edges are undirected
        incidence = pd.DataFrame(nx.incidence_matrix(graph, oriented=True).todense(), index=graph.nodes, columns=list(graph.edges))
        laplacian = pd.DataFrame(nx.laplacian_matrix(graph).todense(), index=graph.nodes, columns=graph.nodes)
        LaplacianPseudoInverse = pd.DataFrame(np.linalg.pinv(laplacian, hermitian=True), index=graph.nodes, columns=graph.nodes)
        PTDFMatrix = incidence.T @ LaplacianPseudoInverse # PDFT = Power Transfer Distribution Factors
        
        self.link_data = {
            'adjacency':adjacency_data.values,
            'incidence': incidence.values,
            #'laplacian': laplacian.values,
            #'LaplacianPseudoInverse':LaplacianPseudoInverse.values ,
            'PTDF': PTDFMatrix.values,
            #'graph':graph
            }
        self.l = self.link_data['incidence'].shape[1]
    
    def run(self):
        self.Initial_data()
        self.Country_data() 
        self.PTDF_func()
        
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
        
        num_nodes, num_links = incidence_matrix.shape

        # Iterate over all links (columns in incidence matrix)
        for link_idx in range(num_links):
            # Get the nodes connected by this link
            nodes = np.where(incidence_matrix[:, link_idx] != 0)[0]
            
            if len(nodes) != 2:
                # Skip if link does not connect exactly two nodes
                continue
            
            node1, node2 = nodes
            
            # Check if the nodes belong to different countries
            if country_of_n[node1] != country_of_n[node2]:
                # Cut international link by zeroing out the column in incidence matrix
                incidence_matrix[:, link_idx] = 0
                # Remove the edge from the adjacency matrix
                adjacency_matrix[node1, node2] = 0
                adjacency_matrix[node2, node1] = 0
        
        # Compute the degree matrix (diagonal matrix with node degrees)
        degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
        
        # Compute the graph Laplacian matrix: L = D - A
        laplacian_matrix = degree_matrix - adjacency_matrix

        LaplacianPseudoInverse = np.linalg.pinv(laplacian_matrix , hermitian=True)
        PTDFMatrix = incidence_matrix .T @ LaplacianPseudoInverse # PDFT = Power Transfer Distribution Factors
       
        self.link_data = {
            'adjacency':adjacency_matrix,
            'incidence': incidence_matrix,
            #'laplacian': laplacian.values,
            #'LaplacianPseudoInverse':LaplacianPseudoInverse.values ,
            'PTDF': PTDFMatrix,
            #'graph':graph
            }
        self.l = self.link_data['incidence'].shape[1]
        
        

    