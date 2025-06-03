import pandas as pd
import numpy as np
import networkx as nx
import geopy.distance
#%%
class data_loader():
    def __init__(self):
        self.BusData = pd.read_csv('Data/bus.csv')
        self.LoadData = np.genfromtxt("Data/Load.csv", delimiter=",", dtype=np.float32)
        self.CF_solar = pd.read_csv('Data/CF_solar.csv').to_numpy(dtype=np.float32)
        self.CF_wind = pd.read_csv('Data/CF_wind.csv').to_numpy(dtype=np.float32)
        self.adjacency = pd.read_csv('Data/adjacency_matrix_dense.csv')
        

    def run(self):
        input_data = {
            'BusData': self.BusData,
            'Load': self.LoadData,
            'CF_solar': self.CF_solar,
            'CF_wind': self.CF_wind,
            'adjacency': self.adjacency}
        return input_data
    
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
        return{"country_of_n": self.country_of_n,
               "country_list": self.country_list,
               "n_country": self.n_country, "country_int": self.country_int}
    # def Nodal_CountrySum(self, data):
    #     try:
    #         country_int = self.country_int
    #     except AttributeError:
    #         _ = self.Country_data()
            
    # def Link_CountrySum(self, data):
    #     try:
    #         country_int = self.country_int
    #     except AttributeError:
    #         _ = self.Country_data()
        # return 
    
    def nodal_data(self):
        BusData = self.BusData
        BusNames = list(BusData['name'])
        BusPositions = {}
        for _, name in enumerate(BusNames):
            BusPositions[name] = (BusData['x'][_], BusData['y'][_])
        self.nodePos = BusPositions
        self.nodeName = BusNames
        return BusPositions, BusNames
    

    def link_data(self):
        BusData = self.BusData
        adjacency_data = self.adjacency
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
        
        # node_names = self.BusData['name'].to_numpy()
        # node_name = node_names[n]
        # node_country = init.country_of_n[n]
    
        # Determine link direction and name
        # for l in 
        # origin_index = np.where(incidence[:, l]  == -1)[0][0]
        # terminal_index = np.where(incidenc[:, l] == 1)[0][0]
        # origin_node = node_names[origin_index]
        # terminal_node = node_names[terminal_index]
        # txt1 = f"l  = '{origin_node} - {terminal_node}',"
        # BusNames = list(BusData['name'])
        #     BusPositions = {}
        #     for _, name in enumerate(BusNames):
        #         BusPositions[name] = (BusData['x'][_], BusData['y'][_])
        #     PowerFlow = flow_from_graph(graph, PowerInjected)
        #     Capacity = {}
        #     for column in PowerFlow:
        #         if Withlengths:
        #             point1 = BusPositions[column[0]][::-1]
        #             point2 = BusPositions[column[1]][::-1] 
        #             length = geopy.distance.distance(point1, point2).km
        #             Capacity[column] = PowerFlow[column].abs().quantile(0.99)*length
        #         else:
        #             Capacity[column] = PowerFlow[column].abs().quantile(0.99)

                
        #     edge_data = []        
        #     for edge, capacity in Capacity.items():
        #         (start, end) = edge
        #         edge_data.append({
        #             # 'start': BusPositions[start],
        #             # 'end': BusPositions[end],
        #             'capacity': capacity,
        #             'geometry': LineString([BusPositions[start], BusPositions[end]])})

    

    def link_groups_by_country(self, smalln =False):
        # Ensure country and graph data are loaded
        if not hasattr(self, 'country_of_n'):
            self.Country_data()
        if not hasattr(self, 'link_data_dict'):
            self.link_data_dict = self.link_data()
    
        # Get necessary data
        country_of_n = self.country_of_n
        graph = self.link_data_dict['graph']
        edge_list = list(graph.edges())
        node_list = list(graph.nodes())
    
        # Map node names to indices
        node_name_to_index = {name: idx for idx, name in enumerate(self.BusData['name'])}
        
        # Create a map from node to country
        node_to_country = {name: country_of_n[node_name_to_index[name]] for name in node_list}
        
        # Group edges by country pair
        group_dict = {}
        for i, (u, v) in enumerate(edge_list):
            c1 = node_to_country[u]
            c2 = node_to_country[v]
            country_pair = '-'.join(sorted([c1, c2]))  # 'DE-DK' or 'DE-DE'
            if country_pair not in group_dict:
                group_dict[country_pair] = []
            group_dict[country_pair].append(i)
    
        # Convert to arrays
        group_names = list(group_dict.keys())
        group_indices = [np.array(group_dict[name]) for name in group_names]
    
        return group_indices, group_names     

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



# def data_loader():
    
    
#     BusData = pd.read_csv('Data/bus.csv')
#     LoadData = np.genfromtxt("Data/Load.csv", delimiter=",", dtype=np.float32)
#     CF_solar = pd.read_csv('Data/CF_solar.csv').to_numpy(dtype=np.float32)
#     CF_wind = pd.read_csv('Data/CF_wind.csv').to_numpy(dtype=np.float32)
    
#     # Creation of NetworkX graph obejct of network
#     adjacency = pd.read_csv('Data/adjacency_matrix_dense.csv')

   
#     # Construct input data dictionary
#     input_data = {
#         'BusData': BusData,
#         'Load': LoadData,
#         'CF_solar': CF_solar,
#         'CF_wind': CF_wind,
#         'adjacency':adjacency
#     }
    
#     return input_data