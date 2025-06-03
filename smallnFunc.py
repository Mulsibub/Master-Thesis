# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:35:26 2025

@author: heide
"""

#%%


import numpy as np
import pandas as pd
import networkx as nx
import geopy.distance
from shapely.geometry import LineString
import geopandas as gpd
import matplotlib.colors as mcolors
import geoplot as gplt
import matplotlib as mpl
import textwrap
mpl.use('QtAgg')
from matplotlib import pyplot as plt
import os
# import os
# import copy
#%%
class Aggregator:
    def __init__(self, data_processor,  Plot_path='ResPlotsApr/'):
        self.Mapperinitialized = False
        self.init = data_processor
        self.init.Country_data()
        self.Ldata_n = self.init.link_dataF()
        self.BusData_c = self.Positions()
        self.group_indices, self.group_names = self.link_groups_by_country()
        self.adjacency, self.country_order = self.country_adjacency_matrix(weighted=False)
        self.PTDF_func()
        self.path = Plot_path

        
    def Positions(self):
        country_list = self.init.country_list
        Load_n = self.init.Load
        Sum_Load_n = Load_n.sum(axis=0)
        BusData_n = self.init.BusData
        Load_c = self.SumForC(Load_n)
        self.Load = Load_c
        Sum_Load_c = Load_c.sum(axis=0)
        x_c = self.SumForC((BusData_n.x.to_numpy()*Sum_Load_n).astype(np.float32))/Sum_Load_c
        y_c = self.SumForC((BusData_n.y.to_numpy()*Sum_Load_n).astype(np.float32))/Sum_Load_c
        BusData_c = pd.DataFrame({'name': country_list, 'x':x_c, 'y':y_c, 'country': country_list})
        return BusData_c
    
   
    def SumForC(self, Var_n, axis=1):
        country_list = self.init.country_list
        country_of_n = self.init.country_of_n
        country_int = self.init.country_int
        Var_c = []
        for country in country_list:
            start_index = np.where(country_of_n == country)[0][0]
            end_index = np.where(country_of_n == country)[0][-1]
            # Always append the tuple (start_index, end_index), even if they are the same
            temp = (start_index, end_index + 1)
            country_int.append(temp)  # Add +1 to make inclusive slicing easier
            if len(np.shape(Var_n))!= 2:
                Var_c.append(Var_n[start_index:end_index+1].sum())
            else:
                Var_c.append(Var_n[:,start_index:end_index+1].sum(axis=1))
        return np.array(Var_c).T
    
    def Run(self, layout_n):
        # for key in ['balancing', 'Flows', 'injection', 'mismatch']:
        #     layout['misc'][key] = layout['misc'][key][0:n_t,:]
        injection = self.SumForC(layout_n['misc']['injection'])
        balancing = self.SumForC(layout_n['misc']['balancing'])
        mismatch = self.SumForC(layout_n['misc']['mismatch'])
        Flows = injection @ self.PTDFMatrix.T.astype(np.float32)
        generation = {}
        for key, item in layout_n['Generation'].items():
            generation[key] = self.SumForC(item)
        kappa = {}
        for key, item in layout_n['kappa'].items():
            if key !='trans':
                kappa[key] = self.SumForC(item)
        kappatrans = np.quantile(np.abs(Flows), q=0.99, axis=0)
        kappa['trans'] = {'length': self.Ldata_c['length'], 'power':kappatrans}
                
        
        # Nodal transmission kappa
        Nodaltrans_Load = self.compute_nodal_transmission_capacity((kappatrans, self.Ldata_c['length']), Method="Load")
        Nodaltrans_Half = self.compute_nodal_transmission_capacity((kappatrans, self.Ldata_c['length']), Method="Half")
        
        # Country acumullations of transmission capacity
       
        misc = {
             'alpha_opt':layout_n['misc']['alpha_opt'],
             'balancing':balancing,
             'backup_energy':layout_n['misc']['backup_energy'],
             'mismatch':mismatch,
             'injection':injection,
             'Flows': Flows.to_numpy(),
             # 'nodal_trans_Load': layout_n['misc']['nodal_trans_Load'],
             # 'nodal_trans_Half': layout_n['misc']['nodal_trans_Half'],
             'country_trans_Load':Nodaltrans_Load,
             'country_trans_Half': Nodaltrans_Half
             }
        link_data = {
            'adjacency':self.adjacency,
            'incidence': self.incidence.values,
            #'laplacian': laplacian.values,
            #'LaplacianPseudoInverse':LaplacianPseudoInverse.values ,
            'PTDF': self.PTDFMatrix.values
            # ,'graph':self.Ldata_n['graph']
            }
        result = {
            'kappa':kappa,
            'Generation':generation,
            'misc':misc,
            'link_data': link_data ,
            'Load_C': self.Load
            }
        return result
    
    def link_dataF(self):
        BusData = self.BusData_c
        graph = self.graph
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
    
    def link_groups_by_country(self):
        # Get necessary data
        country_of_n = self.init.country_of_n
        graph = self.Ldata_n['graph']
        edge_list = list(graph.edges())
        node_list = list(graph.nodes())
    
        # Map node names to indices
        node_name_to_index = {name: idx for idx, name in enumerate(self.init.BusData['name'])}
        
        # Create a map from node to country
        node_to_country = {name: country_of_n[node_name_to_index[name]] for name in node_list}
        
        # Group edges by country pair
        group_dict = {}
        for i, (u, v) in enumerate(edge_list):
            c1 = node_to_country[u]
            c2 = node_to_country[v]
            if c1 != c2: #Remove Internal links like 'DE-DE'
                country_pair = '-'.join(sorted([c1, c2]))  # 'DE-DK'
                if country_pair not in group_dict:
                    group_dict[country_pair] = []
                group_dict[country_pair].append(i)
                
        # Convert to arrays
        group_names = list(group_dict.keys())
        group_indices = [np.array(group_dict[name]) for name in group_names]
    
        return group_indices, group_names
    
    def country_adjacency_matrix(self, weighted=False):
        """
        Constructs an adjacency matrix of the reduced graph where each node is a country.
        Args:
            weighted (bool): If True, matrix entries are the number of links between countries.
                             If False, matrix entries are 1 if any link exists.
        Returns:
            adj_matrix (np.ndarray): [n_countries x n_countries] adjacency matrix
            country_order (List[str]): country names in the order of the matrix
        """
        country_order = list(self.BusData_c['name'])
        country_to_index = {country: idx for idx, country in enumerate(country_order)}
        n = len(country_order)
        adj_matrix = np.zeros((n, n), dtype=int)
    
        for group_name, indices in zip(self.group_names, self.group_indices):
            c1, c2 = group_name.split('-')
            i, j = country_to_index[c1], country_to_index[c2]
            value = len(indices) if weighted else 1
            adj_matrix[i, j] = value
            adj_matrix[j, i] = value  # Symmetric

        return adj_matrix, country_order

    def PTDF_func(self):
        # adjacency_data = pd.DataFrame(self.adjacency,columns=).set_index(self.BusData_c['name'])
        # adjacency_data = adjacency_data.set_columns(self.BusData_c['name'])
        graph = nx.from_numpy_array(self.adjacency, create_using=nx.Graph(), nodelist= self.BusData_c['name'])
        for edge in graph.edges:
            graph.edges[edge]['weight']=1
        self.graph=graph
            
        # Graph matrices
        # NB: Random orientation of incidence matrix since graph edges are undirected
        self.incidence = pd.DataFrame(nx.incidence_matrix(graph, oriented=True).todense(), index=graph.nodes, columns=list(graph.edges))
        laplacian = pd.DataFrame(nx.laplacian_matrix(graph).todense(), index=graph.nodes, columns=graph.nodes)
        LaplacianPseudoInverse = pd.DataFrame(np.linalg.pinv(laplacian, hermitian=True), index=graph.nodes, columns=graph.nodes)
        self.PTDFMatrix = self.incidence.T @ LaplacianPseudoInverse # PDFT = Power Transfer Distribution Factors
        
        
        self.Ldata_c = self.link_dataF()
        
    def compute_nodal_transmission_capacity(self, kappa_trans, Method = "Half"):
        """
        Computes nodal transmission capacity using the incidence matrix
    
        Parameters:
        - PowerFlow (np.array): Array of power flows along links (shape: (num_links,)).
        Returns:
        - np.array: Array of nodal transmission capacities (shape: (num_nodes,)).
        """
        Load = self.Load
        Mean_Load = Load.mean(axis=0)
        Total_Load = Mean_Load.sum()
        Mean_Load
        Total_Load
        (max_flow, length) = kappa_trans
        incidence_matrix = self.incidence.to_numpy()
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
            nodal_capacity = sum_trans*(Mean_Load/Total_Load)
        
        return nodal_capacity.astype(np.float32)
    
    #%% Map Functions
    def InitializeMapper(self, lowresmap = False, dpi=200, plot_path=''):
        self.proj = gplt.crs.Miller(central_longitude=0)
        MinLon, MaxLon = -10, 31.5 # -15, 35
        MinLat, MaxLat = 35.7, 68 # 35.7, 70
        self.extent = (MinLon, MinLat, MaxLon, MaxLat)
        self.dpi = dpi
        BusData = self.BusData_c
        gdf = gpd.GeoDataFrame(BusData, geometry=gpd.points_from_xy(BusData.x, BusData.y)) 
        self.DefineMap()
        self.gdf = gdf
        self.plot_path = plot_path
        self.Mapperinitialized = True
        
        
    def MapOverview(self, Title='', figname = 'SmallnOverview', subpath=''):
        if not self.Mapperinitialized:
            self.InitializeMapper()
        proj = self.proj
        Map = self. Map
        extent = self.extent
        gdf = self.gdf
        BusData = self.BusData_c
        edge_data, edge_gdf = self.edge_dataPrep()
        fig, axs = plt.subplots(figsize=(9,8), subplot_kw={'projection':proj})
        wrapped_title = "\n".join(textwrap.wrap(Title, 80))
        gplt.polyplot(Map, ax=axs, extent=extent, zorder=2, linewidth=0.3,
                      edgecolor='k')
        gplt.sankey(edge_gdf.geometry, projection=proj, zorder=3, ax=axs,
                    linewidth=0.8)

        xtran = gplt.crs.ccrs.Miller(central_longitude=0)
        for i in gdf.index:
            x=gdf.iloc[i]
            xy = xtran.transform_point(x.geometry.centroid.coords[0][0], x.geometry.centroid.coords[0][1], gplt.crs.ccrs.PlateCarree())
            axs.annotate(
                text=str(BusData.name[i]),
                xy=xy,  # Use the transformed coordinates.
                fontsize=6,
                # fontstretch='ultra-expanded',
                ha='center',
                bbox={'facecolor': 'white', 'alpha':0.3, 'pad': 0.1, 'edgecolor':'none'},
                color='red')
        
        fig.suptitle(wrapped_title, fontsize=14, color='Black')
        fig.tight_layout()  # Adjust layout to ensure everything fits without overlap
        path = self.plot_path + subpath
        if path and figname:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+ figname + '.png', dpi=self.dpi,  bbox_inches='tight')
            print(f"Figure saved as {path+ figname + '.png'}")
        plt.close()

    def SingleMapHandler(self, layout, name, Normalize=False, mode = 'div', colormap = 'jet', subpath=''):
        if not self.Mapperinitialized:
            self.InitializeMapper()
        if Normalize:
            types = ['solar', 'wind', 'backup']
        else:
            types = ['solar', 'wind', 'backup', 'backup_energy']
        Tit1 = {'solar':r'Solar Capacity, ','wind': r'Wind Capacity, ','backup': r'Backup Capacity, ','backup_energy': r'Backup Energy, '}
        Tit3 =  {'Global_Global_1.0': r'Global CFProp with Global Synchronized Balancing',
                 # 'Global_Local_1.0': r'Global CFProp with Local Synchronized Balancing',
                 'Global_Nodal_1.0': r'Global CFProp with Nodal Balancing',
                 # 'Global_NoT_1.0': r'Global CFProp with NoT Balancing',
                 'Local_Global_1.0': r'Local CFProp with Global Synchronized Balancing',
                 'Local_Local_1.0': r'Local CFProp with Local Synchronized Balancing',
                 'Local_Nodal_1.0': r'Local CFProp with Nodal Balancing',
                 'Local_NoT_1.0': r'Local CFProp with NoT Balancing',
                 'LocalGAS_Nodal_1.0': r'GAS Layout with Nodal Balancing',
                 'LocalGAS_Global_1.0': r'GAS Layout with Global Synchronized Balancing'}
        for j in types:
            data = (layout['kappa'][j])
            if Normalize:
                try:
                    data = data/self.MeanLoad
                except AttributeError:
                    self.MeanLoad = self.Load.mean(axis=0)
                    data = data/self.MeanLoad
                figname = name+'_kappa'+j+'_Normalized'
                Title = 'Normalized ' + Tit1[j] + Tit3[name]
            else:
                figname = name+'_kappa'+j
                Title = Tit1[j] + Tit3[name]
            path = self.path+subpath+j+'/'
            self.SingleMap(data, figname, Title, path, Normalize, mode, colormap)       
                 
        
                
    def SingleMap(self, data, figname, Title, path, Normalize, mode, colormap):
        if not self.Mapperinitialized:
            self.InitializeMapper()
        gdf = self.gdf
        # mode = self.mode
        Map = self.Map
        Map2 = self.Map2
        proj = self.proj
        extent = self.extent
        
        fig, axs = plt.subplots(figsize=(9,8), subplot_kw={'projection':proj})
        wrapped_title = "\n".join(textwrap.wrap(Title, 80))
        fig.suptitle(wrapped_title, fontsize=14, color='Black')
        cmap = plt.cm.get_cmap(colormap).copy()
        
        
        # Handling Colors
        vmax = max(data)
        vmin = min(data)
        vcenter=(vmin + vmax) / 2
        
        # Apply normalization based on mode
        if mode == 'div':
            cnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=vcenter)
        elif mode == 'div_new':
            cnorm = mcolors.TwoSlopeNorm(vmin=vmin,vmax=vmax,vcenter=vcenter)
        elif mode =='help':
            cnorm = mcolors.CenteredNorm(vcenter=vcenter)
        elif mode == 'fixed':
            cnorm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        elif mode == 'log':
            cnorm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f"Unsupported bounds option: {mode}")
        cmap.set_bad(color='Black')
        if Normalize:
            form = '%.1f'
        else:
            form = '%.0f'
            
        tick_place = [vmin, vcenter, vmax]
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                            orientation='vertical', pad=0.01, ax=axs,
                            fraction=0.04, ticks=tick_place, format=form,
                            spacing='proportional')
        tick_labels = [form % x for x in tick_place]
        cbar.ax.set_yticklabels(tick_labels, fontsize=12)
        if not Normalize:
            cbar.ax.set_ylabel(r'[kW]')
        
        
        gplt.polyplot(Map, ax=axs, extent=extent, zorder=2, linewidth=0.6)
        gplt.choropleth(Map2, ax=axs, extent=extent, zorder=3, linewidth=0.1,
                        hue=data, projection=proj, edgecolor='w',
                        cmap=cmap, norm=cnorm)

        #Format and saving
        wrapped_title = "\n".join(textwrap.wrap(Title, 80))
        fig.suptitle(wrapped_title, fontsize=14, color='Black')
        fig.tight_layout()  # Adjust layout to ensure everything fits without overlap
        if path and figname:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+ figname + '.png', dpi=self.dpi,  bbox_inches='tight')
            print(f"Figure saved as {path+ figname + '.png'}")
        plt.close()
        
    def edge_dataPrep(self, capacity=None):
        linkdata = self.Ldata_c
        edge_data = []
        for pos in linkdata['positions']:
                edge_data.append({
                    'geometry': LineString([pos[0,:], pos[1,:]])})
        edge_gdf = gpd.GeoDataFrame(edge_data)
        return edge_data, edge_gdf
                      
    def DefineMap(self):
        Map = gpd.read_file('plots/Europe/Europe.shp')
                # Map=Map.to_crs(epsg=4326) # overrides old projection, but europe-shape already is 4326
        Map = Map[~Map['NAME'].isin(['Russia','Moldova','Ukraine','Turkey','Belarus'])]
        Map = Map.simplify(0.07) #Helps a lot
        Map = gpd.read_file('plots/Europe/Europe.shp')
        Map = Map[~Map['NAME'].isin(['Russia','Moldova','Ukraine','Turkey',
                                     'Belarus', 'Faeroe Islands (Denmark)',
                                     'Jan Mayen (Norway)', 'Svalbard (Norway)',
                                     'Azerbaijan', 'Armenia', 'Georgia',
                                     'Iceland', 'Malta'])]
        Map2 = Map[~Map['NAME'].isin(['Gibraltar (UK)', 'Monaco',
                                     'Guernsey (UK)','San Marino',
                                     'Isle of Man (UK)', 'Liechtenstein',
                                     'Jersey (UK)', 'Andorra'])]
        Map2= Map2.set_index('NAME')
        name_to_iso2 = {
            'Albania': 'AL', 'Austria': 'AT', 'Bosnia Herzegovina': 'BA', 'Belgium': 'BE',
            'Bulgaria': 'BG', 'Switzerland': 'CH', 'Czech Republic': 'CZ', 'Germany': 'DE',
            'Denmark': 'DK', 'Estonia': 'EE', 'Spain': 'ES', 'Finland': 'FI', 'France': 'FR',
            'United Kingdom': 'GB', 'Greece': 'GR', 'Croatia': 'HR', 'Hungary': 'HU',
            'Ireland': 'IE', 'Italy': 'IT', 'Lithuania': 'LT', 'Luxembourg': 'LU',
            'Latvia': 'LV', 'Montenegro': 'ME', 'Macedonia': 'MK', 'Netherlands': 'NL',
            'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Serbia': 'RS',
            'Sweden': 'SE', 'Slovenia': 'SI', 'Slovakia': 'SK'}
        name, geom = list(), list()
        for key, item in name_to_iso2.items():
            geom.append( Map2['geometry'][key])
            name.append(item) 
        Map2 = gpd.GeoDataFrame(pd.DataFrame({'NAME': name, 'geometry':geom}))
        
        Map = Map.simplify(0.07)
        Map2 = Map2.simplify(0.07)
        self.Map = Map
        self.Map2 = Map2
#%%
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))
        
        
  
