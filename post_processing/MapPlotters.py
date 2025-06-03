# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:19:20 2025

@author: heide
"""
#%%
# import pandas as pd
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
mpl.use('QtAgg')
from matplotlib import pyplot as plt
# import seaborn as sns


import textwrap
import matplotlib.colors as mcolors
import geopandas as gpd
import geoplot as gplt
from scipy.spatial import Voronoi
from shapely.ops import polygonize, unary_union



# import geoplot.crs as gcrs
import os

from shapely.geometry import LineString

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

#%%
class Mapper():
    def __init__(self, data_class, path='plots/', lowresmap =False, dpi=200):
        data = data_class.run()
        BusData = data['BusData']
        lowresmap = False
        gdf = gpd.GeoDataFrame(BusData, geometry=gpd.points_from_xy(BusData.x, BusData.y)) 
        self.DefineMap()
        
        # LoadData = data['Load'] 
        self.MeanLoad = data['Load'].mean(axis=0)    
        self.gdf = gdf
        self.path = path
        self.proj = gplt.crs.Miller(central_longitude=0)
        MinLon, MaxLon = -10, 31.5 # -15, 35
        MinLat, MaxLat = 35.7, 68 # 35.7, 70
        self.extent = (MinLon, MinLat, MaxLon, MaxLat)
        self.dpi =dpi
        self.data = data
        self.data_class = data_class

        
    def SingleMapHandler(self, Dict, Normalize=False, mode = 'div', colormap = 'jet'):
        if Normalize:
            types = ['solar', 'wind', 'backup']
        else:
            types = ['solar', 'wind', 'backup', 'backup_energy']
        Tit1 = {'solar':r'Solar Capacity, ','wind': r'Wind Capacity, ','backup': r'Backup Capacity, ','backup_energy': r'Backup Energy, '}
        Tit3 =  {'Global_Global_1.0': r'Global CFProp $\beta =1$, with Global Synchronized Balancing',
                 # 'Global_Local_1.0': r'Global CFProp $\beta =1$, with Local Synchronized Balancing',
                 'Global_Nodal_1.0': r'Global CFProp $\beta =1$, with Nodal Balancing',
                 # 'Global_NoT_1.0': r'Global CFProp $\beta =1$, with NoT Balancing',
                 'Local_Global_1.0': r'CB CFProp $\beta =1$, with Global Synchronized Balancing',
                 'Local_Local_1.0': r'CB CFProp $\beta =1$, with Local Synchronized Balancing',
                 'Local_Nodal_1.0': r'CB CFProp $\beta =1$, with Nodal Balancing',
                 'Local_NoT_1.0': r'Local CFProp $\beta =1$, with NoT Balancing',}
                          
        for i in Tit3.keys():
            for j in types:
                data = (Dict[i]['Result']['kappa'][j])
                if Normalize:
                    data = data/self.MeanLoad
                    figname = i+'_kappa'+j+'_Normalized'
                    Title = 'Normalized ' + Tit1[j] + Tit3[i]
                else:
                    figname = i+'_kappa'+j
                    Title = Tit1[j] + Tit3[i]
                path = self.path+j+'/'
                self.SingleMap(data, figname, Title, path, Normalize, mode, colormap)
                
    def SingleMap(self, data, figname, Title, path, Normalize, mode, colormap):
        gdf = self.gdf
        # mode = self.mode
        Map = self.Map
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
        # cbar.ax.set_ylabel(r'[KW]')
        
        
        gplt.polyplot(Map, ax=axs, extent=extent, zorder=2, linewidth=0.4)
        gplt.voronoi(gdf, clip=self.Map2, hue=data, projection=proj,
                      cmap=cmap, norm=cnorm, extent=extent, edgecolor='w', linewidth=0,
                      ax=axs)
        wrapped_title = "\n".join(textwrap.wrap(Title, 80))
        fig.suptitle(wrapped_title, fontsize=16, color='Black')   
        # fig.tight_layout()  # Adjust layout to ensure everything fits without overlap
        fig.savefig(path+figname+'.png', dpi=self.dpi)
        plt.close()
        
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
        
        
    def Network_Overview(self, subpath='', figname='Network_Overview', Title= 'Overview of Network Model'):        
        Map = self.Map
        proj = self.proj
        gdf = self.gdf
        extent = self.extent
        try:
            edge_gdf = self.edge_gdf
        except AttributeError:
            self.edge_dataPrep(capacity=None)
            self.edge_gdf = gpd.GeoDataFrame(self.edge_data)
            edge_gdf = self.edge_gdf
            
            
        fig, axs = plt.subplots(1,2, figsize=(12, 6), subplot_kw={'projection':proj})
        # Handling title
        wrapped_title = "\n".join(textwrap.wrap(Title, 80))
        fig.suptitle(wrapped_title, fontsize=14, color='Black')
        
        # Plotting Left-hand map.
        axs[0].set_title("Zone Map with Numbered Nodes", fontsize=10)
        gplt.polyplot(Map, ax=axs[0], extent=extent, zorder=2, linewidth=0.3,
                      edgecolor='k')
        # Plotting nodal-zones
        gplt.voronoi(gdf, clip=Map.geometry, projection=proj,
                     extent=extent, edgecolor='lightgray', facecolor='white',
                     linewidth=0.2, ax=axs[0])
        
       
        # Label zones numerically
        xtran = gplt.crs.ccrs.Miller(central_longitude=0)
        for i in gdf.index:
            x=gdf.iloc[i]
            xy = xtran.transform_point(x.geometry.centroid.coords[0][0], x.geometry.centroid.coords[0][1], gplt.crs.ccrs.PlateCarree())
            axs[0].annotate(
                text=str(i),
                xy=xy,  # Use the transformed coordinates.
                fontsize=2,
                # fontstretch='ultra-expanded',
                ha='center',
                bbox={'facecolor': 'white', 'alpha':0.3, 'pad': 0.1, 'edgecolor':'none'},
                color='red')
    
           
        # Plotting rigth-hand map.
        axs[1].set_title("Link Map with Numbered Edges", fontsize=10)
        gplt.polyplot(Map, ax=axs[1], extent=extent, zorder=2, linewidth=0.3,
                      edgecolor='k')
        gplt.sankey(edge_gdf.geometry, projection=proj, zorder=3, ax=axs[1],
                    linewidth=0.4)
        
        for i in edge_gdf.index:
            x=edge_gdf.iloc[i]
            xy = xtran.transform_point(x.geometry.centroid.coords[0][0], x.geometry.centroid.coords[0][1], gplt.crs.ccrs.PlateCarree())
            axs[1].annotate(
                text=str(i),
                xy=xy,  # Use the transformed coordinates.
                fontsize=2,
                # fontstretch='ultra-expanded',
                ha='center',
                bbox={'facecolor': 'white', 'alpha':0.3, 'pad': 0.1, 'edgecolor':'none'},
                color='red')
        
        
        # fig.tight_layout()  # Adjust layout to ensure everything fits without overlap
        fig.savefig(self.path+subpath+figname+'.png', dpi=400)
        plt.close()
    
    def edge_dataPrep(self, capacity=None):
        linkdata = self.data_class.link_data()
        edge_data = []
        for pos in linkdata['positions']:
                edge_data.append({
                    'geometry': LineString([pos[0,:], pos[1,:]])})
        self.edge_data = edge_data
        self.edge_gdf = gpd.GeoDataFrame(edge_data)

  
        
    