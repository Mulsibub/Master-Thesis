import numpy as np
import copy
class used_kappa_class:
    def __init__(self,init,layout,flowtrace):
        self.init = init
        self.layout = layout
        self.flowtrace = flowtrace
        self.N, self.L = np.shape(flowtrace['trans'])
        
   
    
    def get_kappa_used(self,kappa_installed,avg_export,avg_generation,avg_curtailment):
        kappa_used = np.zeros((self.N,self.N))
        for n in range(self.N):
            for m in range(self.N):
                if m!=n:
                    kappa_used[m,n] = (avg_export[m,n] / (avg_generation[m]-avg_curtailment[m])) * kappa_installed[m]
        kappa_used = np.nan_to_num(kappa_used)
        np.fill_diagonal(kappa_used,(kappa_installed - kappa_used.sum(axis=1)))
        return kappa_used
    
  
    

    def run(self):
        
        wind_gen = self.layout['Generation']['wind'][0:self.init.t,:]
        solar_gen = self.layout['Generation']['solar'][0:self.init.t,:]
        backup_gen = self.layout['Generation']['backup'][0:self.init.t,:]
        curtailment = self.layout['Generation']['curtailment'][0:self.init.t,:]

        wind_kappa = self.layout['kappa']['wind']
        solar_kappa = self.layout['kappa']['solar']
        backup_kappa = self.layout['kappa']['backup']
        BE_kappa = self.layout['kappa']['backup_energy']
        
        avg_export_wind = self.flowtrace['wind'] / self.init.t
        avg_export_solar = self.flowtrace['solar'] / self.init.t
        avg_export_backup = self.flowtrace['backup'] / self.init.t
        
        avg_backup_gen = backup_gen.mean(axis=0)
        
        ren_gen = wind_gen + solar_gen
        
        wind_fraction = wind_gen / ren_gen
        wind_fraction[np.isnan(wind_fraction)] = 0
        
        solar_fraction = solar_gen / ren_gen
        solar_fraction[np.isnan(solar_fraction)] = 0
        
        Injection = self.layout['misc']['injection'][0:self.init.t,:]
        curtailment_local = copy.deepcopy(curtailment)
        curtailment_local[Injection<0] = 0

        avg_wind_curtailment_local = np.multiply(curtailment_local,wind_fraction).mean(axis=0)
        avg_solar_curtailment_local = np.multiply(curtailment_local,solar_fraction).mean(axis=0)
    
        kappa_used_wind = self.get_kappa_used(wind_kappa,avg_export_wind,wind_gen.mean(axis=0),avg_wind_curtailment_local)
        kappa_used_solar = self.get_kappa_used(solar_kappa,avg_export_solar,solar_gen.mean(axis=0),avg_solar_curtailment_local)
        kappa_used_backup = self.get_kappa_used(backup_kappa,avg_export_backup,avg_backup_gen,np.zeros(self.N))

     
        
        # Total_backup_Gen = (backup_gen*(self.init.t2/self.init.t)).sum(axis=0)
        BE_used = np.zeros(self.N)
        for n in range(self.N):
            BE_used[n] = (kappa_used_backup[:,n]*BE_kappa/backup_kappa).sum()

        # Transmission
        try:
            linklength = self.layout['kappa']['trans']['length']
        except IndexError:
            lDat = self.init.link_dataF()
            linklength = lDat['length']
        
        kappa_used_trans = np.multiply(self.flowtrace['trans'], linklength).T

        #Total kappa used is the sum of each row
        kappa = {
            'wind':kappa_used_wind.sum(axis=0),
            'solar':kappa_used_solar.sum(axis=0),
            'backup':kappa_used_backup.sum(axis=0),
            'backup_energy': BE_used,
            "trans": kappa_used_trans.sum(axis=0)
            }
        return kappa
