import numpy as np
# from post_processing.PlotFunctions import PostageSplot as PF



#%%
class LCOE2_class:
    def __init__(self, init, Prices, input_parameters, usedkappa, small_n=False):
        self.init = init
        self.Prices = Prices
        self.usedkappa = usedkappa
        self.small_n =small_n
        # faktor = 1000
        # Prices = {
        #     # Capital expenditure
        #     "Capextrans": 400,  # [EURO/(MW km)]
        #     "Capexwind": 1623 * faktor, # [EURO/(MW)]
        #     "Capexsolar": 762.5 * faktor, # [EURO/(MW)]
        #     "Capexbackup": 880 * faktor, # [EURO/(MW)]
        #     # Operational expenditure
        #     "OpExwind": 25.355 * faktor,  # [EURO/(MW year)]
        #     "OpExsolar": 17.24 * faktor, # [EURO/(MW year)]
        #     "OpExbackup": 29.04 * faktor, # [EURO/(MW year)]
        #     "OpExbackupVar": 20.1, # Variable expenditure per unit fuel [EURO/(MWh)]
        #     "OpExtrans": 8, # [EURO/(MW km year)]
        #     "etabackup": 0.59, # Thermal effeciency of fuel for backup [MW_out/MW_th]
        #     "MWtoKW": 1,  # Conversion from MW to KW
        #     # Lifetime [years]
        #     "LifeTimewind": 27,
        #     "LifeTimesolar": 30,
        #     "LifeTimebackup": 25,
        #     "LifeTimetrans": 40,
        #     # Rate of return
        #     "ror": 0.04
        # }  # Calls prices dict from function
        self.input_parameters = input_parameters
        
    def LCOEWrap(self, kappa, nodal_trans, scope= 'nodal'):
        # kappa = GD['kappa']
        # kappa['Trans'] = GD['time_series']
        #flowfrom grah ...
        Tech = ['wind', 'solar', 'backup','backup_energy', 'trans']
        # scope = 'country'
        # kappa = GD['kappa']
        # Tech = ['Wind', 'Solar', 'Backup', 'Backup_Energy']
       
        if self.small_n:
            LoadData = self.init.Load.sum(axis=0)
            BusData = self.init.BusData
            country_of_n = BusData['country'].to_numpy()  # Convert to NumPy array
            country_list = np.unique(country_of_n)
            country_int = []
            Load = []
            for country in country_list:
                start_index = np.where(country_of_n == country)[0][0]
                end_index = np.where(country_of_n == country)[0][-1]
                # Always append the tuple (start_index, end_index), even if they are the same
                Load.append((LoadData[start_index:end_index+1]).sum())
            LoadData=np.array(Load)
        else:
            LoadData = self.init.Load.sum(axis=0)
            BusData = self.init.BusData
            
       
        kappa['trans'] = nodal_trans
        
        # Making sure the Data formatted properly
        # LoadData = (LoadData.to_numpy()).sum(axis=0) # Total nodal Load
        
        Capacities = {}        
        if scope == 'country':
            country_of_n = BusData['country'].to_numpy()  # Convert to NumPy array
            country_list = np.unique(country_of_n)
            country_int = []
            Load = []
            for country in country_list:
                start_index = np.where(country_of_n == country)[0][0]
                end_index = np.where(country_of_n == country)[0][-1]
                # Always append the tuple (start_index, end_index), even if they are the same
                temp = (start_index, end_index + 1)
                country_int.append(temp)  # Add +1 to make inclusive slicing easier
                Load.append((LoadData[start_index:end_index+1]).sum())
            for tech, var in kappa.items():
                values = []
                for i, (start, end) in enumerate(country_int):
                    values.append(var[start:end].sum())
                Capacities[tech] = values
    
        elif scope == 'total':
            Load = LoadData.sum()
            for tech in Tech:
                Capacities[tech] = kappa[tech].sum()
                
        elif scope == 'nodal':
            Load = LoadData
            Capacities = kappa        
        else:
            print("Error: No such LCOE scope. scopes must be either 'nodal'(default), 'country', or 'total'.")
        return self.LCOECalc(Capacities, Load)
    
    
    def LCOECalc(self, Capacities, Load):
        Prices = self.Prices
        Tech = ['wind', 'solar', 'backup','backup_energy', 'trans']
        # Tech = ['Wind', 'Solar', 'backup']
        r = Prices["ror"]
        LCOE = {}
        
        # Ensure Load is a numpy array (to handle both scalars and arrays)
        Load = np.array(Load, dtype=float)
        # Types= {'imp': imp, 'exp': exp, 'ins':ins}
        # for key, kappa in Types.items():
        #     CapEx = Prices["Capex" + tech]
        #     OpEx = Prices["OpEx" + tech]
        #     LifeTime = Prices["LifeTime" + tech]
        #     V = np.add(np.dot(CapEx, kappa), sum((np.dot(OpEx, kappa)) / ((1 + r) ** t) for t in range(1, LifeTime + 1)))
        #     # Calculate LCOE: handle array or scalar Load
        #     discounted_load = sum(np.divide(Load, (1 + r) ** (t + 1)) for t in range(LifeTime))
        #     LCOE[key] = np.divide(V, discounted_load)
        for tech in Tech:
            kappa = np.array(Capacities[tech], dtype=float)
            if tech == 'backup_energy':
                V = np.dot(kappa, Prices["OpExbackupVar"]/Prices["etabackup"])
                LCOE[tech] = np.divide(V,Load)
                # V_varEx = np.dot(np.array(Capacities[tech], dtype=float), Prices["OpExbackupVar"])
                # kappa = np.zeros(np.shape(Capacities['backup']))
            else:
                CapEx = Prices["Capex" + tech]
                OpEx = Prices["OpEx" + tech]
                LifeTime = Prices["LifeTime" + tech]
                V = np.add(np.dot(CapEx, kappa), sum((np.dot(OpEx, kappa)) / ((1 + r) ** t) for t in range(1, LifeTime + 1)))
                # Calculate LCOE: handle array or scalar Load
                discounted_load = sum(np.divide(Load, (1 + r) ** (t + 1)) for t in range(LifeTime))
                LCOE[tech] = np.divide(V, discounted_load)
                
            
        Res = {}
        summer = np.zeros(np.shape(LCOE['wind']))
        for key1, var in LCOE.items():
            key2 = 'LCOE_'+key1
            Res[key2] = var
            summer += var
        Res['LCOE'] = summer
            
        return Res
    
    # def Run(self, LL, Trans_Method="Load"):
    #     # Trans_Method can be 'Half' or 'Load' for the 
    #     kappa = LL['kappa'] # Capacities

    #     if any([Trans_Method =="PS_Opt", Trans_Method =="PS_0", Trans_Method =="PS_1", Trans_Method =="PS_0.5"]):
    #         nodal_trans = LL['misc']['nodal_trans_Load']
    #     elif any([Trans_Method =="FT_ex", Trans_Method =="FT_im"]):
    #         if self.init.Balancing_scheme == 'NoT':
    #             nodal_trans = LL['misc']['nodal_trans_'+Trans_Method]  
    #         else:
    #             if Trans_Method =="FT_ex":
    #                 nodal_trans = self.usedkappa['export']['trans']
    #             elif Trans_Method =="FT_im":
    #                 nodal_trans = self.usedkappa['import']['trans']
    #     else:
    #         nodal_trans = LL['misc']['nodal_trans_'+Trans_Method]  

    #     LCOE_EU = self.LCOEWrap(kappa, nodal_trans, scope= 'total')
    #     LCOE_C = self.LCOEWrap(kappa, nodal_trans, scope= 'country')
    #     LCOE_n = self.LCOEWrap(kappa, nodal_trans, scope= 'nodal')
    #     if Trans_Method == "PS_Opt":
    #         res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, path='plots/PS/', plot = False )
    #         LCOE_C = res['LCOE_c'] ## OBS: NOTE That res['LCOE_c'] and res['LCOE_n'] can be very different and is not the same!! ###
    #         # optimal_delta_c = res['optimal_delta_c']
    #         LCOE_n = res['LCOE_n'] ## OBS: NOTE That res['LCOE_c'] and res['LCOE_n'] can be very different and is not the same!! ###
    #         # optimal_delta_n = res['optimal_delta_n']
    #     #Postage stamp method:
    #     elif Trans_Method == "PS_0":
    #         res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, delta=0 )
    #         LCOE_C = res['LCOE_c'] 
    #         LCOE_n = res['LCOE_n']
    #     elif Trans_Method == "PS_1":
    #         res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, delta=1)
    #         LCOE_C = res['LCOE_c'] 
    #         LCOE_n = res['LCOE_n']
    #     elif Trans_Method == "PS_0.5":
    #         res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, delta=0.5)
    #         LCOE_C = res['LCOE_c'] 
    #         LCOE_n = res['LCOE_n']
       
    #     return {'LCOE_EU': LCOE_EU, 'LCOE_C': LCOE_C, 'LCOE_n': LCOE_n}
    
    def Run(self, LL, Trans_Method="Load", tilde =False):
        self.small_n =False
        self.tilde = tilde
        # Trans_Method can be 'Half' or 'Load' for the 
        if tilde:
            kappa = self.usedkappa['import'] if self.init.Balancing_scheme != 'NoT' else LL['kappa']
        else:
            kappa = LL['kappa']

        if any([Trans_Method =="PS_Opt", Trans_Method =="PS_0", Trans_Method =="PS_1", Trans_Method =="PS_0.5"]):
            nodal_trans = LL['misc']['nodal_trans_Load']
        elif Trans_Method == "FT_im":
            nodal_trans = self.usedkappa['import']['trans']  if self.init.Balancing_scheme != 'NoT' else LL['kappa']['trans']
        elif Trans_Method == "FT_ex":
            nodal_trans = self.usedkappa['export']['trans'] if self.init.Balancing_scheme != 'NoT' else LL['misc']['trans']
        elif Trans_Method == "Load":
            nodal_trans = LL['misc']['nodal_trans_Load']
        elif Trans_Method == "Half":
            nodal_trans = LL['misc']['nodal_trans_Half']
        else:
            print('error! No such transmethod')
            return None
            
        LCOE_EU = self.LCOEWrap(kappa, nodal_trans, scope= 'total')
        LCOE_C = self.LCOEWrap(kappa, nodal_trans, scope= 'country')
        LCOE_n = self.LCOEWrap(kappa, nodal_trans, scope= 'nodal')
        if Trans_Method == "PS_Opt":
            res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, path='plots/PS/', plot = False )
            LCOE_C = res['LCOE_c'] ## OBS: NOTE That res['LCOE_c'] and res['LCOE_n'] can be very different and is not the same!! ###
            # optimal_delta_c = res['optimal_delta_c']
            LCOE_n = res['LCOE_n'] ## OBS: NOTE That res['LCOE_c'] and res['LCOE_n'] can be very different and is not the same!! ###
            # optimal_delta_n = res['optimal_delta_n']
        #Postage stamp method:
        elif Trans_Method == "PS_0":
            res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, delta=0 )
            LCOE_C = res['LCOE_c'] 
            LCOE_n = res['LCOE_n']
        elif Trans_Method == "PS_1":
            res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, delta=1)
            LCOE_C = res['LCOE_c'] 
            LCOE_n = res['LCOE_n']
        elif Trans_Method == "PS_0.5":
            res = self.PostageStamp(LL['misc'], LCOE_C, LCOE_n, delta=0.5)
            LCOE_C = res['LCOE_c'] 
            LCOE_n = res['LCOE_n']
       
        return {'LCOE_EU': LCOE_EU, 'LCOE_C': LCOE_C, 'LCOE_n': LCOE_n}
    
    def Run_n(self, LL, Trans_Method="Load", tilde =False):
        self.small_n =True
        self.tilde = tilde
        self.layout=LL
        # Trans_Method can be 'Half' or 'Load' for the 
        if tilde:
            try:
                kappa = self.usedkappa['import']
            except TypeError:
                kappa = LL['kappa'] # Capacities
        else:
            kappa = LL['kappa']

        if any([Trans_Method =="PS_Opt", Trans_Method =="PS_0", Trans_Method =="PS_1", Trans_Method =="PS_0.5"]):
            nodal_trans = LL['misc']['country_trans_Load']
        elif any([Trans_Method =="FT_ex", Trans_Method =="FT_im"]):
            if self.init.Balancing_scheme == 'NoT':
                nodal_trans = LL['misc']['country_trans_'+Trans_Method]  
            else:
                if Trans_Method =="FT_ex":
                    nodal_trans = self.usedkappa['export']['trans']
                elif Trans_Method =="FT_im":
                    nodal_trans = self.usedkappa['import']['trans']
                
        else:
            nodal_trans = LL['misc']['country_trans_'+Trans_Method]  

        LCOE_EU = self.LCOEWrap(kappa, nodal_trans, scope= 'total')
        LCOE_n = self.LCOEWrap(kappa, nodal_trans, scope= 'nodal')
        if Trans_Method == "PS_Opt":
            res = self.PostageStamp_n(LL['misc'], LCOE_n, path='plots/PS/', plot = False )
            # optimal_delta_c = res['optimal_delta_c']
            LCOE_n = res['LCOE_n'] ## OBS: NOTE That res['LCOE_c'] and res['LCOE_n'] can be very different and is not the same!! ###
            # optimal_delta_n = res['optimal_delta_n']
        #Postage stamp method:
        elif Trans_Method == "PS_0":
            res = self.PostageStamp_n(LL['misc'], LCOE_n, delta=0 )
            LCOE_n = res['LCOE_n']
        elif Trans_Method == "PS_1":
            res = self.PostageStamp_n(LL['misc'], LCOE_n, delta=1) 
            LCOE_n = res['LCOE_n']
        elif Trans_Method == "PS_0.5":
            res = self.PostageStamp_n(LL['misc'], LCOE_n, delta=0.5)
            LCOE_n = res['LCOE_n']
       
        return {'LCOE_EU': LCOE_EU, 'LCOE_C': LCOE_n}

    def PostageStamp(self, data, LCOE_C, LCOE_N, delta =None, plot = True, figname = '', path=''):
        Prices = self.Prices  # Calls prices dict from function
        r = Prices["ror"]
        tech = 'trans'
        CapEx = Prices["Capex" + tech]
        OpEx = Prices["OpEx" + tech]
        LifeTime = Prices["LifeTime" + tech]

        Load = self.init.Load.sum(axis=0)
        countries = self.init.country_list
        country_int = self.init.country_int
        p = data['injection']
        M_n = 'nodal_trans_Load'
        # M_C = 'country_trans_Load'
        kappa_n = data[M_n]
      
     
        # Calculate V (present value of CapEx and OpEx)
        V = np.add(np.dot(CapEx, kappa_n), sum((np.dot(OpEx, kappa_n)) / ((1 + r) ** t) for t in range(1, LifeTime + 1)))
        V_tot = V.sum()
        
        # Sender side
        p_plus = np.where(p > 0, p, 0)
        p_p_avg = p_plus.mean(axis=0)
        a_plus = np.divide(p_p_avg, p_p_avg.sum()) # Allocation coeff
        V_n_plus = np.multiply(a_plus, V_tot)
        
        # Reciever side
        p_minus = -np.where(p < 0, p, 0)
        p_m_avg = p_minus.mean(axis=0)
        a_minus = np.divide(p_m_avg, p_m_avg.sum())  # Allocation coeff
        V_n_minus = np.multiply(a_minus, V_tot)
    
        # TotLoad_n = LoadData.sum(axis=0)
        # TotalCost = LCOE_trans(GD['PowerFlow'], BusPositions, LoadData, GD['kappa']['trans']) * TotLoad_n.sum() # LCOE * System Load
        
        # Mixing nodal
        if delta == None:
            delta = np.linspace(0, 1, 101) # Mixing parameter
            opt=True
        else:
            plot = False
            opt = False
        
        V_n_T = np.outer(delta, V_n_plus) + np.outer(1-delta, V_n_minus)
        discounted_load = sum(np.divide(Load,( (1 + r) ** (t + 1))) for t in range(LifeTime))
        LCOE_n_T = np.divide(V_n_T, discounted_load)
        
        
        # Mixing country
        Load_C = np.zeros(len(countries))
        V_c_plus = np.zeros(len(countries))
        V_c_minus = np.zeros(len(countries))
        for i, (start, end) in enumerate(country_int):
            Load_C[i] = Load[start:end].sum()
            V_c_plus[i] = V_n_plus[start:end].sum()
            V_c_minus[i] = V_n_minus[start:end].sum()
        V_c_T = np.outer(delta, V_c_plus) + np.outer(1-delta, V_c_minus)
        discounted_load = sum(np.divide(Load_C,( (1 + r) ** (t + 1))) for t in range(LifeTime))
        LCOE_c_T = np.divide(V_c_T, discounted_load)
        
        #Summation of LCOE
        otherLCOE_n = np.subtract(LCOE_N['LCOE'],LCOE_N['LCOE_trans'])
        LCOE_n = np.add(LCOE_n_T, otherLCOE_n)
        otherLCOE_c = np.subtract(LCOE_C['LCOE'],LCOE_C['LCOE_trans'])
        LCOE_c = LCOE_c_T + otherLCOE_c 
        
        if opt:
            # Calculate variance across the rows (variance over 'a' for each delta)
            LCOE_n_var = np.var(LCOE_n, axis=1)
            LCOE_c_var = np.var(LCOE_c, axis=1)
            
            # Find the index of the minimum variance for nodes
            min_var_index_n = np.argmin(LCOE_n_var)
            optimal_delta_n = delta[min_var_index_n]
            min_var_n = LCOE_n_var[min_var_index_n]
            
            # Find the index of the minimum variance for countries
            min_var_index_c = np.argmin(LCOE_c_var)
            optimal_delta_c = delta[min_var_index_c]
            min_var_c = LCOE_c_var[min_var_index_c]
    
            if plot:
                param = self.input_parameters 
                name = f"{param['Layout_Scheme']}_{param['Balancing_Scheme']}_{param['beta']}"
                   # PostageSplot(optimal_delta_n, optimal_delta_c, min_var_n, min_var_c, LCOE_n_var, LCOE_c_var, LCOE_n, LCOE_c, name, delta, path, figname, Method):
                # PF(optimal_delta_n, optimal_delta_c, min_var_n, min_var_c, LCOE_n_var, LCOE_c_var, LCOE_n, LCOE_c, name, delta, path=path, figname=name)
            LCOE_C['LCOE'] = LCOE_c[min_var_index_c,:]
            LCOE_C['LCOE_trans'] = LCOE_c_T[min_var_index_c,:]
            LCOE_N['LCOE'] = LCOE_n[min_var_index_n,:]
            LCOE_N['LCOE_trans'] = LCOE_n_T[min_var_index_n,:]
            res = {'optimal_delta_c' : optimal_delta_c,
                   'optimal_delta_n' : optimal_delta_n,
                   'Minvar_c': min_var_c, 'Minvar_n': min_var_n,
                   'LCOE_c': LCOE_C, 'LCOE_n': LCOE_N }
        else:
            LCOE_C['LCOE'] = LCOE_c
            LCOE_C['LCOE_trans'] = LCOE_c_T
            LCOE_N['LCOE'] = LCOE_n
            LCOE_N['LCOE_trans'] = LCOE_n_T
            res = {'LCOE_c': LCOE_C, 'LCOE_n': LCOE_N }
    
        return res
    
    def PostageStamp_n(self, data, LCOE_N, delta =None, plot = True, figname = '', path=''):
        Prices = self.Prices  # Calls prices dict from function
        r = Prices["ror"]
        tech = 'trans'
        CapEx = Prices["Capex" + tech]
        OpEx = Prices["OpEx" + tech]
        LifeTime = Prices["LifeTime" + tech]

        countries = self.init.country_list
        Load = self.layout['Load_C'].sum(axis=0)
        country_int = self.init.country_int
        p = data['injection']
        M_n = 'country_trans_Load'
        # M_C = 'country_trans_Load'
        kappa_n = data[M_n]
      
     
        # Calculate V (present value of CapEx and OpEx)
        V = np.add(np.dot(CapEx, kappa_n), sum((np.dot(OpEx, kappa_n)) / ((1 + r) ** t) for t in range(1, LifeTime + 1)))
        V_tot = V.sum()
        
        # Sender side
        p_plus = np.where(p > 0, p, 0)
        p_p_avg = p_plus.mean(axis=0)
        a_plus = np.divide(p_p_avg, p_p_avg.sum()) # Allocation coeff
        V_n_plus = np.multiply(a_plus, V_tot)
        
        # Reciever side
        p_minus = -np.where(p < 0, p, 0)
        p_m_avg = p_minus.mean(axis=0)
        a_minus = np.divide(p_m_avg, p_m_avg.sum())  # Allocation coeff
        V_n_minus = np.multiply(a_minus, V_tot)
    
        # TotLoad_n = LoadData.sum(axis=0)
        # TotalCost = LCOE_trans(GD['PowerFlow'], BusPositions, LoadData, GD['kappa']['trans']) * TotLoad_n.sum() # LCOE * System Load
        
        # Mixing nodal
        if delta == None:
            delta = np.linspace(0, 1, 101) # Mixing parameter
            opt=True
        else:
            plot = False
            opt = False
        
        V_n_T = np.outer(delta, V_n_plus) + np.outer(1-delta, V_n_minus)
        discounted_load = sum(np.divide(Load,( (1 + r) ** (t + 1))) for t in range(LifeTime))
        LCOE_n_T = np.divide(V_n_T, discounted_load)
        
        
        #Summation of LCOE
        otherLCOE_n = np.subtract(LCOE_N['LCOE'],LCOE_N['LCOE_trans'])
        LCOE_n = np.add(LCOE_n_T, otherLCOE_n)

        
        if opt:
            # Calculate variance across the rows (variance over 'a' for each delta)
            LCOE_n_var = np.var(LCOE_n, axis=1)
            
            # Find the index of the minimum variance for nodes
            min_var_index_n = np.argmin(LCOE_n_var)
            optimal_delta_n = delta[min_var_index_n]
            min_var_n = LCOE_n_var[min_var_index_n]
            
          
    
            if plot:
                param = self.input_parameters 
                name = f"{param['Layout_Scheme']}_{param['Balancing_Scheme']}_{param['beta']}"
                   # PostageSplot(optimal_delta_n, optimal_delta_c, min_var_n, min_var_c, LCOE_n_var, LCOE_c_var, LCOE_n, LCOE_c, name, delta, path, figname, Method):
                # PF(optimal_delta_n, optimal_delta_c, min_var_n, min_var_c, LCOE_n_var, LCOE_c_var, LCOE_n, LCOE_c, name, delta, path=path, figname=name)
           
            LCOE_N['LCOE'] = LCOE_n[min_var_index_n,:]
            LCOE_N['LCOE_trans'] = LCOE_n_T[min_var_index_n,:]
            res = {'optimal_delta_n' : optimal_delta_n,
                   'Minvar_n': min_var_n,
                   'LCOE_n': LCOE_N }
        else:
            
            LCOE_N['LCOE'] = LCOE_n
            LCOE_N['LCOE_trans'] = LCOE_n_T
            res = {'LCOE_n': LCOE_N }
    
        return res
