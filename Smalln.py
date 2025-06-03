# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:32:21 2025

@author: heide
"""
#%%
from initialize.data_loader import data_loader
from initialize.data_class import data_class
from layout.layout_class import layout_class
from flowtracing.flowtracing_class import flowtracing_class
from flowtracing.used_kappa_class import used_kappa_class
from smallnFunc import Aggregator
from post_processing.LCOE3_class import LCOE2_class
from post_processing import PlotFunctions as PF

import pickle
import os
import copy
#%%
def initialize(data, input_parameters):
    data_processor = data_class(data, input_parameters)
    data_processor.run()
    return data_processor

def execute_layout(data_processor):
    layout_manager = layout_class(data_processor)
    return layout_manager.run()

def execute_flow_tracing(data_processor, layout, path='FTPlots/', smalln=False):
    print('export')
    flow_tracer_export = flowtracing_class(data_processor, layout, Import=False, path=path, smalln=smalln)
    export_FT = flow_tracer_export.run()
    print('import')
    flow_tracer_import = flowtracing_class(data_processor, layout, Import=True, path=path, smalln=smalln)
    import_FT = flow_tracer_import.run()
    return {"export": export_FT, "import": import_FT}


def calculate_utilized_capacity(data_processor, layout, flow_tracing):
    used_kappa = {}
    for imex in ["import", 'export']:
        FT = flow_tracing[imex]
        kappa_utilization = used_kappa_class(data_processor,layout, FT)
        used_kappa[imex] = kappa_utilization.run()
    return used_kappa

def load_results_pickle(save_path="pipeline_results.pkl"):
    """
    Reads a layout results from the pickle file one-by-one to avoid high memory usage.
    """
    results = {}
    with open(save_path, 'rb') as f:
        while True:
            try:
                result = pickle.load(f)  # Load each object separately
                key = f"{result['Layout_Scheme']}_{result['Balancing_Scheme']}_{result['beta']}"
                results[key] = result  # Store in dictionary
            except EOFError:  # Stop when reaching end of file
                break

    return results
def execute_LCOE(data_processor, layout, input_parameters, usedkappa = None):
    faktor = 1000
    Prices = {
        # Capital expenditure
        "Capextrans": 400,  # [EURO/(MW km)]
        "Capexwind": 1623 * faktor, # [EURO/(MW)]
        "Capexsolar": 762.5 * faktor, # [EURO/(MW)]
        "Capexbackup": 880 * faktor, # [EURO/(MW)]
        # Operational expenditure
        "OpExwind": 25.355 * faktor,  # [EURO/(MW year)]
        "OpExsolar": 17.24 * faktor, # [EURO/(MW year)]
        "OpExbackup": 29.04 * faktor, # [EURO/(MW year)]
        "OpExbackupVar": 20.1, # Variable expenditure per unit fuel [EURO/(MWh)]
        "OpExtrans": 8, # [EURO/(MW km year)]
        "etabackup": 0.59, # Thermal effeciency of fuel for backup [MW_out/MW_th]
        "MWtoKW": 1,  # Conversion from MW to KW
        # Lifetime [years]
        "LifeTimewind": 27,
        "LifeTimesolar": 30,
        "LifeTimebackup": 25,
        "LifeTimetrans": 40,
        # Rate of return
        "ror": 0.04
    }  # Calayouts prices dict from function
    Transmethods = ["Load", "Half", "PS_Opt", "PS_0", "PS_1", "PS_0.5"]
    LCOE_Calc = LCOE2_class(data_processor, Prices, input_parameters, usedkappa)

    LCOE_installed = {'LCOE_C': {}, 'LCOE_n': {}}
    LCOE_used = {'LCOE_C': {}, 'LCOE_n': {}}

    for m in Transmethods:
        temp_installed = LCOE_Calc.Run(layout, m, tilde=False)
        LCOE_installed['LCOE_C'][m] = temp_installed['LCOE_C']
        LCOE_installed['LCOE_n'][m] = temp_installed['LCOE_n']
        if m == "Half":
            LCOE_installed['LCOE_EU'] = temp_installed['LCOE_EU']

        temp_used = LCOE_Calc.Run(layout, m, tilde=True)
        LCOE_used['LCOE_C'][m] = temp_used['LCOE_C']
        LCOE_used['LCOE_n'][m] = temp_used['LCOE_n']
        if m == "Half":
            LCOE_used['LCOE_EU'] = temp_used['LCOE_EU']

    if usedkappa is not None:
        # Installed
        temp_ex = LCOE_Calc.Run(layout, 'FT_ex')
        temp_im = LCOE_Calc.Run(layout, 'FT_im')
        LCOE_installed['LCOE_C']['FT_ex'] = temp_ex['LCOE_C']
        LCOE_installed['LCOE_n']['FT_ex'] = temp_ex['LCOE_n']
        LCOE_installed['LCOE_C']['FT_im'] = temp_im['LCOE_C']
        LCOE_installed['LCOE_n']['FT_im'] = temp_im['LCOE_n']
        LCOE_installed['LCOE_C']['FT_0.5'] = copy.deepcopy(temp_ex['LCOE_C'])
        LCOE_installed['LCOE_n']['FT_0.5'] = copy.deepcopy(temp_ex['LCOE_n'])

        for key in temp_ex['LCOE_C']:
            LCOE_installed['LCOE_C']['FT_0.5'][key] = (
                temp_ex['LCOE_C'][key] + temp_im['LCOE_C'][key]) / 2
            LCOE_installed['LCOE_n']['FT_0.5'][key] = (
                temp_ex['LCOE_n'][key] + temp_im['LCOE_n'][key]) / 2

        # Used
        temp_ex_used = LCOE_Calc.Run(layout, 'FT_ex', tilde=True)
        temp_im_used = LCOE_Calc.Run(layout, 'FT_im', tilde=True)
        LCOE_used['LCOE_C']['FT_ex'] = temp_ex_used['LCOE_C']
        LCOE_used['LCOE_n']['FT_ex'] = temp_ex_used['LCOE_n']
        LCOE_used['LCOE_C']['FT_im'] = temp_im_used['LCOE_C']
        LCOE_used['LCOE_n']['FT_im'] = temp_im_used['LCOE_n']
        LCOE_used['LCOE_C']['FT_0.5'] = copy.deepcopy(temp_ex_used['LCOE_C'])
        LCOE_used['LCOE_n']['FT_0.5'] = copy.deepcopy(temp_ex_used['LCOE_n'])

        for key in temp_ex_used['LCOE_C']:
            LCOE_used['LCOE_C']['FT_0.5'][key] = (
                temp_ex_used['LCOE_C'][key] + temp_im_used['LCOE_C'][key]) / 2
            LCOE_used['LCOE_n']['FT_0.5'][key] = (
                temp_ex_used['LCOE_n'][key] + temp_im_used['LCOE_n'][key]) / 2

    return {'installed': LCOE_installed, 'used': LCOE_used}


def execute_LCOE_n(data_processor, layout, input_parameters, usedkappa = None):
    faktor = 1000
    Prices = {
        # Capital expenditure
        "Capextrans": 400,  # [EURO/(MW km)]
        "Capexwind": 1623 * faktor, # [EURO/(MW)]
        "Capexsolar": 762.5 * faktor, # [EURO/(MW)]
        "Capexbackup": 880 * faktor, # [EURO/(MW)]
        # Operational expenditure
        "OpExwind": 25.355 * faktor,  # [EURO/(MW year)]
        "OpExsolar": 17.24 * faktor, # [EURO/(MW year)]
        "OpExbackup": 29.04 * faktor, # [EURO/(MW year)]
        "OpExbackupVar": 20.1, # Variable expenditure per unit fuel [EURO/(MWh)]
        "OpExtrans": 8, # [EURO/(MW km year)]
        "etabackup": 0.59, # Thermal effeciency of fuel for backup [MW_out/MW_th]
        "MWtoKW": 1,  # Conversion from MW to KW
        # Lifetime [years]
        "LifeTimewind": 27,
        "LifeTimesolar": 30,
        "LifeTimebackup": 25,
        "LifeTimetrans": 40,
        # Rate of return
        "ror": 0.04
    }  # Calayouts prices dict from function
    Transmethods = ["Load", "Half", "PS_Opt", "PS_0", "PS_1", "PS_0.5"]
    LCOE_Calc = LCOE2_class(data_processor, Prices, input_parameters, usedkappa)

    LCOE_installed = {'LCOE_C':{}}
    LCOE_used = {'LCOE_C':{}}

    for m in Transmethods:
        temp_installed = LCOE_Calc.Run_n(layout, m, tilde=False)
        LCOE_installed['LCOE_C'][m] = temp_installed['LCOE_C']
        if m == "Half":
            LCOE_installed['LCOE_EU'] = temp_installed['LCOE_EU']

        temp_used = LCOE_Calc.Run_n(layout, m, tilde=True)
        LCOE_used['LCOE_C'][m] = temp_used['LCOE_C']
        
        if m == "Half":
            LCOE_used['LCOE_EU'] = temp_used['LCOE_EU']

    if usedkappa is not None:
        # Installed
        temp_ex = LCOE_Calc.Run_n(layout, 'FT_ex')
        temp_im = LCOE_Calc.Run_n(layout, 'FT_im')
        LCOE_installed['LCOE_C']['FT_ex'] = temp_ex['LCOE_C']
        LCOE_installed['LCOE_C']['FT_im'] = temp_im['LCOE_C']
        LCOE_installed['LCOE_C']['FT_0.5'] = copy.deepcopy(temp_ex['LCOE_C'])
        for key in temp_ex['LCOE_C']:
            LCOE_installed['LCOE_C']['FT_0.5'][key] = (temp_ex['LCOE_C'][key] + temp_im['LCOE_C'][key]) / 2

        # Used
        temp_ex_used = LCOE_Calc.Run_n(layout, 'FT_ex', tilde=True)
        temp_im_used = LCOE_Calc.Run_n(layout, 'FT_im', tilde=True)
        LCOE_used['LCOE_C']['FT_ex'] = temp_ex_used['LCOE_C']
        LCOE_used['LCOE_C']['FT_im'] = temp_im_used['LCOE_C']
        LCOE_used['LCOE_C']['FT_0.5'] = copy.deepcopy(temp_ex['LCOE_C'])
        for key in temp_ex_used['LCOE_C']:
            LCOE_used['LCOE_C']['FT_0.5'][key] = (temp_ex['LCOE_C'][key] + temp_im_used['LCOE_C'][key]) / 2
            
        

    return {'installed': LCOE_installed, 'used': LCOE_used}


def Refileing(pipeline_results, data, save_path, plot_path):
    append_mode = os.path.exists(save_path)  # Check if file exists
    NewRes = {}
    with open(save_path, 'ab') as f:  # 'ab' = append in binary mode
        for key, item in pipeline_results.items():
            layout_scheme = item['Layout_Scheme']
            balancing_scheme = item['Balancing_Scheme']
            beta = 1.0         
            input_parameters = {
                 'Layout_Scheme': layout_scheme,
                 'Balancing_Scheme': balancing_scheme,
                 'gamma': 1,  # Fixed for now
                 'beta': beta,
                 'alpha_opt_parameter': 'min_backup_energy',
                 'choice_of_critical_parameter': 'Sand_use_mg.pr.kWh',
                 'n_alpha': 11,
                 'n_t': 8760
             }
            name = f"{layout_scheme}_{balancing_scheme}_{beta}"
            print(name)
            # res = pipeline_results[name]
            
            data_processor = initialize(data, input_parameters)
            layout = execute_layout(data_processor)
            Agg = Aggregator(data_processor, Plot_path=plot_path)
            # # Agg.MapOverview()
            C_layout = Agg.Run(layout)
            # Agg.SingleMapHandler(C_layout, name, subpath='kappamap/')
            # Agg.SingleMapHandler(C_layout, name, Normalize=True, subpath='kappamapnorm/')
            # C_layout=res['Result']

            if balancing_scheme != 'NoT' and not (layout_scheme == 'Local' and balancing_scheme == 'Local'):
                # flow_tracing = res['flow_tracing']
                flow_tracing = execute_flow_tracing(data_processor, C_layout, path=plot_path+'FTPlots/', smalln=True)
                used_kappa = calculate_utilized_capacity(data_processor, C_layout, flow_tracing)
                imp = used_kappa['import']['trans']
                faktor = 1000
                Prices = {
                    # Capital expenditure
                    "Capextrans": 400,  # [EURO/(MW km)]
                    "Capexwind": 1623 * faktor, # [EURO/(MW)]
                    "Capexsolar": 762.5 * faktor, # [EURO/(MW)]
                    "Capexbackup": 880 * faktor, # [EURO/(MW)]
                    # Operational expenditure
                    "OpExwind": 25.355 * faktor,  # [EURO/(MW year)]
                    "OpExsolar": 17.24 * faktor, # [EURO/(MW year)]
                    "OpExbackup": 29.04 * faktor, # [EURO/(MW year)]
                    "OpExbackupVar": 20.1, # Variable expenditure per unit fuel [EURO/(MWh)]
                    "OpExtrans": 8, # [EURO/(MW km year)]
                    "etabackup": 0.59, # Thermal effeciency of fuel for backup [MW_out/MW_th]
                    "MWtoKW": 1,  # Conversion from MW to KW
                    # Lifetime [years]
                    "LifeTimewind": 27,
                    "LifeTimesolar": 30,
                    "LifeTimebackup": 25,
                    "LifeTimetrans": 40,
                    # Rate of return
                    "ror": 0.04
                }  # Calayouts prices dict from function
                LCOE = execute_LCOE_n(data_processor, C_layout, input_parameters, usedkappa = used_kappa)
                #quickfix:
                LCOE_class=LCOE2_class(data_processor, Prices, input_parameters, used_kappa,small_n=True)
                LCOE['installed']['LCOE_C']['FT_im'] = LCOE_class.LCOEWrap(C_layout['kappa'], imp, scope= 'nodal')
                LCOE['used']['LCOE_C']['FT_im'] = LCOE_class.LCOEWrap(used_kappa['import'], imp, scope= 'nodal')
                for key,_ in LCOE['used']['LCOE_C']['FT_0.5'].items():
                    LCOE['used']['LCOE_C']['FT_0.5'][key] = (
                        LCOE['used']['LCOE_C']['FT_ex'][key] + LCOE['used']['LCOE_C']['FT_im'][key]) / 2
                    LCOE['installed']['LCOE_C']['FT_0.5'][key] = (
                        LCOE['installed']['LCOE_C']['FT_ex'][key] + LCOE['installed']['LCOE_C']['FT_im'][key]) / 2
                    
                result = {'Layout_Scheme': layout_scheme,
                          'Balancing_Scheme': balancing_scheme,
                          'beta': beta,
                          'Result': C_layout,
                          'flow_tracing': flow_tracing,
                          'used_kappa': used_kappa,
                          'LCOE': LCOE}
            else:
                used_kappa =None
                LCOE = execute_LCOE_n(data_processor, C_layout, input_parameters, usedkappa = used_kappa, )
                result = {'Layout_Scheme': layout_scheme,
                          'Balancing_Scheme': balancing_scheme,
                          'beta': beta,
                          'Result': C_layout,
                          'LCOE': LCOE}
            Agg.InitializeMapper(plot_path=plot_path)
            Agg.MapOverview()
            C_layout = Agg.Run(layout)
            Agg.SingleMapHandler(C_layout, name, subpath='kappamap/')
            Agg.SingleMapHandler(C_layout, name, Normalize=True, subpath='kappamapnorm/')    
            PF.Repackage_forCspecialsStacks2(result, data_processor, path=plot_path+'Fig7/')
            NewRes[name] = result
            pickle.dump(result, f)
    return NewRes

#%%
plot_path = 'ResPlotsFAS/'
save_path = "pipeline_resultsFAS.pkl"
data = data_loader()   
pipeline_resultsFAB = load_results_pickle(save_path="pipeline_resultsFAB.pkl")
pipeline_resultsFAS = load_results_pickle(save_path="pipeline_resultsFAS.pkl")
# pipeline_resultsFAS = Refileing(pipeline_results, data, save_path, plot_path)
PF.Repackage_CompareLCOE(pipeline_resultsFAS, data, path='ResPlotsFAS/'+'Fig6/',Small_n =True, Type='used')
PF.Repackage_CompareLCOE(pipeline_resultsFAS, data, path='ResPlotsFAS/'+'Fig6/',Small_n =True, Type='installed')
PF.Repackage_CompareLCOE(pipeline_resultsFAB, data, path='ResPlotsFAB/'+'Fig6/', Type='installed')
PF.Repackage_CompareLCOE(pipeline_resultsFAB, data, path='ResPlotsFAB/'+'Fig6/', Type='used')

input_parameters = {
    'Layout_Scheme': 'Global',
    'Balancing_Scheme': 'Global',
    'gamma': 1,  # Fixed for now
    'beta': 1.0,
    'alpha_opt_parameter': 'min_backup_energy',
    'choice_of_critical_parameter': 'Sand_use_mg.pr.kWh',
    'n_alpha': 11,
    'n_t': 8760
}
data_processor = initialize(data, input_parameters)
title = 'Installed Capacity Costs For The Small-n network'
PF.C_LCOE(pipeline_resultsFAS,  data_processor, path=plot_path+'InstalledLCOE/', figname='Installed_LCOE', Title = title)
title = 'Country Distribution of Renewables, With '+r'$\gamma_N=1$ For The Small-n network'
tab= PF.alphaPlots(pipeline_resultsFAS,  data_processor, path=plot_path+'Installedalpha/', figname='alpha', Title = title)
title = 'Cost Components of Various Scheme Combinations For The Small-n network'
PF.EU_LCOE(pipeline_resultsFAS, path=plot_path, figname='LCOE_EU', Title = title)


#%%
# for key, item in pipeline_resultsFAS.items():
#     layout_scheme = item['Layout_Scheme']
#     balancing_scheme = item['Balancing_Scheme']
#     beta = 1.0         
#     input_parameters = {
#          'Layout_Scheme': layout_scheme,
#          'Balancing_Scheme': balancing_scheme,
#          'gamma': 1,  # Fixed for now
#          'beta': beta,
#          'alpha_opt_parameter': 'min_backup_energy',
#          'choice_of_critical_parameter': 'Sand_use_mg.pr.kWh',
#          'n_alpha': 11,
#          'n_t': 8760
#      }
#     data_processor = initialize(data, input_parameters)
#     name = f"{layout_scheme}_{balancing_scheme}_{beta}"
#     layout = execute_layout(data_processor)
#     Agg = Aggregator(data_processor, Plot_path=plot_path)
#     Agg.InitializeMapper(plot_path=plot_path)
#     Agg.MapOverview()
#     C_layout = Agg.Run(layout)
#     Agg.SingleMapHandler(C_layout, name, subpath='kappamap/')
#     Agg.SingleMapHandler(C_layout, name, Normalize=True, subpath='kappamapnorm/')    
#     PF.Repackage_forCspecialsStacks2(result, data_processor, path=plot_path+'Fig7/')
#     NewRes[name] = result
#     pickle.dump(result, f)

#%%
# plot_path = 'ResPlots_smalln/'
# betas = [1.0]
# data = data_loader()  # Call the data_loader function
# pipeline_results = load_results_pickle(save_path="pipeline_results.pkl")
# layout_schemes = ['Global']
# balancing_schemes = ['Global']
# Refileing(pipeline_results, data, layout_schemes, balancing_schemes, betas,save_path= "pipeline__smalln.pkl", plot_path = plot_path)
# # data = data_loader()
# input_parameters = {
#     'Layout_Scheme': 'Global',
#     'Balancing_Scheme': 'Global',
#     'gamma': 1,  # Fixed for now
#     'beta': 1.0,
#     'alpha_opt_parameter': 'min_backup_energy',
#     'choice_of_critical_parameter': 'Sand_use_mg.pr.kWh',
#     'n_alpha': 11,
#     'n_t': 8760
# }


# # plot_path = 'ResPlots_smalln/'
# # save_path = "pipeline_results.pkl"

# data_processor = initialize(data, input_parameters)
# # layout = execute_layout(data_processor)
# # Agg = Aggregator(data_processor, Plot_path=plot_path)
# # Agg.InitializeMapper(plot_path='ResPlots_smalln/')
# # Agg.MapOverview()
# # C_layout = Agg.Run(layout)
# # name = f"Global_Global_1.0"
# # Agg.SingleMapHandler(C_layout, name, subpath='kappamap/')
# # Agg.SingleMapHandler(C_layout, name, Normalize=True, subpath='kappamapnorm/')



# flow_tracing = execute_flow_tracing(data_processor, C_layout, path=plot_path+'FTPlots/', smalln=True)
# used_kappa = calculate_utilized_capacity(data_processor, C_layout, flow_tracing)
# LCOE= execute_LCOE_n(data_processor, C_layout, input_parameters, usedkappa = used_kappa)

# result = {'Layout_Scheme': 'Global',
#           'Balancing_Scheme': 'Global',
#           'beta': 1.0,
#           'Result': C_layout,
#           'flow_tracing': flow_tracing,
#           'used_kappa': used_kappa,
#           'LCOE': execute_LCOE_n(data_processor, C_layout, input_parameters, usedkappa = used_kappa)}
# PF.Repackage_forCspecialsStacks2(result, data_processor, path=plot_path+'Fig7/')


#%%









