# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:26:18 2025

@author: heide
"""

from multiprocessing import Pool
from layout.layout_class import layout_class
from initialize.data_loader import data_loader
from initialize.data_class import data_class
from flowtracing.flowtracing_class import flowtracing_class
from flowtracing.used_kappa_class import used_kappa_class
from post_processing.LCOE3_class import LCOE2_class
from post_processing import PlotFunctions as PF
import numpy as np
import pickle
import os
import copy


def initialize(data, input_parameters):
    data_processor = data_class(data, input_parameters)
    data_processor.run()
    return data_processor

def execute_layout(data_processor):
    layout_manager = layout_class(data_processor)
    return layout_manager.run()

def execute_flow_tracing(data_processor, layout, path='FTPlots/', smalln=False):
    flow_tracer_export = flowtracing_class(data_processor, layout, Import=False, path=path, smalln=smalln)
    export_FT = flow_tracer_export.run()
    flow_tracer_import = flowtracing_class(data_processor, layout, Import=True, path=path, smalln=smalln)
    import_FT = flow_tracer_import.run()
    return {"export": export_FT, "import": import_FT}

def calculate_utilized_capacity(data_processor, layout, flow_tracing):
    used_kappa = {}
    for imex in ["import", 'export']:
        FT = flow_tracing[imex]
        kappa_utilization = used_kappa_class(data_processor, layout, FT)
        used_kappa[imex] = kappa_utilization.run()
    return used_kappa


def execute_LCOE(data_processor, layout, input_parameters, usedkappa = None):
    """OBS: Error in the LCOE calculations using FT methods. These are later
    corected When running the 'refileing function script"""
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
  #%%
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
           res = pipeline_results[name]
           
           data_processor = initialize(data, input_parameters)
           layout = res['Result']
           
           if balancing_scheme != 'NoT':
               # flow_tracing = res['flow_tracing']
               flow_tracing = res['flow_tracing']
               # Step 4: Calculate utilized capacity
               used_kappa = calculate_utilized_capacity(data_processor, layout, flow_tracing)
               imp = used_kappa['import']['trans']
               exp = used_kappa['export']['trans']
               ins = layout['kappa']['trans']
               t1 = np.allclose(imp, exp)
               t2 = np.allclose(imp, ins)
               t3 = np.allclose(ins, exp)
               if not any([t1,t2,t3]):
                   print("We now have different import, installed, and export kappa.",)
               t1 = abs(imp.sum()- exp.sum())
               t2 = abs(imp.sum()- ins.sum())
               t3 = abs(ins.sum()- exp.sum())
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
              
               
               LCOE = execute_LCOE(data_processor, layout, input_parameters, usedkappa = used_kappa)
               #quickfix:
               LCOE_class=LCOE2_class(data_processor, Prices, input_parameters, used_kappa)
               LCOE['installed']['LCOE_C']['FT_im'] = LCOE_class.LCOEWrap(layout['kappa'], imp, scope= 'country')
               LCOE['installed']['LCOE_n']['FT_im'] = LCOE_class.LCOEWrap(layout['kappa'], imp, scope= 'nodal')
               LCOE['used']['LCOE_C']['FT_im'] = LCOE_class.LCOEWrap(used_kappa['import'], imp, scope= 'country')
               LCOE['used']['LCOE_n']['FT_im'] = LCOE_class.LCOEWrap(used_kappa['import'], imp, scope= 'nodal')
               for key,_ in LCOE['used']['LCOE_C']['FT_0.5'].items():
                   LCOE['used']['LCOE_C']['FT_0.5'][key] = (
                       LCOE['used']['LCOE_C']['FT_ex'][key] + LCOE['used']['LCOE_C']['FT_im'][key]) / 2
                   LCOE['installed']['LCOE_C']['FT_0.5'][key] = (
                       LCOE['installed']['LCOE_C']['FT_ex'][key] + LCOE['installed']['LCOE_C']['FT_im'][key]) / 2
                   LCOE['used']['LCOE_n']['FT_0.5'][key] = (
                       LCOE['used']['LCOE_n']['FT_ex'][key] + LCOE['used']['LCOE_n']['FT_im'][key]) / 2
                   LCOE['installed']['LCOE_n']['FT_0.5'][key] = (
                       LCOE['installed']['LCOE_n']['FT_ex'][key] + LCOE['installed']['LCOE_n']['FT_im'][key]) / 2  

           else:
               flow_tracing = None
               used_kappa = None
               LCOE = execute_LCOE(data_processor, layout, input_parameters, usedkappa = used_kappa) 

           result = {'Layout_Scheme': layout_scheme,
                     'Balancing_Scheme': balancing_scheme,
                     'beta': beta,
                     'Result': layout,
                     'flow_tracing': flow_tracing,
                     'used_kappa': used_kappa,
                     'LCOE': LCOE}
           PF.Repackage_forCspecialsStacks2(result, data_processor, path=plot_path+'Fig7/')
           NewRes[name] = result
           pickle.dump(result, f)
    return NewRes
def run_pipeline(args):
    data, layout_scheme, balancing_scheme, beta, plot_path = args
    input_parameters = {
        'Layout_Scheme': layout_scheme,
        'Balancing_Scheme': balancing_scheme,
        'gamma': 1,  # Fixed for now
        'beta': beta,
        'alpha_opt_parameter': 'min_backup_energy',
        'choice_of_critical_parameter': 'Sand_use_mg.pr.kWh',
        'n_alpha': 11,
        # 'n_t': 200
        'n_t': 8760
    }
    name = f"{layout_scheme}_{balancing_scheme}_{beta}"
    
    data_processor = initialize(data, input_parameters)
    layout = execute_layout(data_processor)
    
    if balancing_scheme != 'NoT':
        flow_tracing = execute_flow_tracing(data_processor, layout, path=plot_path+'FTPlots/')
        used_kappa = calculate_utilized_capacity(data_processor, layout, flow_tracing)
        result = {
            'Layout_Scheme': layout_scheme,
            'Balancing_Scheme': balancing_scheme,
            'beta': beta,
            'Result': layout,
            'flow_tracing': flow_tracing,
            'used_kappa': used_kappa,
            'LCOE': execute_LCOE(data_processor, layout, input_parameters, usedkappa=used_kappa)
        }
    else:
        result = {
            'Layout_Scheme': layout_scheme,
            'Balancing_Scheme': balancing_scheme,
            'beta': beta,
            'Result': layout,
            'LCOE': execute_LCOE(data_processor, layout, input_parameters, usedkappa=None)
        }

    return result

def pipelinesaver(data, layout_schemes, balancing_schemes, betas, save_path="pipeline_results.pkl", plot_path='ResPlots/'):
    append_mode = os.path.exists(save_path)
    with open(save_path, 'ab') as f:
        args_list = [(data, layout_scheme, balancing_scheme, beta, plot_path) for layout_scheme in layout_schemes for balancing_scheme in balancing_schemes for beta in betas]
        with Pool() as pool:
            results = pool.map(run_pipeline, args_list)
            for result in results:
                pickle.dump(result, f)
    print(f"layout results saved in {save_path}")


    
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
#%%
if __name__ == '__main__':
    data = data_loader()  # Call the data_loader function

    plot_path = 'ResPlots_LargeN1/'
    save_path = "pipeline_results__LargeN1.pkl"
    data = data_loader()  # Call the data_loader function
    
    layout_schemes = ['Local']
    balancing_schemes = ['Global', 'Local', 'Nodal', 'NoT']
    betas = np.array([1.0])
    pipelinesaver(data, layout_schemes, balancing_schemes, betas, save_path=save_path, plot_path=plot_path)

    layout_schemes = ['Global']
    balancing_schemes = ['Global', 'Nodal']
    betas = np.array([1.0])
    pipelinesaver(data, layout_schemes, balancing_schemes, betas, save_path=save_path, plot_path=plot_path)

#%% Correcting Errors in LCOE 
pipeline_results = load_results_pickle(save_path=save_path)
Correct_pipeline_results = Refileing(pipeline_results, data, save_path, plot_path)
#%% Plotting
PF.Repackage_CompareLCOE(Correct_pipeline_results, data, path='ResPlotsFAB/'+'Fig6/', Type='installed')
PF.Repackage_CompareLCOE(Correct_pipeline_results, data, path='ResPlotsFAB/'+'Fig6/', Type='used')

# To enable using the data_processor for plotting, an arbitrary case is used:
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
title = 'Installed Capacity Costs For The Large-N network'
PF.C_LCOE(Correct_pipeline_results,  data_processor, path=plot_path+'InstalledLCOE/', figname='Installed_LCOE', Title = title)
title = 'Country Distribution of Renewables, With '+r'$\gamma_N=1$ For The Large-N network'
tab= PF.alphaPlots(Correct_pipeline_results,  data_processor, path=plot_path+'Installedalpha/', figname='alpha', Title = title)
title = 'Cost Components of Various Scheme Combinations For The Large-N network'
PF.EU_LCOE(Correct_pipeline_results, path=plot_path, figname='LCOE_EU', Title = title)

