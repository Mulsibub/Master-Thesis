import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('QtAgg')
from matplotlib import pyplot as plt
import seaborn as sns
from initialize.data_loader2 import data_loader
import textwrap
from matplotlib.colors import LinearSegmentedColormap
import os
from math import nan
import networkx as nx
#%%

class Plotter1:
    def __init__(self, data):
        self.data = data
        self.country_of_n = data['BusData']['country'].to_numpy()  # Convert to NumPy array
        self.country_list = np.unique(self.country_of_n)
        self.country_int = []
        for country in self.country_list:
            start_index = np.where(self.country_of_n == country)[0][0]
            end_index = np.where(self.country_of_n == country)[0][-1]
            
            # Always append the tuple (start_index, end_index), even if they are the same
            self.country_int.append((start_index, end_index + 1))  # Add +1 to make inclusive slicing easier
            
        self.n_country = len(self.country_list)
        
        Load_avg_n = data['Load'].mean(axis=0)
        Load_avg = Load_avg_n.sum()
        Load_avg_c  = np.zeros(len(self.country_list))
        for i, (start, end) in enumerate(self.country_int):
            Load_avg_c[i] = (Load_avg_n[start:end]).sum() 
        self.Load_avg_c = Load_avg_c
        self.Load_avg_n = Load_avg_n
        self.Load_avg = Load_avg
        self.weight_c = np.divide(Load_avg_c, Load_avg)
        self.weight_n = np.divide(Load_avg_n, Load_avg)
        
        
    
        
        
    def plot_transmission_capacities(self, results, layout_schemes, balancing_schemes, countries, TransmissionMethod ="Load"):
        # postage stamp currently not a viable method as no capacity allocations are changed only cost
        
        
        sns.set_style("whitegrid")
        Load = self.Load_avg_c
        M = 'country_trans_'+TransmissionMethod
        for beta in [1.0, 2.0]:
            fig, axes = plt.subplots(len(layout_schemes), 1, figsize=(6 * len(layout_schemes), 5), sharey=True)
            
            for i, layout in enumerate(layout_schemes):
                ax = axes[i] if len(layout_schemes) > 1 else axes
                data = []
                
                # Collect transmission capacities per country for each balancing scheme
                for key, entry in results.items():
                    if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                        if 'misc' in entry['Result'] and M in entry['Result']['misc']:
                            values = np.divide(entry['Result']['misc'][M],Load)
                            if isinstance(values, np.ndarray) and len(values) == len(countries):
                                for country, value in zip(countries, values):
                                    data.append([country, entry['Balancing_Scheme'], value])
                
                df = pd.DataFrame(data, columns=['Country', 'Balancing_Scheme', 'Value'])
                df.sort_values(by='Country', inplace=True)
                
                # Add overall European level (EU) as the first group
                eu_values = []
                for key, entry in results.items():
                    if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                        eu_values.append(["EU", entry['Balancing_Scheme'], entry['Result']['misc'][M].sum()/self.Load_avg])                         
                df = pd.concat([pd.DataFrame(eu_values, columns=['Country', 'Balancing_Scheme', 'Value']), df], ignore_index=True)
                
                # Ensure correct categorical order with extra space after "EU"
                order = np.insert(countries, 0, "EU")  # No empty string
                sns.barplot(x='Country', y='Value', hue='Balancing_Scheme', data=df, ax=ax, order=order)
                # Manually adjust tick positions for spacing
                ax.set_xticklabels(order, rotation=90)
                ax.set_title(fr"Normalized Transmission Capacities ({layout}), $\beta$ = {beta}, Transmissions Cost Allocation Method: "+TransmissionMethod)
                ax.set_xlabel("Country")
                ax.set_ylabel(r"$ \frac{\kappa_{C}^T}{\langle L_{C}(t) \rangle _t}}$ [MW km]/[MW]")
                ax.get_legend().remove()
            # Create a shared legend below the last plot
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(balancing_schemes), bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout()
            fig.savefig(f'plots/sumkappaT_{beta}'+'_'+TransmissionMethod+'method.png', dpi=200, bbox_inches='tight')
            plt.close()
            
    def plot_transmission_LCOE(self, results, layout_schemes, balancing_schemes, countries, TransmissionMethod ="Load"):
        # "Look into metods for possibly using different methods of kappa_T". Might  be done unsing entry['Result']['kappa']['country_trans']
        sns.set_style("whitegrid")
        M = 'LCOE_'+TransmissionMethod
        for beta in [1.0, 2.0]:
            fig, axes = plt.subplots(len(layout_schemes), 1, figsize=(6 * len(layout_schemes), 5), sharey=True)
            
            for i, layout in enumerate(layout_schemes):
                ax = axes[i] if len(layout_schemes) > 1 else axes
                data = []
                
                # Collect transmission capacities per country for each balancing scheme
                for key, entry in results.items():
                    if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                        values = entry[M]['LCOE_C']['LCOE_trans']
                        if isinstance(values, np.ndarray) and len(values) == len(countries):
                            for country, value in zip(countries, values):
                                data.append([country, entry['Balancing_Scheme'], value])
                
                df = pd.DataFrame(data, columns=['Country', 'Balancing_Scheme', 'Value'])
                df.sort_values(by='Country', inplace=True)
                
                # Add overall European level (EU) as the first group
                eu_values = []
                for key, entry in results.items():
                    if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                        eu_values.append(["EU", entry['Balancing_Scheme'], entry[M]['LCOE_EU']['LCOE_trans']])
                                               
                df = pd.concat([pd.DataFrame(eu_values, columns=['Country', 'Balancing_Scheme', 'Value']), df], ignore_index=True)
                
                # Ensure correct categorical order with extra space after "EU"
                order = np.insert (countries, 0, "EU")  # No empty string
                sns.barplot(x='Country', y='Value', hue='Balancing_Scheme', data=df, ax=ax, order=order)
                # Manually adjust tick positions for spacing
                ticks = list(range(len(order)))  # Tick positions
                ticks[1:] = [t + 1 for t in ticks[1:]]  # Shift all but the first
                
                # ax.set_xticks(countries)
                ax.set_xticklabels(order, rotation=90)
                ax.set_title(fr"Country Transmission LCOE ({layout}), $\beta$ = {beta}, Transmissions Cost Allocation Method: "+TransmissionMethod)
                ax.set_xlabel("Country")
                ax.set_ylabel(r"$LCOE_{C}^T$ [EUR/MWh]")
                ax.get_legend().remove()
                ax.set_ylim(0, 65)
                
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(balancing_schemes), bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout()
            fig.savefig('plots/TransmissionLCOE_'+str(beta)+'_'+TransmissionMethod+'method.png', dpi=200,  bbox_inches='tight')
            plt.close(fig)

        
    def plot_country_lcoe(self, results, layout_schemes, balancing_schemes, countries, TransmissionMethod = "Load"):
        sns.set_style("whitegrid")
        M = 'LCOE_'+TransmissionMethod
        for beta in [1.0, 2.0]:
            fig, axes = plt.subplots(len(layout_schemes), 1, figsize=(6 * len(layout_schemes), 5), sharey=True)
            
            for i, layout in enumerate(layout_schemes):
                ax = axes[i] if len(layout_schemes) > 1 else axes
                data = []
                
                # Collect transmission capacities per country for each balancing scheme
                for key, entry in results.items():
                    if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                        values = entry['LCOE']['LCOE_C'][M]['LCOE']
                        if isinstance(values, np.ndarray) and len(values) == len(countries):
                            for country, value in zip(countries, values):
                                data.append([country, entry['Balancing_Scheme'], value])
                
                df = pd.DataFrame(data, columns=['Country', 'Balancing_Scheme', 'Value'])
                df.sort_values(by='Country', inplace=True)
                
                # Add overall European level (EU) as the first group
                eu_values = []
                for key, entry in results.items():
                    if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                        eu_values.append(["EU", entry['Balancing_Scheme'], entry['LCOE']['LCOE_EU']['LCOE']])
                                               
                df = pd.concat([pd.DataFrame(eu_values, columns=['Country', 'Balancing_Scheme', 'Value']), df], ignore_index=True)
                
                # Ensure correct categorical order with extra space after "EU"
                order = np.insert (countries, 0, "EU")  # No empty string
                sns.barplot(x='Country', y='Value', hue='Balancing_Scheme', data=df, ax=ax, order=order)
                # Manually adjust tick positions for spacing
                ticks = list(range(len(order)))  # Tick positions
                ticks[1:] = [t + 1 for t in ticks[1:]]  # Shift all but the first
                
                # ax.set_xticks(countries)
                ax.set_xticklabels(order, rotation=90)
                ax.set_title(fr"Country LCOE ({layout}), $\beta$ = {beta}, Transmissions Cost Allocation Method: "+TransmissionMethod)
                ax.set_xlabel("Country")
                ax.set_ylabel(r"$LCOE_{C}$ [EUR/MWh]")
                ax.get_legend().remove()
                ax.set_ylim(0, 200)
                
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(balancing_schemes), bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout()
            fig.savefig('plots/LCOE_'+str(beta)+'_'+TransmissionMethod+'method.png', dpi=200,  bbox_inches='tight')
            plt.close(fig)
            

        for i, layout in enumerate(layout_schemes):
            ax = axes[i] if len(layout_schemes) > 1 else axes
            data = []
            
            # Collect transmission capacities per country for each balancing scheme
            for key, entry in results.items():
                if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                    values = entry[M]['LCOE_C']['LCOE']
                    if isinstance(values, np.ndarray) and len(values) == len(countries):
                        for country, value in zip(countries, values):
                            data.append([country, entry['Balancing_Scheme'], value])
            
            df = pd.DataFrame(data, columns=['Country', 'Balancing_Scheme', 'Value'])
            df.sort_values(by='Country', inplace=True)
            
            # Add overall European level (EU) as the first group
            eu_values = []
            for key, entry in results.items():
                if entry['Layout_Scheme'] == layout and entry['beta'] == beta:
                    eu_values.append(["EU", entry['Balancing_Scheme'], entry[M]['LCOE_EU']['LCOE']])
                                           
            df = pd.concat([pd.DataFrame(eu_values, columns=['Country', 'Balancing_Scheme', 'Value']), df], ignore_index=True)
            
            # Ensure correct categorical order with extra space after "EU"
            order = np.insert (countries, 0, "EU")  # No empty string
            sns.barplot(x='Country', y='Value', hue='Balancing_Scheme', data=df, ax=ax, order=order)
            # Manually adjust tick positions for spacing
            ticks = list(range(len(order)))  # Tick positions
            ticks[1:] = [t + 1 for t in ticks[1:]]  # Shift all but the first
            
            # ax.set_xticks(countries)
            ax.set_xticklabels(order, rotation=90)
            ax.set_title(fr"Country LCOE ({layout}), $\beta$ = {beta}, Transmissions Cost Allocation Method: "+TransmissionMethod)
            ax.set_xlabel("Country")
            ax.set_ylabel(r"$LCOE_{C}$ [EUR/MWh]")
            ax.get_legend().remove()
            ax.set_ylim(0, 200)
            
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(balancing_schemes), bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        fig.savefig('plots/LCOE_'+str(beta)+'_'+TransmissionMethod+'method.png', dpi=200,  bbox_inches='tight')
        plt.close()

            
    def plot_lcoe_vs_beta(self, results, layout_schemes, balancing_schemes, betas):
        """
        Generate a plot for LCOE vs beta.
        
        Parameters:
        - results (dict): The output from the pipeline function.
        - layout_schemes (list): List of layout schemes.
        - balancing_schemes (list): List of balancing schemes.
        - betas (list): List of beta values.
        """
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, len(layout_schemes), figsize=(6 * len(layout_schemes), 5), sharey=True)
    
        for i, layout in enumerate(layout_schemes):
            ax = axes[i] if len(layout_schemes) > 1 else axes
            
            for balancing in balancing_schemes:
                beta_vals = []
                LCOE_vals = []
                
                for key, entry in results.items():
                    if entry['Layout_Scheme'] == layout and entry['Balancing_Scheme'] == balancing:
                        beta_vals.append(entry['beta'])
                        LCOE_vals.append(entry['LCOE_Load']['LCOE_EU']['LCOE'])
                
                # Sort beta values and reorder LCOE_vals accordingly
                sorted_indices = np.argsort(beta_vals)  # Get sorted indices
                beta_vals = np.array(beta_vals)[sorted_indices]  # Sort beta values
                LCOE_vals = np.array(LCOE_vals)[sorted_indices]  # Reorder LCOE values
                
                ax.plot(beta_vals, LCOE_vals, marker='o', label=balancing)
            
            ax.set_title(fr"$LCOE$ Over $\beta$ ({layout})")
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel(r"$LCOE_{EU}$ [EUR/MWh]")
            ax.legend()
    
        plt.tight_layout()
        fig.savefig('plots/LCOEoverbeta.png', dpi=200)
        plt.close()  

    
    
#%%
# def alphaPlots(Dicts,  data, path='', figname='alpha', Title = ''):
#     Tit = {'Global_Global_1.0': r'Global',
#            'LocalGAS_Nodal_1.0': r'GAS',
#            'Local_Global_1.0': r'Local'}
#     country_list = data.country_list
#     for key, tit, in Tit.items():
#         item = Dicts[key]
#         GenW = item['Result']['Generation']['wind'].mean(axis=0)
#         GenS = item['Result']['Generation']['solar'].mean(axis=0)
#         Load = item['Result']['Load_C'].mean(axis=0)
#         alphaW_c =  GenW/Load
#         alphaS_c =  GenS/Load
#         df = pd.DataFrame({'Load':Load, r'alpha_C^W': alphaW_c, r'alpha_C^S': alphaS_c},index= country_list)
#         df = df.sort_values(by='Load', ascending=False)
#         df = df.drop(columns=['Load'])
#         colors=['b',  'gold']
#         fig, ax = plt.subplots(figsize=(7, 4))
#         df.plot(kind='bar', stacked=True, ax=ax,
#                 color=colors)
#         ax.set_xlabel("Countries")
#         ax.set_ylabel(r"$\gamma_C$")
#         ax.get_legend().remove()
#         ax.tick_params(axis='x', rotation=45)
#         wrapped_title = "\n".join(textwrap.wrap(Title+' And The '+tit+' Layout Scheme', 80))
#         fig.suptitle(wrapped_title, fontsize=13, color='Black') 
#         fig.tight_layout()
#         fig.savefig(path+tit+figname+'.png', dpi=200)
#         plt.close()
#         print(f"Figure saved as {path+key+figname+'.png'}")

def alphaPlots(Dicts, data, path='', figname='alpha', Title=''):
    Tit = {
        'Global_Global_1.0': r'Global',
        'LocalGAS_Nodal_1.0': r'GAS',
        'Local_Global_1.0': r'Local'
    }

    country_list = data.country_list
    table_frames = {}

    for key, tit in Tit.items():
        item = Dicts[key]
        GenW = item['Result']['Generation']['wind'].mean(axis=0)
        GenS = item['Result']['Generation']['solar'].mean(axis=0)
        Load = item['Result']['Load_C'].mean(axis=0)

        alphaW_c = GenW / Load
        alphaS_c = GenS / Load
        gammaC = alphaW_c + alphaS_c

        df = pd.DataFrame({
            r'$\alpha_C^W$': alphaW_c,
            r'$\alpha_C^S$': alphaS_c,
            r'$\gamma_C$': gammaC
        }, index=country_list)

        df_sorted = df.sort_values(by=r'$\gamma_C$', ascending=False)
        table_frames[tit] = df_sorted

        # Plotting
        colors = ['b', 'gold']
        fig, ax = plt.subplots(figsize=(7, 4))
        df_sorted[[r'$\alpha_C^W$', r'$\alpha_C^S$']].plot(
            kind='bar', stacked=True, ax=ax, color=colors)
        ax.set_xlabel("Countries")
        ax.set_ylabel(r"$\gamma_C$")
        ax.get_legend().remove()
        ax.tick_params(axis='x', rotation=45)
        wrapped_title = "\n".join(textwrap.wrap(Title + ' And The ' + tit + ' Layout Scheme', 80))
        fig.suptitle(wrapped_title, fontsize=13, color='Black')
        fig.tight_layout()
        fig.savefig(path + tit + figname + '.png', dpi=200)
        plt.close()
        print(f"Figure saved as {path + tit + figname + '.png'}")

    # Combine tables into a single DataFrame
    combined_df = pd.concat([
        table_frames['Global'].add_suffix(' (Global)'),
        table_frames['GAS'].add_suffix(' (GAS)'),
        table_frames['Local'].add_suffix(' (Local)')
    ], axis=1)

    # Print single LaTeX table with multicolumn header
    col_header = r"""
\begin{table}[ht]
\centering
\small
\begin{tabular}{lccc|ccc|ccc}
\toprule
\textbf{Country} 
& \multicolumn{3}{c|}{\textbf{Global}} 
& \multicolumn{3}{c|}{\textbf{GAS}} 
& \multicolumn{3}{c}{\textbf{Local}} \\
& $\alpha_C^W$ & $\alpha_C^S$ & $\gamma_C$ 
& $\alpha_C^W$ & $\alpha_C^S$ & $\gamma_C$
& $\alpha_C^W$ & $\alpha_C^S$ & $\gamma_C$ \\
\midrule
"""

    latex_rows = ""
    for country in combined_df.index:
        row = [country]
        row += [f"{combined_df.at[country, col]:.3f}" for col in combined_df.columns]
        latex_rows += " & ".join(row) + r" \\" + "\n"

    col_footer = r"""\bottomrule
\end{tabular}
\caption{Comparison of $\alpha_C^W$, $\alpha_C^S$, and $\gamma_C$ for Global, GAS, and Local layouts.}
\label{tab:alpha_comparison}
\end{table}
"""

    # Output the full LaTeX code
    print("\n% --- Combined LaTeX Table for All Layouts ---")
    print(col_header + latex_rows + col_footer)

    return combined_df


                          #%%
def PostageSplot(optimal_delta_n, optimal_delta_c, min_var_n, min_var_c, LCOE_n_var, LCOE_c_var, LCOE_n, LCOE_c, name, delta, path='', figname=''):
    mpl.use('QtAgg')
    fig, axs = plt.subplots(1,2, figsize=(12, 6))
    
    # Variance plot
    axs[0].scatter(optimal_delta_n, min_var_n, color='red', marker='X', label=f'Node Optimal δ = {optimal_delta_n:.2f} \nMin Variance = {min_var_n:.2e}')
    axs[0].scatter(optimal_delta_c, min_var_c, color='blue', marker='X', label=f'Country Optimal δ = {optimal_delta_c:.2f} \nMin Variance = {min_var_c:.2e}')
    axs[0].plot(delta, LCOE_n_var, color='orange', alpha=0.5, marker='o', label='Variance of $LCOE_n$')
    axs[0].plot(delta, LCOE_c_var, color='green', alpha=0.5, marker='o', label='Variance of $LCOE_c$')
    axs[0].set_title('Variance of $LCOE$ over $\delta$')
    axs[0].set_xlabel('δ (Cost-mixing parameter)')
    axs[0].set_ylabel('Variance of $LCOE$')
    axs[0].legend()
    axs[0].grid(True)
    # Box plots for nodes and countries (10 evenly spaced delta values)
    selected_indices = np.linspace(0, len(delta) - 1, 11, dtype=int)
    selected_deltas = delta[selected_indices]
    
    width = 0.04
    offset = (width+0.01)/2  # Horizontal shift for separation

    # Nodes
    LCOE_n_samples = [LCOE_n[i,:] for i in selected_indices]
    box1 = axs[1].boxplot(LCOE_n_samples, positions=selected_deltas - offset, widths=width, patch_artist=True, 
                 boxprops=dict(facecolor='purple', alpha=0.5))

    # Countries
    LCOE_c_samples = [LCOE_c[i,:] for i in selected_indices]
    box2 = axs[1].boxplot(LCOE_c_samples, positions=selected_deltas + offset, widths=width, patch_artist=True, 
                 boxprops=dict(facecolor='brown', alpha=0.5))

    # Adjust x-limits to keep spacing clean
    axs[1].set_xlim(min(selected_deltas) - 0.05, max(selected_deltas) + 0.05)

    # Rotate x-ticks for better readability
    axs[1].set_xticks(selected_deltas, [f'{d:.2f}' for d in selected_deltas], rotation=45)

    axs[1].set_title('LCOE for Nodes and Countries over δ')
    axs[1].set_xlabel('δ (Cost-mixing parameter)')
    axs[1].set_ylabel('LCOE')
    axs[1].set_yscale('log')
    axs[1].legend([box1["boxes"][0], box2["boxes"][0]], ['Nodes', 'Countries'])
    axs[1].grid(True)
    
    fig.suptitle( 'Postagestamp Optimisation For ' + name, fontsize=14)
    fig.tight_layout()
    fig.savefig(path + figname+'.png', dpi =100)
    plt.close()  
#%%
def EU_LCOE(Dict, path='', figname='', Title = ''):
    Types = [r'$LCOE^{\kappa^W}_{EU}$',r'$LCOE^{\kappa^S}_{EU}$',r'$LCOE^{\kappa^B}_{EU}$',r'$LCOE^{B}_{EU}$']
    tech_colors = {'LCOE_wind': 'b', 'LCOE_solar': 'gold', 'LCOE_backup': 'r',
                   'LCOE_backup_energy': 'firebrick', 'LCOE_trans':'green'}
    Vars = {'LCOE_solar':r'$LCOE^{\kappa^S}_{EU}$',
            'LCOE_wind': r'$LCOE^{\kappa^W}_{EU}$',
            'LCOE_backup': r'$LCOE^{\kappa^B}_{EU}$',
            'LCOE_trans': r'$LCOE^{\kappa^T}_{EU}$',
            'LCOE_backup_energy': r'$LCOE^{B}_{EU}$'}
    
    Tit = {'Global_Global_1.0': r'Global - Global',
           'Global_Nodal_1.0': r'Global - Nodal',
           'LocalGAS_Global_1.0': r'LocalGAS - Global',
           'LocalGAS_Nodal_1.0': r'LocalGAS - Nodal',
           'Local_Global_1.0': r'Local - Global',
           'Local_Local_1.0': r'Local - Local',
           'Local_Nodal_1.0': r'Local - Nodal',
           'Local_NoT_1.0': r'Local - NoT'}
    data={}                 
    for i, _ in Dict.items():
        j = Tit[i]
        data[j] = Dict[i]['LCOE']['installed']['LCOE_EU']
    df = pd.DataFrame(data)
    df = df.T.sort_values(by='LCOE', ascending=False)
    df = df.drop(columns=['LCOE'])
    colors=[tech_colors.get(x) for x in df.columns]
    legends=[Vars.get(x) for x in df.columns]
    df.columns = [Vars.get(x) for x in df.columns]
    fig, ax = plt.subplots()
    df.plot(kind='bar', stacked=True, ax=ax,
            color=colors)
    ax.set_xlabel("Layouts")
    ax.set_ylabel("[EURO/MWh]")
    ax.legend(legends)
    # ax.legend([Vars.get(x) for x in df.columns])
    ax.tick_params(axis='x', rotation=14)
    fig.suptitle(Title, fontsize=16, color='Black') 
    fig.tight_layout()
    fig.savefig(path+figname+'.png', dpi=200)
    plt.close()
    
def EU_LCOE_compare(BigDict, SmallDict, path='', figname='', Title=''):
    tech_colors = {'LCOE_wind': 'b', 'LCOE_solar': 'gold', 'LCOE_backup': 'r',
                   'LCOE_backup_energy': 'firebrick', 'LCOE_trans': 'green'}
    
    Vars = {'LCOE_wind': r'$LCOE^{\kappa^W}_{EU}$',
            'LCOE_solar': r'$LCOE^{\kappa^S}_{EU}$',
            'LCOE_backup': r'$LCOE^{\kappa^B}_{EU}$',
            'LCOE_backup_energy': r'$LCOE^{E^B}_{EU}$',
            'LCOE_trans': r'$LCOE^{\kappa^T}_{EU}$'}

    stack_order = ['LCOE_wind', 'LCOE_solar', 'LCOE_backup', 'LCOE_backup_energy', 'LCOE_trans']

    Tit = {'Global_Global_1.0': r'Global - Global',
           'Global_Nodal_1.0': r'Global - Nodal',
           'LocalGAS_Global_1.0': r'LocalGAS - Global',
           'LocalGAS_Nodal_1.0': r'LocalGAS - Nodal',
           'Local_Global_1.0': r'Local - Global',
           'Local_Local_1.0': r'Local - Local',
           'Local_Nodal_1.0': r'Local - Nodal',
           'Local_NoT_1.0': r'Local - NoT'}

    def extract_data(Dict):
        data = {}
        for key in Dict:
            layout = Tit.get(key, key)
            data[layout] = Dict[key]['LCOE']['installed']['LCOE_EU']
        df = pd.DataFrame(data).T
        df = df.drop(columns=['LCOE'], errors='ignore')
        df = df[[col for col in stack_order if col in df.columns]]  # Ordered subset
        df.columns = [Vars[c] for c in df.columns]
        return df

    df1 = extract_data(BigDict).astype(float)
    df2 = extract_data(SmallDict).astype(float)

    layouts = df1.index
    x = np.arange(len(layouts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bottoms1 = np.zeros(len(x), dtype=float)
    bottoms2 = np.zeros(len(x), dtype=float)

    for key in stack_order:
        col = Vars[key]
        if col not in df1.columns: continue
        values1 = df1[col].values
        values2 = df2[col].values
        color = tech_colors[key]
        ax.bar(x - width/2, values1, width, label=col + ' (Large-N)', bottom=bottoms1, color=color)
        ax.bar(x + width/2, values2, width, label=col + ' (Small-N)', bottom=bottoms2, color=color, hatch='//')
        bottoms1 += values1
        bottoms2 += values2

    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=14)
    ax.set_xlabel("Layouts")
    ax.set_ylabel("[EURO/MWh]")
    ax.set_title(Title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(path + figname + '.png', dpi=200)
    plt.close()
    
    
def C_LCOE(Dicts,  data_processor, path='', figname='', Title = ''):
    tech_colors = {'LCOE_wind': 'b', 'LCOE_solar': 'gold', 'LCOE_backup': 'r',
                   'LCOE_backup_energy': 'firebrick', 'LCOE_trans':'green'}
    Vars = {'LCOE_solar':r'$LCOE^{\kappa^S}_{EU}$',
            'LCOE_wind': r'$LCOE^{\kappa^W}_{EU}$',
            'LCOE_backup': r'$LCOE^{\kappa^B}_{EU}$',
            'LCOE_trans': r'$LCOE^{\kappa^T}_{EU}$',
            'LCOE_backup_energy': r'$LCOE^{B}_{EU}$'}
    
    Tit =  {'Global_Global_1.0': r' For The "Global - Global" Scheme',
             # 'Global_Local_1.0': r'Global CFProp $\beta =1$, with Local Synchronized Balancing',
             'Global_Nodal_1.0': r' For The "Global - Nodal" Scheme',
             # 'Global_NoT_1.0': r'Global CFProp $\beta =1$, with NoT Balancing',
             'Local_Global_1.0': r' For The "Local - Global" Scheme',
             'Local_Local_1.0': r' For The "Local - Local" Scheme',
             'Local_Nodal_1.0': r' For The "Local - Nodal" Scheme',
             'Local_NoT_1.0': r' For The "Local - NoT" Scheme',
             'LocalGAS_Global_1.0': r' For The "GAS - Global" Scheme',
             'LocalGAS_Nodal_1.0': r' For The "GAS - Nodal" Scheme'}
    country_list = data_processor.country_list
    for key, item, in Dicts.items():
        Data = item['LCOE']['installed']['LCOE_C']['Load']
        df = pd.DataFrame(Data,index=country_list)
        df = df.sort_values(by='LCOE', ascending=False)
        df = df.drop(columns=['LCOE'])
        colors=[tech_colors.get(x) for x in df.columns]
        legends=[Vars.get(x) for x in df.columns]
        df.columns = [Vars.get(x) for x in df.columns]
        fig, ax = plt.subplots(figsize=(7, 4))
        df.plot(kind='bar', stacked=True, ax=ax,
                color=colors)
        ax.set_xlabel("Countries")
        ax.set_ylabel(r"$LCOE_C$ [EURO/MWh]")
        ax.legend(legends, loc='upper right')
        ax.tick_params(axis='x', rotation=45)
        wrapped_title = "\n".join(textwrap.wrap(Title+' '+Tit[key], 80))
        fig.suptitle(wrapped_title, fontsize=13, color='Black') 
        fig.tight_layout()
        fig.savefig(path+key+figname+'.png', dpi=200)
        plt.close()
        print(f"Figure saved as {path+key+figname+'.png'}")
        

        
        
   
#%% Functions for stacked bar plot comparisons for various transmission allocation methods


def plot_stacked_LCOE(Base_LCOE_dict, Transmission_LCOE_df, country_labels, EU_base_dict,
                      EU_trans, L_c_avg=None, Title='', path='', figname='', ylab ='LCOE [€/MWh]'):
    num_countries = len(next(iter(Base_LCOE_dict.values())))
    num_methods = Transmission_LCOE_df.shape[1]

    x = np.arange(num_countries)
    total_bar_width = 0.8
    method_bar_width = total_bar_width / num_methods

    if L_c_avg is not None:
        sorted_indices = np.argsort(-L_c_avg)
        for tech in Base_LCOE_dict:
            Base_LCOE_dict[tech] = Base_LCOE_dict[tech][sorted_indices]
        Transmission_LCOE_df = Transmission_LCOE_df.iloc[sorted_indices]
        country_labels = np.array(country_labels)[sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Legend tracker
    legend_labels = set()

    # --- EU bar ---
    avg_x = -2
    base_colors = {'LCOE_wind': 'b', 'LCOE_solar': 'gold', 'LCOE_backup': 'r', 'LCOE_backup_energy': 'firebrick'}
    bottom_eu = 0
    for tech in ['LCOE_wind', 'LCOE_solar', 'LCOE_backup', 'LCOE_backup_energy']:
        label = tech.replace('LCOE_', '').capitalize()
        show_label = label not in legend_labels
        if show_label:
            legend_labels.add(label)

        ax.bar(avg_x, EU_base_dict[tech], bottom=bottom_eu,
               label=label if show_label else None,
               color=base_colors.get(tech, 'gray'), edgecolor='black')
        bottom_eu += EU_base_dict[tech]

    # Add transmission cost to EU bar
    ax.bar(avg_x, EU_trans, bottom=bottom_eu,
           label='Transmission (EU)', color='green', edgecolor='black')

    # --- Country bars ---
    bottom = np.zeros_like(next(iter(Base_LCOE_dict.values())))
    for tech, values in Base_LCOE_dict.items():
        label = tech.replace('LCOE_', '').capitalize()
        show_label = label not in legend_labels
        if show_label:
            legend_labels.add(label)

        ax.bar(x, values, bottom=bottom,
               label=label if show_label else None,
               color=base_colors.get(tech, 'gray'), edgecolor='black')
        bottom += values

    # --- Transmission bars ---
    colors = ['lightgreen', 'forestgreen', 'limegreen', 'darkgreen', 'lime', 'olive', 'springgreen']
    labels = Transmission_LCOE_df.columns.tolist()

    for i, method in enumerate(labels):
        x_offset = x - total_bar_width / 2 + i * method_bar_width + method_bar_width / 2
        ax.bar(x_offset, Transmission_LCOE_df[method], bottom=bottom,
               width=method_bar_width, label=method,
               color=colors[i % len(colors)], edgecolor='black')

    ax.set_xlabel('Countries')
    ax.set_ylabel(ylab)
    ax.set_title(Title)

    xtick_positions = np.concatenate(([avg_x], x))
    xtick_labels = np.insert(country_labels, 0, 'EU')
    ax.set_xticks(xtick_positions)
    ax.set_xlim([np.min(xtick_positions)-0.5, np.max(xtick_positions)+0.5])
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.legend(ncol=1, loc='upper center', bbox_to_anchor=(1.1, 0.75))
    plt.tight_layout()
    fig.savefig(path + figname + '.png', dpi=200)
    plt.close()
    
def Repackage_forCspecialsStacks(results, data_processor, path='ResPlotsMar/Fig7/'):
    Tech = ['LCOE_wind', 'LCOE_solar', 'LCOE_backup', 'LCOE_backup_energy']

    # Base generation LCOE per country
    Base_LCOE_dict = {}
    for T in Tech:
        Base_LCOE_dict[T] = np.array(results['LCOE']['LCOE_C']['Half'][T])

    # Transmission LCOE
    LCOE_T = {}
    for method, res in results['LCOE']['LCOE_C'].items():
        temp = res['LCOE_trans']
        if temp.ndim == 2:
            temp = temp[0]
        LCOE_T[method] = temp
    LCOE_T = pd.DataFrame(LCOE_T)

    # EU-average LCOE components
    EU_base_dict = {T: results['LCOE']['LCOE_EU'][T] for T in Tech}
    EU_trans = results['LCOE']['LCOE_EU']['LCOE_trans']

    country_labels = data_processor.country_list
    figname = data_processor.Layout_Scheme + '_' + data_processor.Balancing_scheme
    Title = data_processor.Layout_Scheme + '_' + data_processor.Balancing_scheme + ' Comparison of Transmission Costs Allocation Methods'

    # Load-based sorting
    L_n_avg = data_processor.Load.mean(axis=0)
    L_c_avg = np.zeros(data_processor.n_country)
    for i, (start, end) in enumerate(data_processor.country_int):
        L_c_avg[i] = L_n_avg[start:end].sum()

    plot_stacked_LCOE(Base_LCOE_dict, LCOE_T, country_labels, EU_base_dict, EU_trans,
                      L_c_avg=L_c_avg, Title=Title, path=path, figname=figname)

    
#%%
def plot_CompareLCOE(LCOE_C, LCOE_EU, country_list, L_C_avg, Title='', path='', figname='', ylab=r'$LCOE_c$ [EUR/MWh]'):
    # Convert inputs to Series for easier alignment
    load_series = pd.Series(L_C_avg, index=country_list)
    
    # Sort by descending load
    sorted_countries = load_series.sort_values(ascending=False).index.tolist()
    
    # Filter and reorder LCOE data
    data = LCOE_C.loc[sorted_countries]

    # Get number of countries and methods
    n_countries = len(data)
    n_methods = data.shape[1]
    
    # Bar positioning setup
    bar_width = 0.15
    index = np.arange(n_countries)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10.colors
    group_offset = bar_width * n_methods / 2  # To center bars
    Cpos = index*n_methods*bar_width + index*bar_width - group_offset
    for i, column in enumerate(data.columns):
        ax.bar(Cpos +(i-n_methods/2)*bar_width, data[column], bar_width, label=column, color=colors[i % len(colors)])
    # Add EU average line
    ax.axhline(y=LCOE_EU, color='gray', linestyle='--', linewidth=1.5)

    # Labels, ticks, and title
    # ax.set_xlabel('Country')
    ax.set_ylabel(ylab)
    wrapped_title = "\n".join(textwrap.wrap(Title, 80))
    fig.suptitle(wrapped_title, fontsize=14, color='Black')
    ax.set_xticks(Cpos-bar_width/2)
    ax.set_xticklabels(sorted_countries, rotation=45, ha='right')
    ax.set_xlim([Cpos[0] -(n_methods/2)*bar_width-bar_width*1.5, np.max(Cpos) +(n_methods/2)*bar_width+ bar_width*1.5])
    ax.legend(ncol=n_methods, loc='lower center', bbox_to_anchor=(0.5, -0.4))

    plt.tight_layout()

    # Save figure if path and name are provided
    if path and figname:
        if not os.path.exists(path):
            print('newfolder')
            os.makedirs(path)
        plt.savefig(path+ figname+'.png', dpi=200)
    plt.close()
    
def plot_MethodCentricLCOE(Dict_LCOE_C, LCOE_EU_dict, country_list, L_C_avg, path='', ylab=r'$LCOE_c$ [EUR/MWh]'):
    keysandNames = {
        'Local_NoT_1.0': 'Local-NoT',
        'Local_Local_1.0': 'Local-Local',
        'Local_Nodal_1.0': 'Local-Nodal',
        'Local_Global_1.0': 'Local-Global',
        'LocalGAS_Nodal_1.0': 'GAS-Nodal',
        'LocalGAS_Global_1.0': 'GAS-Global',
        'Global_Nodal_1.0': 'Global-Nodal',
        'Global_Global_1.0': 'Global-Global'
    }

    load_series = pd.Series(L_C_avg, index=country_list)
    sorted_countries = load_series.sort_values(ascending=False).index.tolist()

    # Transpose structure: method -> scenario -> values
    all_methods = list(next(iter(Dict_LCOE_C.values())).keys())
    method_data = {m: {} for m in all_methods}
    for scenario, methods in Dict_LCOE_C.items():
        for method, values in methods.items():
            method_data[method][scenario] = values

    colors = plt.cm.tab10.colors

    for method, scenario_vals in method_data.items():
        # Reorder by keysandNames order
        ordered_keys = [k for k in keysandNames if k in scenario_vals]
        ordered_names = [keysandNames[k] for k in ordered_keys]
        df = pd.DataFrame({keysandNames[k]: scenario_vals[k] for k in ordered_keys}, index=country_list)
        df = df.loc[sorted_countries]

        n_scenarios = df.shape[1]
        n_countries = len(df)
        bar_width = 0.15
        index = np.arange(n_countries)

        fig, ax = plt.subplots(figsize=(18, 4))
        group_offset = bar_width * n_scenarios / 2
        Cpos = index * n_scenarios * bar_width + index * bar_width - group_offset

        for i, scenario in enumerate(df.columns):
            ax.bar(Cpos + (i - n_scenarios / 2) * bar_width, df[scenario], bar_width,
                   label=scenario, color=colors[i % len(colors)])

        for i, key in enumerate(ordered_keys):
            if key in LCOE_EU_dict:
                ax.axhline(y=LCOE_EU_dict[key], color=colors[i % len(colors)], linestyle='--', linewidth=1)

        ax.set_ylabel(ylab)
        method_title = method.replace('_', '-')
        ax.set_title(f"{method_title}: Country LCOE Comparison Across Scenarios", fontsize=12)
        ax.set_xticks(Cpos - bar_width / 2)
        ax.set_xticklabels(sorted_countries, rotation=45, ha='right')
        ax.set_xlim([Cpos[0] - (n_scenarios / 2) * bar_width - bar_width * 1.5,
                     np.max(Cpos) + (n_scenarios / 2) * bar_width + bar_width * 1.5])

        # Legend to the right outside
        ax.legend(ncol=n_scenarios/2, loc='lower center', bbox_to_anchor=(0.5, -0.4))

        plt.tight_layout()

        if path:
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f"{method}_methodLCOE.png"), dpi=200)

        plt.close()
def plot_MethodCentricLCOE2(Dict_LCOE_C, LCOE_EU_dict, country_list, L_C_avg, path='', ylab=r'$LCOE_c$ [EUR/MWh]'):
    keysandNames = {
        'LocalGAS_Nodal_1.0': 'GAS-Nodal',
        'LocalGAS_Global_1.0': 'GAS-Global',
        'Local_NoT_1.0': 'Local-NoT',
        'Local_Local_1.0': 'Local-Local',
        'Local_Nodal_1.0': 'Local-Nodal',
        'Local_Global_1.0': 'Local-Global',
        'Global_Nodal_1.0': 'Global-Nodal',
        'Global_Global_1.0': 'Global-Global'
    }

    # Define allowed scenario names
    # allowed_scenarios = {'Global_Global_1.0', 'LocalGAS_Global_1.0', 'Local_Local_1.0'}
    allowed_scenarios = {'Global_Global_1.0', 'Local_Global_1.0', 'Local_Local_1.0'}
    color_scheme = ('#4878CF', '#6ACC64', '#D65F5F')

    load_series = pd.Series(L_C_avg, index=country_list)
    sorted_countries = load_series.sort_values(ascending=False).index.tolist()

    all_methods = list(next(iter(Dict_LCOE_C.values())).keys())
    method_data = {m: {} for m in all_methods}
    
    # Ensure FT methods fallback to 'Load' when missing
    fallback_scenarios = {'Local_Local_1.0', 'Local_NoT_1.0'}
    ft_methods = {'FT_im', 'FT_ex', 'FT_0.5'}
    
    for scenario, methods in Dict_LCOE_C.items():
        for method, values in methods.items():
            method_data.setdefault(method, {})[scenario] = values
    
        # Inject fallback FT methods if missing
        if scenario in fallback_scenarios:
            for ft_method in ft_methods:
                if ft_method not in methods:
                    method_data.setdefault(ft_method, {})[scenario] = methods['Load']
    for scenario in fallback_scenarios:
        if scenario in Dict_LCOE_C and 'Load' in Dict_LCOE_C[scenario]:
            load_val = LCOE_EU_dict.get(scenario)
            for ft_method in ft_methods:
                if scenario not in LCOE_EU_dict and ft_method in method_data and scenario in method_data[ft_method]:
                    # Set fallback EU LCOE if it will be plotted
                    LCOE_EU_dict[scenario] = load_val

    for method, scenario_vals in method_data.items():
        # Filter scenarios
        ordered_keys = [k for k in keysandNames if k in scenario_vals and k in allowed_scenarios]
        if not ordered_keys:
            continue  # Skip if no relevant scenarios

        ordered_names = [keysandNames[k] for k in ordered_keys]
        df = pd.DataFrame({keysandNames[k]: scenario_vals[k] for k in ordered_keys}, index=country_list)
        df = df.loc[sorted_countries]

        n_scenarios = df.shape[1]
        n_countries = len(df)
        bar_width = 0.15
        index = np.arange(n_countries)

        fig, ax = plt.subplots(figsize=(18, 4))
        group_offset = bar_width * n_scenarios / 2
        Cpos = index * n_scenarios * bar_width + index * bar_width - group_offset

        for i, scenario in enumerate(df.columns):
            ax.bar(Cpos + (i - n_scenarios / 2) * bar_width, df[scenario], bar_width,
                   label=scenario, color=color_scheme[i % len(color_scheme)])

        for i, key in enumerate(ordered_keys):
            if key in LCOE_EU_dict:
                ax.axhline(y=LCOE_EU_dict[key], color=color_scheme[i % len(color_scheme)],
                           linestyle='--', linewidth=2, alpha=0.7)
            

        ax.set_ylabel(ylab)
        method_title = method.replace('_', '-')
        ax.set_title(f"{method_title}: Country LCOE Comparison Across Scenarios", fontsize=12)
        ax.set_xticks(Cpos - bar_width / 2)
        ax.set_xticklabels(sorted_countries, rotation=45, ha='right')
        ax.set_xlim([Cpos[0] - (n_scenarios / 2) * bar_width - bar_width * 1.5,
                     np.max(Cpos) + (n_scenarios / 2) * bar_width + bar_width * 1.5])

        ax.legend(ncol=n_scenarios, loc='lower center', bbox_to_anchor=(0.5, -0.4))
        plt.tight_layout()

        if path:
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f"{method}_methodLCOE.png"), dpi=200)

        plt.close()

    

def Repackage_CompareLCOE(Dict, data_processor, path='ResPlotsMar/Fig6/', Small_n =False, Type='installed'):
    
    # allmethods = [k  for  k in Dict['Global_Global_1.0']['LCOE']['LCOE_C'].keys()]
    # Weights For WSD
    L_n_avg = data_processor['Load'].mean(axis=0)
    L_eu_avg = L_n_avg.sum()
    Weights_n = np.divide(L_n_avg, L_eu_avg)
    country_of_n, country_list, country_int, n_country = Country_data(data_processor)
    L_c_avg = np.zeros(n_country).astype(np.float32)
    for (i, (start,end)) in enumerate(country_int):
        L_c_avg[i] = L_n_avg[start:end].sum()
    Weights_c = np.divide(L_c_avg, L_eu_avg)

    def wsd(Weights, LCOE, LCOE_eu):
        difSq = (LCOE - LCOE_eu) ** 2
        return np.sqrt(np.multiply(Weights, difSq).sum())

    # Store results
    LCOE_C, LCOE_EU, WSD_N, WSD_C = {}, {}, {}, {}
    LCOE_C_T, LCOE_EU_T, WSD_N_T, WSD_C_T = {}, {}, {}, {}
    
    for key, item in Dict.items():
        LCOE_c, WSD_n, WSD_c = {}, {}, {}
        LCOE_c_T, WSD_n_T, WSD_c_T = {}, {}, {}
        LCOE_EU[key] = item['LCOE'][Type]['LCOE_EU']['LCOE']
        LCOE_EU_T[key] = item['LCOE'][Type]['LCOE_EU']['LCOE_trans']

        for method, res in item['LCOE'][Type]['LCOE_C'].items():
            # print(method)
            temp = res['LCOE']
            temp_T = temp_T = item['LCOE'][Type]['LCOE_C'][method]['LCOE_trans']
            if temp.ndim == 2:
                temp = temp[0]
            if temp_T.ndim == 2:
                temp_T = temp_T[0]
            LCOE_c[method] = temp
            LCOE_c_T[method] = temp_T
            
            # WSD
            if not Small_n:
                LCOE_n = item['LCOE'][Type]['LCOE_n'][method]['LCOE']
                WSD_n[method] = wsd(Weights_n, LCOE_n, LCOE_EU[key])
                LCOE_n_T = item['LCOE'][Type]['LCOE_n'][method]['LCOE_trans']
                WSD_n_T[method] = wsd(Weights_n, LCOE_n_T, LCOE_EU_T[key])
            WSD_c[method] = wsd(Weights_c, LCOE_c[method], LCOE_EU[key])
            # WSD Transmission
            WSD_c_T[method] = wsd(Weights_c, LCOE_c_T[method], LCOE_EU_T[key])
        
        LCOE_C[key] = LCOE_c
        if not Small_n:
            WSD_N[key] = WSD_n
            WSD_N_T[key] = WSD_n_T
        WSD_C[key] = WSD_c
        LCOE_C_T[key] = LCOE_c_T
        WSD_C_T[key] = WSD_c_T

        # Plotting
        figname = key + "_countryLCOE"
        Title = key + " Comparison Of Total Country LCOE for Different Methods Of Transmision Cost Distribution"
        plot_CompareLCOE(pd.DataFrame(LCOE_c, index=country_list),
                         LCOE_EU[key], country_list, L_c_avg, Title=Title,
                         path=path+'Total/', figname=figname)
        
        # Plotting Transmission
        if Type=='installed':
            figname = key + "_countryLCOE_T"
            Title = key + " Comparison Of Country Transmision LCOE for Different Methods Of Cost Distribution"
            plot_CompareLCOE(pd.DataFrame(LCOE_c_T, index=country_list),
                             LCOE_EU_T[key], country_list, L_c_avg, Title=Title,
                             path=path+'Transmission/', figname=figname, ylab=r'$LCOE_c^T$ [EUR/MWh]')
    if Type=='installed':
        ylab=r'$LCOE_c$ [EUR/MWh]'
    else:
        ylab=r'$\widetilde{LCOE}_c$ [EUR/MWh]'
        
    plot_MethodCentricLCOE(
    LCOE_C,               # Dict[scenario][method] = values
    LCOE_EU,              # Dict[scenario] = EU average
    country_list,
    L_c_avg,
    path=path + 'ByMethod/'+Type+'/',
    ylab=ylab)
    plot_MethodCentricLCOE2(
    LCOE_C,               # Dict[scenario][method] = values
    LCOE_EU,              # Dict[scenario] = EU average
    country_list,
    L_c_avg,
    path=path + 'ByMethod2/'+Type+'/',
    ylab=ylab)

    # === Create Table === #
    def print_grouped_latex_table(results_df, all_methods, variable='LCOE_{EU}', is_node_level=True):
        # Build multicolumn header
        top_header = "Scenario & " + variable
        for method in all_methods:
            top_header += f" & {f'WSDn' if is_node_level else 'WSDc'} {method}"
        top_header += " \\\\"
    
        # Use LaTeX tabular format from DataFrame (skip default header)
        body = results_df.reset_index().to_latex(index=False, header=False, float_format="%.2f")
    
        # Combine with our custom header
        body_lines = body.splitlines()
        latex_table = "\n".join(body_lines[:3])  # \begin{tabular}, \toprule, column names
        latex_table += "\n" + top_header
        latex_table += "\n" + "\n".join(body_lines[3:])  # data rows + bottomrule
    
        print("\nGrouped LaTeX Table:\n")
        print(latex_table)
    def CreateTable(Dict, WSD, variable, path, is_node_level=True):
        all_methods = list(next(iter(WSD.values())).keys())
        table_data = []
    
        for key in Dict.keys():
            if  variable == 'LCOE_N':
                row = [LCOE_EU[key]]
            if  variable == 'LCOE_N_T':
                row = [LCOE_EU_T[key]]
            for method in all_methods:
                try:
                    val = WSD[key][method]
                    row.append(round(val, 2))
                except KeyError:
                    row.append(np.nan)
            table_data.append(row)
    
        columns = [variable] + [f"WSD{'n' if is_node_level else 'c'}_{method}" for method in all_methods]
        results_df = pd.DataFrame(table_data, index=Dict.keys(), columns=columns)
    
        os.makedirs(path, exist_ok=True)
        fname = 'WSDn_table.csv' if is_node_level else 'WSDc_table.csv'
        if variable == 'LCOE_N_T':
            fname = 'WSDn_Transmission_table.csv' if is_node_level else 'WSDc_Transmission_table.csv'
        results_df.to_csv(os.path.join(path, fname))
        print_grouped_latex_table(results_df, all_methods, variable=variable, is_node_level=is_node_level)
        return results_df
   
    if not Small_n:
        results_df_n = CreateTable(Dict, WSD_N, 'LCOE_N', path, is_node_level=True)
        results_df_C = CreateTable(Dict, WSD_C, 'LCOE_N', path, is_node_level=False)
        if Type=='installed':
            results_df_n_T = CreateTable(Dict, WSD_N_T, 'LCOE_N_T', path, is_node_level=True)
            results_df_C_T = CreateTable(Dict, WSD_C_T, 'LCOE_N_T', path, is_node_level=False)
            return results_df_n, results_df_C, results_df_n_T, results_df_C_T
        else:
            return results_df_n, results_df_C
    if Small_n:
        results_df_C = CreateTable(Dict, WSD_C, 'LCOE_N', path, is_node_level=False)
        if Type=='installed':
            results_df_C_T = CreateTable(Dict, WSD_C_T, 'LCOE_N_T', path, is_node_level=False)
            return results_df_C, results_df_C_T
        else:
            return results_df_C
       

    
        
def Country_data(data):
    #########################
    #Input data for countries
    #########################
    
    
    country_of_n = data["BusData"]['country'].to_numpy()  # Convert to NumPy array
    country_list = np.unique(country_of_n)
    country_int = []
    for country in country_list:
        start_index = np.where(country_of_n == country)[0][0]
        end_index = np.where(country_of_n == country)[0][-1]
        
        # Always append the tuple (start_index, end_index), even if they are the same
        country_int.append((start_index, end_index + 1))  # Add +1 to make inclusive slicing easier
        
    n_country = len(country_list)
    return country_of_n, country_list, country_int, n_country


#%% K_ln_T Plots
class FTPlotter():
    def __init__(self, init, path='FTPlots/', dpi =300, smalln =False):
        self.path = path
        self.init = init 
        
        data_class = data_loader()
        self.BusPositions, self.BusNames = data_class.nodal_data()
        self.group_indices, self.group_names = data_class.link_groups_by_country()
        self.CDat = data_class.Country_data()

        self.lDat = data_class.link_data()
        self.dpi=dpi
        self.smalln =smalln
       
        
    def smallngroups(self, C_layout, k_ln_T, Title='', figname='K_ln_T', subpath=''):
        kappa_l_T = C_layout['kappa']['trans']
        length = kappa_l_T['length']
        kappa_l_T=kappa_l_T['power']

        lDat =  C_layout['link_data']
        country_list = self.CDat["country_list"]
        graph = nx.from_numpy_array(lDat['adjacency'], create_using=nx.Graph(), nodelist= country_list)
        edge_list = list(graph.edges())
        
        # data = []
        # for i, (u, v) in enumerate(edge_list):
        #     # k_ln_sum = np.sum(k_ln_T[:,i], axis=1)
        #     norm_factor = np.sum(kappa_l_T[i])
        #     if norm_factor > 0:
        #         row = np.array(k_ln_T) / norm_factor
        #     else:
        #         row = np.zeros(len(country_list))
        #     data.append(row)

        data = np.nan_to_num(np.array(k_ln_T) / kappa_l_T, nan=(0)).T
        
        # Sort rows (link groups) by total summed value
        kappa_l_T2 = kappa_l_T*length
        row_sort_idx = np.argsort(kappa_l_T2)[::-1]
        data = data[row_sort_idx, :]
        link_labels =[]
        for i, (u, v) in enumerate(edge_list):
            link_labels.append( '-'.join(sorted([u, v])))
        link_labels = [link_labels[i] for i in row_sort_idx]
        
        # Sort columns (countries) by nodal load
        country_loads = C_layout['Load_C'].sum(axis=0)
        col_sort_idx = np.argsort(country_loads)[::-1]
        data = data[:, col_sort_idx]
        country_list = [country_list[i] for i in col_sort_idx]
        
        kappa_labels = [f'{kappa_l_T2[i]/1000000:.2f} [TW km]' for i in row_sort_idx]
        
        
            
        self.HeatMap(data, link_labels, subpath, figname, kappa_labels, Title, country_list)
   
    
    def CountryUsage(self, k_ln_T, kappa_l_T, Title='', figname='K_ln_T', subpath=''):
        
        country_node_indices  = self.CDat["country_int"] # Group nodes by country
        country_list = self.CDat["country_list"]
        link_groups, link_labels = self.group_indices, self.group_names
    
    
        # Aggregate over links (rows)
        data = []
        for link_group in link_groups:
            if len(link_group) == 0:
                row = np.zeros(len(country_list))
            else:
                k_ln_sum = np.sum(k_ln_T[:, link_group], axis=1)  # sum over links
                country_values = [np.sum(k_ln_sum[start:end]) for (start, end) in country_node_indices]  # sum over nodes per country
                norm_factor = np.sum(kappa_l_T[link_group])
                if norm_factor > 0:
                    row = np.array(country_values) / norm_factor
                else:
                    row = np.zeros(len(country_list))
            data.append(row)
            
        data = np.array(data)  # shape: (num_link_groups, num_countries)
        
        kappa_l_T2 = kappa_l_T*self.lDat['length']
        
        # Sort rows (link groups) by total summed value
        group_kappa_sums = [np.sum(kappa_l_T2[link_group]) for link_group in link_groups]
        # group_kappa_sums = [np.sum(kappa_l_T[link_group]) for link_group in link_groups]
        row_sort_idx = np.argsort(group_kappa_sums)[::-1]
        data = data[row_sort_idx, :]
        link_labels = [ link_labels[i] for i in row_sort_idx]
        
        # Sort columns (countries) by nodal load
        country_loads = [np.sum(self.init.Load[start:end]) for (start, end) in country_node_indices]
        col_sort_idx = np.argsort(country_loads)[::-1]
        data = data[:, col_sort_idx]
        country_list = [country_list[i] for i in col_sort_idx]
        
        kappa_labels = [f'{group_kappa_sums[i]/1000000:.2f} [TW km]' for i in row_sort_idx]
        self.HeatMap(data, link_labels, subpath, figname, kappa_labels, Title, country_list)
        
        
    def HeatMap(self, data, link_labels, subpath, figname, kappa_labels, Title, country_list):
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10.5, 14.85))
        # Create white-starting colormap
        colors = plt.cm.jet(np.linspace(0, 1, 256))
        colors[0] = np.array([1, 1, 1, 1])
        new_cmap = LinearSegmentedColormap.from_list("custom_jet", colors)
        
        im = ax.imshow(data, aspect='auto', cmap=new_cmap, vmin=0, vmax=1)
        
        # Label axes
        ax.set_yticks(range(len(link_labels)))
        ax.set_yticklabels(link_labels, fontsize=6)
    
        ax.set_xticks(range(len(country_list)))
        ax.set_xticklabels(country_list, rotation=90, fontsize=7)
        # Right-hand group-sum labels
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(len(link_labels)))
        ax2.set_yticklabels(kappa_labels, fontsize=6)
        ax2.tick_params(axis='y', which='both', length=0)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax2, pad=0.09)
        cbar.set_label(r'$\mathcal{K}^{T}_{lc} / \mathcal{K}^{T}_{l}$', fontsize=12)

        ax.set_xticks(np.arange(self.CDat['n_country'] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(link_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(Title, fontsize=14)
        plt.tight_layout() 
    
        # Save
        path = self.path + subpath
        if path and figname:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + figname + '.png', dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved as {path + figname + '.png'}")
    
        plt.close() 
   
    
    def scattermean(self, norm_flow_l, Q_lnt, cond_avg_ln, l, n, Title='',
                figname='Scattermean', subpath=''):
        """
        Description: Function to recreate Figure 3 of Tranberg et.al (2015): a 
        blue scatterplot of the coloring vector, Q_ln(F_l) for a specific node,
        n, and a specific link, l, with the value of the coloring vector on the
        y-axis, and the corresponding normalized flow-value on the x-axis.
        Along with the scatter plot a red line showing the average y-axis
        value inside a number of bins (cond_avg_ln).
        
        Parameters
        ----------
        norm_flow_l : np.array of size (len(time))
            The normalized flow of a specific link i.e. norm_flow(l,:)
        Q_lnt : np.array of size (len(time)).
            The coloring vector of a specific link and node, i.e. C_ln(l,n,:)
        cond_avg_ln : np.array of size (len(num_bins)).
            The conditional average inside each bin for a specific link and
            node, i.e. cond_avg(l,n,:).
        l : int
            The link number
        n : int
            The node number
        Title : string (default = '')
            The title of the plot
        figname : string (default = 'Scattermean')
            The savename of the plot figure
        subpath : string (default = '')
            The subfolder in to which the plot will be saved. Should end on '/'
        """
        
        #Temporary
        # l=0
        # n=0
        # norm_flow_l =self.norm_flow[:,l]
        # cond_avg_ln=cond_avg[n,l,:]
        # Q_lnt=Q_ln[l,n,:]
    
        init = self.init
        node_names = init.BusData['name'].to_numpy()
        node_name = node_names[n]
        # node_country = init.country_of_n[n]
    
        # Determine link direction and name
        linkdata = init.link_data['incidence'][:, l]
        origin_index = np.where(linkdata == -1)[0][0]
        terminal_index = np.where(linkdata == 1)[0][0]
        origin_node = node_names[origin_index]
        terminal_node = node_names[terminal_index]
        

    
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(6, 5))
    
        ax.scatter(norm_flow_l, Q_lnt, color='blue', alpha=0.3, s=10,
                   label='_nolegend_')
    
        # Plot conditional average
        num_bins = len(cond_avg_ln)
        bin_edges = np.linspace(0, 1, num_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, cond_avg_ln, color='red', linewidth=1.5,
                label='Conditional average')
    
        # Axis labels and ticks
        ax.set_xlabel(r'$f_l(t)/\kappa_l^T$', fontsize=12)
        ax.set_ylabel(r'$c_{ln}(t)$', fontsize=12)
        ax.set_title(Title, fontsize=14)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # txt1 = f"l  = '{origin_node} - {terminal_node}',"
        # txt2 = f"n = '{node_name}'"
        # ax.text(0.6, 0.95, txt1, fontsize=10)
        # ax.text(0.6, 0.91, txt2, fontsize=10)
        # ax.legend(loc=(0.6, 0.82))
        # plt.tight_layout()
        txt1 = f"l = '{origin_node} - {terminal_node}'"
        txt2 = f"n = '{node_name}'"
        ax.plot([], [], ' ', label=txt1)
        ax.plot([], [], ' ', label=txt2)

        # Add legend (with new labels included)
        ax.legend(loc='best')  # Slightly lower to fit all items

        plt.tight_layout()

        # Save plot
        path = self.path + subpath
        if path and figname:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+ figname + '.png', dpi=200,  bbox_inches='tight')
            print(f"Figure saved as {path+ figname + '.png'}")
    
        plt.close()
    



#%%

def Repackage_CompareLCOE2(Dict, data_processor, path='ResPlotsMar/Fig6/'):
    
    # allmethods = [k  for  k in Dict['Global_Global_1.0']['LCOE']['LCOE_C'].keys()]
    # Weights For WSD
    L_n_avg = data_processor['Load'].mean(axis=0)
    L_eu_avg = L_n_avg.sum()
    Weights_n = np.divide(L_n_avg, L_eu_avg)
    country_of_n, country_list, country_int, n_country = Country_data(data_processor)
    L_c_avg = np.zeros(n_country).astype(np.float32)
    for (i, (start,end)) in enumerate(country_int):
        L_c_avg[i] = L_n_avg[start:end].sum()
    Weights_c = np.divide(L_c_avg, L_eu_avg)

    def wsd(Weights, LCOE, LCOE_eu):
        difSq = (LCOE - LCOE_eu) ** 2
        return np.sqrt(np.multiply(Weights, difSq).sum())
    def print_grouped_latex_table(results_df, all_methods, variable='LCOE_{EU}', is_node_level=True):
        # Build multicolumn header
        top_header = "Scenario & " + variable
        for method in all_methods:
            top_header += f" & {f'WSDn' if is_node_level else 'WSDc'} {method}"
        top_header += " \\\\"
    
        # Use LaTeX tabular format from DataFrame (skip default header)
        body = results_df.reset_index().to_latex(index=False, header=False, float_format="%.2f")
    
        # Combine with our custom header
        body_lines = body.splitlines()
        latex_table = "\n".join(body_lines[:3])  # \begin{tabular}, \toprule, column names
        latex_table += "\n" + top_header
        latex_table += "\n" + "\n".join(body_lines[3:])  # data rows + bottomrule
    
        print("\nGrouped LaTeX Table:\n")
        print(latex_table)
    def CreateTable(Dict, WSD_C, WSD_N, variable = 'LCOE_EU'):
        # === Create Table with both WSDn and WSDc === #
        all_methods = list(next(iter(WSD_C.values())).keys())
        table_data = []
    
        for key in Dict.keys():
            row = [LCOE_EU[key]]
            for method in all_methods:
                try:
                    row.append(WSD_N[key][method])  # WSD at node level
                    row.append(WSD_C[key][method])  # WSD at country level
                except KeyError:
                    row.append(nan)  # WSD at node level
                    row.append(nan)  # WSD at country level
            table_data.append(row)
    
        # Column headers
        columns = [variable]
        for method in all_methods:
            columns.append(f'WSDn_{method}')
            columns.append(f'WSDc_{method}')
    
        results_df = pd.DataFrame(table_data, index=Dict.keys(), columns=columns)
    
        # Save CSV
        os.makedirs(path, exist_ok=True)
        if variable == 'LCOE_EU_T':
            results_df.to_csv(os.path.join(path, 'WSD_Transmission_table.csv'))
        else:
            results_df.to_csv(os.path.join(path, 'WSD_table.csv'))
        print_grouped_latex_table(results_df, all_methods, variable)
        return results_df
    results, results_T = list(), list()
    for view in ['installed', 'used']:
        # Store results
        LCOE_C, LCOE_EU, WSD_N, WSD_C = {}, {}, {}, {}
        LCOE_C_T, LCOE_EU_T, WSD_N_T, WSD_C_T = {}, {}, {}, {}
        
        for key, item in Dict.items():
            LCOE_c, WSD_n, WSD_c = {}, {}, {}
            LCOE_c_T, WSD_n_T, WSD_c_T = {}, {}, {}
            LCOE_EU[key] = item['LCOE'][view]['LCOE_EU']['LCOE']
            LCOE_EU_T[key] = item['LCOE']['installed']['LCOE_EU']['LCOE_trans']
    
            for method, res in item['LCOE'][view]['LCOE_C'].items():
                temp = res['LCOE']
                temp_T = item['LCOE'][view]['LCOE_C'][method]['LCOE_trans']
                if temp.ndim == 2:
                    temp = temp[0]
                if temp_T.ndim == 2:
                    temp_T = temp_T[0]
                LCOE_c[method] = temp
                LCOE_c_T[method] = temp_T
                
                # WSD
                LCOE_n = item['LCOE'][view]['LCOE_n'][method]['LCOE']
                WSD_n[method] = wsd(Weights_n, LCOE_n, LCOE_EU[key])
                WSD_c[method] = wsd(Weights_c, LCOE_c[method], LCOE_EU[key])
                # WSD Tranmission
                LCOE_n_T = item['LCOE']['installed']['LCOE_n'][method]['LCOE_trans']
                WSD_n_T[method] = wsd(Weights_n, LCOE_n_T, LCOE_EU_T[key])
                WSD_c_T[method] = wsd(Weights_c, LCOE_c_T[method], LCOE_EU_T[key])
            
            LCOE_C[key] = LCOE_c
            WSD_N[key] = WSD_n
            WSD_C[key] = WSD_c
            LCOE_C_T[key] = LCOE_c_T
            WSD_N_T[key] = WSD_n_T
            WSD_C_T[key] = WSD_c_T
    
            # Plotting
            if view == 'installed':
                Var = r'$LCOE_C$'
                ylab = r'$LCOE_C^T$ [EUR/MWh]'
                # Plotting Transmission
                figname = key + "_countryLCOE_T"
                Title = key + " Comparison Of Country Transmision "+Var+" for Different Methods Of Cost Distribution"
                plot_CompareLCOE(pd.DataFrame(LCOE_c_T, index=country_list),
                                 LCOE_EU_T[key], country_list, L_c_avg, Title=Title,
                                 path=path+'Transmission/', figname=figname, ylab=ylab)
            else:
                Var = r'$\widetilde{LCOE}_C$'
                ylab = r'$\widetilde{LCOE}_C^T$ [EUR/MWh]'
            
            figname = key + "_countryLCOE_"+view
            Title = key + " Comparison Of Total "+Var+" for Different Methods Of Transmision Cost Distribution"
            plot_CompareLCOE(pd.DataFrame(LCOE_c, index=country_list),
                             LCOE_EU[key], country_list, L_c_avg, Title=Title,
                             path=path+'Total/', figname=figname, ylab=Var)
            
            

        #=== Create Table === #
        results_df = CreateTable(Dict, WSD_C, WSD_N)
        results_df_T = CreateTable(Dict, WSD_C_T, WSD_N_T, variable = 'LCOE_EU_T')
        results.append(results_df)
        results_T.append(results_df_T)
    return results, results_T







def Repackage_forCspecialsStacks2(results, data_processor, path='ResPlotsApr/Fig7/'):
    Tech = ['LCOE_wind', 'LCOE_solar', 'LCOE_backup', 'LCOE_backup_energy']
    for view in ['installed', 'used']:
        # Base generation LCOE per country
        Base_LCOE_dict = {}
        for T in Tech:
            Base_LCOE_dict[T] = np.array(results['LCOE'][view]['LCOE_C']['Half'][T])
    
        # Transmission LCOE
        LCOE_T = {}
        for method, res in results['LCOE']['installed']['LCOE_C'].items():
            temp = res['LCOE_trans']
            if temp.ndim == 2:
                temp = temp[0]
            LCOE_T[method] = temp
        LCOE_T = pd.DataFrame(LCOE_T)
    
        # EU-average LCOE components
        EU_base_dict = {T: results['LCOE']['installed']['LCOE_EU'][T] for T in Tech}
        EU_trans = results['LCOE']['installed']['LCOE_EU']['LCOE_trans']
    
        country_labels = data_processor.country_list
        figname = data_processor.Layout_Scheme + '_' + data_processor.Balancing_scheme+ '_' +view
        Title = data_processor.Layout_Scheme + '_' + data_processor.Balancing_scheme + ' Comparison of Transmission Costs Allocation Methods'
    
        # Load-based sorting
        L_n_avg = data_processor.Load.mean(axis=0)
        L_c_avg = np.zeros(data_processor.n_country)
        # if small_n:
        #     L_c_avg
        # else:
        for i, (start, end) in enumerate(data_processor.country_int):
            print(i)
            # L_c_avg[i] = L_n_avg[start:end].sum()

        if view == 'installed':
            ylab = r'$LCOE_C$ [€/MWh]'
        else:
            ylab = r'$\widetilde{LCOE}_C$ [€/MWh]'
    
        plot_stacked_LCOE(Base_LCOE_dict, LCOE_T, country_labels, EU_base_dict, EU_trans,
                          L_c_avg=L_c_avg, Title=Title, path=path, figname=figname, ylab=ylab)

def Comparemethods(Dicts, DictB, data_processor, path=''):
    
    # allmethods = [k  for  k in Dict['Global_Global_1.0']['LCOE']['LCOE_C'].keys()]
    # Weights For WSD
    L_n_avg = data_processor['Load'].mean(axis=0)
    L_eu_avg = L_n_avg.sum()
    country_of_n, country_list, country_int, n_country = Country_data(data_processor)
    L_c_avg = np.zeros(n_country).astype(np.float32)
    for (i, (start,end)) in enumerate(country_int):
        L_c_avg[i] = L_n_avg[start:end].sum()
        
    load_series = pd.Series(L_c_avg, index=country_list)



    LCOE_C, LCOE_EU = {}, {}
    LCOE_C_T, LCOE_EU_T = {}, {}
    LCOE_C2, LCOE_C_T2 = {}, {}
    
    for key, item in Dicts.items():
        item2 = DictB[key]
        LCOE_c = {}
        LCOE_c_T = {}
        LCOE_c2 = {}
        LCOE_c_T2 = {}
        LCOE_EU[key] = item['LCOE']['used']['LCOE_EU']['LCOE']
        LCOE_EU_T[key] = item['LCOE']['installed']['LCOE_EU']['LCOE_trans']

        for method, res in item['LCOE']['used']['LCOE_C'].items():
            res2 = item2['LCOE']['used']['LCOE_C'][method]
            Title = key + " LCOE Over Load for "+method
            temp = res['LCOE']
            temp_T = item['LCOE']['used']['LCOE_C'][method]['LCOE_trans']
            temp2 = res2['LCOE']
            temp_T2 = DictB[key]['LCOE']['used']['LCOE_C'][method]['LCOE_trans']
            if temp.ndim == 2:
                temp = temp[0]
            if temp_T.ndim == 2:
                temp_T = temp_T[0]
            if temp2.ndim == 2:
                temp2 = temp2[0]
            if temp_T2.ndim == 2:
                temp_T2 = temp_T2[0]
          
            
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            colors = plt.cm.tab10.colors
            axs[0].scatter(load_series, temp, label=method, color='r', alpha=0.7)
            axs[0].scatter(load_series, temp2, label=method, color='b', alpha=0.7)
            axs[1].scatter(load_series, temp_T, label=method, color='r', alpha=0.7)
            axs[1].scatter(load_series, temp_T2, label=method, color='b', alpha=0.7)
            # Fit line

            axs[0].axhline(y=LCOE_EU[key], color='gray', linestyle=':', linewidth=1.5, label='EU Avg')
            axs[1].axhline(y=LCOE_EU_T[key], color='gray', linestyle=':', linewidth=1.5, label='EU Avg')
            # ax.set_title(Title)
            axs[0].set_xlabel("Normalized Load")
            axs[1].set_xlabel("Normalized Load")
            
            wrapped_title = "\n".join(textwrap.wrap(Title, 100))
            fig.suptitle(wrapped_title, fontsize=14, color='black')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            figname = key + method
            if path and figname:
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path + figname + '.png', dpi=200)
            plt.close()
            # ax.set_ylabel(ylab)
            # ax.legend()
        # Store result            
            LCOE_C[key] = LCOE_c
            LCOE_C_T[key] = LCOE_c_T
            LCOE_C2[key] = LCOE_c2
            LCOE_C_T2[key] = LCOE_c_T2
   

