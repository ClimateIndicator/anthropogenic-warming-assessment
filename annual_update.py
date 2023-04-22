import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymagicc

import graphing as gr
from gwi import load_HadCRUT, load_PiC_CMIP6, filter_PiControl

###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################


###############################################################################
# LOAD DATA ###################################################################
###############################################################################


start_pi, end_pi = 1850, 1900
start_yr, end_yr = 1850, 2022


# Temperature dataset
df_temp_Obs = load_HadCRUT(start_pi, end_pi)
n_yrs = df_temp_Obs.shape[0]
timeframes = [1, 3, 30]
df_temp_PiC = load_PiC_CMIP6(n_yrs, start_pi, end_pi)
df_temp_PiC = filter_PiControl(df_temp_PiC, timeframes)
df_temp_PiC.set_index(np.arange(end_yr-start_yr+1)+1850, inplace=True)


# RESULTS FROM ANNUAL UPDATES #################################################
# WALSH
files = os.listdir('results')
file_ts = [f for f in files if 'GWI_results_timeseries' in f][0]
file_hs = [f for f in files if 'GWI_results_headlines' in f][0]
df_Walsh_ts = pd.read_csv(
    f'results/{file_ts}', index_col=0,  header=[0, 1])
df_Walsh_hl = pd.read_csv(
    f'results/{file_hs}', index_col=0,  header=[0, 1])
n = file_ts.split('.csv')[0].split('_')[-1]

# RIBES
files = os.listdir('results')
file_Ribes_ts = [f for f in files if 'Ribes_results_timeseries' in f][0]
file_Ribes_hs = [f for f in files if 'Ribes_results_headlines' in f][0]
df_Ribes_ts = pd.read_csv(
    f'results/{file_Ribes_ts}', index_col=0,  header=[0, 1])
df_Ribes_hl = pd.read_csv(
    f'results/{file_Ribes_hs}', index_col=0,  header=[0, 1])

# Combine all methods into one dictionary
dict_updates_hl = {'Walsh': df_Walsh_hl,
                   'Ribes': df_Ribes_hl,
                   }
dict_updates_ts = {'Walsh': df_Walsh_ts,
                   'Ribes': df_Ribes_ts,
                    }

# MULTI-METHOD ASSESSMENT - AR6 STYLE
# Create a list of the variables in df_Walsh_hl
list_of_dfs = []
periods_to_assess = ['2010-2019', '2013-2022']
for period in periods_to_assess:
    dict_updates_Assessment = {}

    variables = ['Tot', 'Ant', 'GHG', 'OHF', 'Nat']

    for var in variables:
        # Find the highest 95%, lowest 5%, and all medians
        minimum = min([dict_updates_hl[m].loc[period, (var, '5')]
                       for m in dict_updates_hl.keys()])
        maximum = max([dict_updates_hl[m].loc[period, (var, '95')]
                       for m in dict_updates_hl.keys()])
        medians = [dict_updates_hl[m].loc[period, (var, '50')]
                   for m in dict_updates_hl.keys()]

        # Follow AR6 assessment method of best estimate being the mean of the
        # estimates for each method, and the likely range being the smallest
        # 0.1C-precision range that envelops the 5-95% range for each method

        # The simplest way to handle the multiple cases of some or all values
        # being negative is just to simply translate them all to be posisitve
        minimum, maximum = minimum + 10, maximum + 10

        likely_min = (np.floor(np.sign(minimum) * minimum * 10) / 10 *
                      np.sign(minimum))
        # round maximum value in maximum up to the highest 0.1
        likely_max = (np.ceil(np.sign(maximum) * maximum * 10) / 10 *
                      np.sign(maximum))

        # subraction loses the 0.1-precision from the above steps, so round.
        likely_min = np.round(likely_min - 10, 1)
        likely_max = np.round(likely_max - 10, 1)

        # calculate best estimate as mean across methods to 0.01 precision
        best_est = np.round(np.mean(medians), 2)

        dict_updates_Assessment.update(
            {(var, '50'): best_est,
             (var,  '5'): likely_min,
             (var, '95'): likely_max}
        )
    # Create a dataframe
    df_updates_Assessment = pd.DataFrame(
        dict_updates_Assessment, index=[period])
    df_updates_Assessment.columns.names = ['variable', 'percentile']
    df_updates_Assessment.index.name = 'Year'
    # Add it to the list
    list_of_dfs.append(df_updates_Assessment)

dict_updates_hl['Assessment'] = pd.concat(list_of_dfs)

# RESULTS FROM AR6 WG1 Ch.3 ###################################################
df_AR6_Assessment = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Tot', '50'): 1.06,  # 3.3.1.1.2 p442 from observations
    ('Tot',  '5'): 0.88,  # 3.3.1.1.2 p442 from observations
    ('Tot', '95'): 1.21,  # 3.3.1.1.2 p442 from observations
    ('Ant', '50'): 1.07,  # 3.3.1.1.2 p442, and SPM A.1.3
    ('Ant',  '5'): 0.80,  # 3.3.1.1.2 p442, and SPM A.1.3
    ('Ant', '95'): 1.30,  # 3.3.1.1.2 p442, and SPM A.1.3
    ('GHG', '50'): 1.50,  # 3.3.1.1.2 just used midpoint of likely range
    ('GHG',  '5'): 1.00,  # 3.3.1.1.2 p442, SPM A.1.3
    ('GHG', '95'): 2.00,  # 3.3.1.1.2 p442, SPM A.1.3
    ('Nat', '50'): 0.00,  # 3.3.1.1.2 just used midpoint of likely range
    ('Nat',  '5'): -0.10,  # 3.3.1.1.2 p442, SPM A.1.3
    ('Nat', '95'): 0.10,  # 3.3.1.1.2 p442, SPM A.1.3
    ('OHF', '50'): -0.4,  # 3.3.1.1.2 just used midpoint of likely range
    ('OHF',  '5'): -0.80,  # 3.3.1.1.2 p442, SPM A.1.3
    ('OHF', '95'): 0.00,  # 3.3.1.1.2 p442, SPM A.1.3
    ('PiC', '50'): 0.00,  # 3.3.1.1.2 just used midpoint of likely range
    ('PiC',  '5'): -0.20,  # 3.3.1.1.2 p443, SPM A.1.3
    ('PiC', '95'): 0.20,  # 3.3.1.1.2 p443, SPM A.1.3
}, index=['2010-2019'])
df_AR6_Assessment.columns.names = ['variable', 'percentile']
df_AR6_Assessment.index.name = 'Year'

df_AR6_Haustein = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.06,
    ('Ant',  '5'): 0.94,
    ('Ant', '95'): 1.22,
}, index=['2010-2019'])
df_AR6_Haustein.columns.names = ['variable', 'percentile']
df_AR6_Haustein.index.name = 'Year'

df_AR6_Ribes = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.03,
    ('Ant',  '5'): 0.89,
    ('Ant', '95'): 1.17,
}, index=['2010-2019'])
df_AR6_Ribes.columns.names = ['variable', 'percentile']
df_AR6_Ribes.index.name = 'Year'

df_AR6_Gillett = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.11,
    ('Ant',  '5'): 0.92,
    ('Ant', '95'): 1.30,
}, index=['2010-2019'])
df_AR6_Gillett.columns.names = ['variable', 'percentile']
df_AR6_Gillett.index.name = 'Year'

dict_AR6_hl = {
    'Assessment': df_AR6_Assessment,
    'Haustein': df_AR6_Haustein,
    'Ribes': df_AR6_Ribes,
    'Gillett': df_AR6_Gillett,
}

# Note that the central estimate for OHF isn't given; only the range is
# specified; a pixel ruler was used on the pdf to get the rough central
# value.

var_colours = {'Tot': '#d7827e',
               'Ant': '#b4637a',
               'GHG': '#907aa9',
               'Nat': '#56949f',
               'OHF': '#ea9d34',
               'Res': '#9893a5',
               'Obs': '#797593',
               'PiC': '#cecacd'}

source_colours = {
    'Walsh': '#9bd6fa',
    'IPCC AR6 WG1': '#4a8fcc',
    'Ribes': 'orange'
}
period_colours = {
    '2010-2019': '#e0def4',  # '#4a8fbb',
    '2013-2022': '#31748f',  # '#4a8fcc',
    '2022': '#9ccfd8',  # '#9bd6fa'
    '2017': 'red',
    '2022 (SR15 definition)': 'orange',
    '2017 (SR1.5 definition)': 'green',
}

source_markers = {
    'Haustein': 'o',
    'Walsh': 'o',
    'Ribes': 'v',
    'Gillett': 's',
    'Smith': 'D'
}


# PLOT TIMESERIES FOR EACH METHOD #########################################
for method in dict_updates_ts.keys():
    print(f'Creating {method} Simple Plot...')
    plot_vars = ['Ant', 'GHG', 'Nat', 'OHF']
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
    gr.gwi_timeseries(ax, df_temp_Obs, df_temp_PiC, dict_updates_ts[method],
                      plot_vars, var_colours)
    gr.overall_legend(fig, 'lower center', 6)
    fig.suptitle(f'{method} Timeseries Plot')
    fig.savefig(f'plots/2_{method}_timeseries.png')

# PLOT THE VALIDATION PLOT
print('Creating Fig 3.8 Validation Plot')
bar_plot_vars = ['Tot', 'Ant', 'GHG', 'OHF', 'Nat']
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
gr.Fig_3_8_validation_plot(ax, bar_plot_vars, dict_AR6_hl, dict_updates_hl,
                           source_markers, var_colours)
gr.overall_legend(fig, 'lower center', 4)
fig.suptitle('Validation of Methodological and Dataset Updates')
fig.savefig('plots/3_WG1_Ch3_Validation.png')