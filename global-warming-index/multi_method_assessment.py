import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
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
periods_to_assess = ['2010-2019', '2013-2022', '2017', '2022']
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
df_AR6_assessment = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Tot', '50'): 1.06,  # AR6 3.3.1.1.2 p442 from observations
    ('Tot',  '5'): 0.88,  # AR6 3.3.1.1.2 p442 from observations
    ('Tot', '95'): 1.21,  # AR6 3.3.1.1.2 p442 from observations
    ('Ant', '50'): 1.07, # AR6 3.3.1.1.2 p442, and SPM A.1.3
    ('Ant',  '5'): 0.80,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
    ('Ant', '95'): 1.30,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
    ('GHG', '50'): 1.40,  # AR6 We introduce multi-method assessment here # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of likely range of 1.5
    ('GHG',  '5'): 1.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
    ('GHG', '95'): 2.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
    ('Nat', '50'): 0.03,  # We introduce multi-method assessment here # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of likely range of 0.0
    ('Nat',  '5'): -0.10,  # AR6 3.3.1.1.2 p442, SPM A.1.3
    ('Nat', '95'): 0.10,  # AR6 3.3.1.1.2 p442, SPM A.1.3
    ('OHF', '50'): -0.32,  # We introduce multi-method assessment here # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of likely range of -0.4
    ('OHF',  '5'): -0.80,  # AR6 3.3.1.1.2 p442, SPM A.1.3
    ('OHF', '95'): 0.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
    ('Int', '50'): 0.00,  # AR6 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of likely range of 0.0
    ('Int',  '5'): -0.20,  # AR6 3.3.1.1.2 p443, SPM A.1.3
    ('Int', '95'): 0.20,  # AR6 3.3.1.1.2 p443, SPM A.1.3
}, index=['2010-2019'])
df_AR6_assessment.columns.names = ['variable', 'percentile']
df_AR6_assessment.index.name = 'Year'

df_SR15_assessment = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.0,  # SR15 1.2.1.3
    ('Ant',  '5'): 0.8,  # SR15 1.2.1.3
    ('Ant', '95'): 1.2,  # SR15 1.2.1.3
}, index=['2017'])
df_SR15_assessment.columns.names = ['variable', 'percentile']
df_SR15_assessment.index.name = 'Year'

df_IPCC_assessment = pd.concat([df_AR6_assessment, df_SR15_assessment])

# RESULTS FROM AR6 METHODS ####################################################
# Data available from https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3_nathan/esmvaltool/diag_scripts/ipcc_ar6/fig3_8.py


df_AR6_Haustein = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.064,
    ('Ant',  '5'): 0.941,
    ('Ant', '95'): 1.222,
    ('GHG', '50'): 1.259,
    ('GHG',  '5'): 1.259,
    ('GHG', '95'): 1.259,
    ('Nat', '50'): 0.026,
    ('Nat',  '5'): 0.001,
    ('Nat', '95'): 0.069,
    ('OHF', '50'): -0.195,
    ('OHF',  '5'): -0.195,
    ('OHF', '95'): -0.195,
}, index=['2010-2019'])
df_AR6_Haustein.columns.names = ['variable', 'percentile']
df_AR6_Haustein.index.name = 'Year'

df_SR15_Haustein = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.02,  # SR15 1.2.1.3
    ('Ant',  '5'): 0.87,  # SR15 1.2.1.3
    ('Ant', '95'): 1.22,  # SR15 1.2.1.3
}, index=['2017'])
df_SR15_Haustein.columns.names = ['variable', 'percentile']
df_SR15_Haustein.index.name = 'Year'

df_IPCC_Haustein = pd.concat([df_AR6_Haustein, df_SR15_Haustein])


df_AR6_Ribes = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.03,
    ('Ant',  '5'): 0.89,
    ('Ant', '95'): 1.17,
    ('GHG', '50'): 1.44,
    ('GHG',  '5'): 1.12,
    ('GHG', '95'): 1.76,
    ('Nat', '50'): 0.06,
    ('Nat',  '5'): 0.04,
    ('Nat', '95'): 0.08,
    ('OHF', '50'): -0.40,
    ('OHF',  '5'): -0.69,
    ('OHF', '95'): -0.12,
    ('Int', '50'): -0.02,
    ('Int',  '5'): -0.18,
    ('Int', '95'): 0.14,
}, index=['2010-2019'])
df_AR6_Ribes.columns.names = ['variable', 'percentile']
df_AR6_Ribes.index.name = 'Year'
df_IPCC_Ribes = df_AR6_Ribes

df_AR6_Gillett = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.11,
    ('Ant',  '5'): 0.92,
    ('Ant', '95'): 1.30,
    ('GHG', '50'): 1.50,
    ('GHG',  '5'): 1.06, 
    ('GHG', '95'): 1.94, 
    ('Nat', '50'): 0.01,
    ('Nat',  '5'): -0.02,
    ('Nat', '95'): 0.05,
    ('OHF', '50'): -0.37,
    ('OHF',  '5'): -0.71,
    ('OHF', '95'): -0.03,
}, index=['2010-2019'])
df_AR6_Gillett.columns.names = ['variable', 'percentile']
df_AR6_Gillett.index.name = 'Year'
df_IPCC_Gillett = df_AR6_Gillett

df_AR6_Smith = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Ant', '50'): 1.066304612,
    ('Ant',  '5'): 0.823021383,
    ('Ant', '95'): 1.353390492,
    ('GHG', '50'): 1.341781251,
    ('GHG',  '5'): 0.993864648,
    ('GHG', '95'): 1.836139027,
    ('Nat', '50'): 0.073580353,
    ('Nat',  '5'): 0.04283156,
    ('Nat', '95'): 0.119195551,
    ('OHF', '50'): -0.269287921,
    ('OHF',  '5'): -0.628487091,
    ('OHF', '95'): -0.026618862,
}, index=['2010-2019'])
df_AR6_Smith.columns.names = ['variable', 'percentile']
df_AR6_Smith.index.name = 'Year'
df_IPCC_Smith = df_AR6_Smith

dict_IPCC_hl = {
    'Assessment': df_IPCC_assessment,
    'Haustein': df_IPCC_Haustein,
    'Ribes': df_IPCC_Ribes,
    'Gillett': df_IPCC_Gillett,
    'Smith': df_IPCC_Smith,
}


# Observations
df_AR6_Obs = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Obs', '50'): 1.06,  # AR6 3.3.1.1.2 p442 from observations
    ('Obs',  '5'): 0.88,  # AR6 3.3.1.1.2 p442 from observations
    ('Obs', '95'): 1.21,  # AR6 3.3.1.1.2 p442 from observations
}, index=['2010-2019'])
df_AR6_Obs.columns.names = ['variable', 'percentile']
df_AR6_Obs.index.name = 'Year'

df_update_Obs = pd.DataFrame({
    # (VARIABLE, PERCENTILE): VALUE
    ('Obs', '50'): 1.15,  # from annual updates paper section 4 
    ('Obs',  '5'): 1.00,  # from annual updates paper section 4 
    ('Obs', '95'): 1.25,  # from annual updates paper section 4 
}, index=['2013-2022'])
df_update_Obs.columns.names = ['variable', 'percentile']
df_update_Obs.index.name = 'Year'

df_All_Obs = pd.concat([df_AR6_Obs, df_update_Obs])
dict_All_Obs = {'Assessment': df_All_Obs}

# Plotting colours
var_colours = {'Tot': '#d7827e',
               'Ant': '#b4637a',
               'GHG': '#907aa9',
               'Nat': '#56949f',
               'OHF': '#ea9d34',
               'Res': '#9893a5',
               'Obs': '#797593',
               'PiC': '#cecacd'}

# period_colours = {
#     '2010-2019': '#e0def4',  # '#4a8fbb',
#     '2013-2022': '#31748f',  # '#4a8fcc',
#     '2022': '#9ccfd8',  # '#9bd6fa'
#     '2017': 'red',
#     '2022 (SR15 definition)': 'orange',
#     '2017 (SR1.5 definition)': 'green',
# }

source_markers = {
    'Haustein': 'o',
    'Walsh': 'o',
    'Ribes': 'v',
    'Gillett': 's',
    'Smith': 'D'
}

plot_folder = 'plots/assessment/'
# PLOT TIMESERIES FOR EACH METHOD ############################################
for method in dict_updates_ts.keys():
    print(f'Creating {method} Simple Plot...')
    plot_vars = ['Ant', 'GHG', 'Nat', 'OHF']
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
    gr.gwi_timeseries(ax, df_temp_Obs, df_temp_PiC, dict_updates_ts[method],
                      plot_vars, var_colours)
    ax.set_ylim(-1, 2)
    ax.set_xlim(1850, 2022)
    gr.overall_legend(fig, 'lower center', 6)
    fig.suptitle(f'{method} Timeseries Plot')
    fig.savefig(f'{plot_folder}/2_{method}_timeseries.png')

###############################################################################
# PLOT THE NULTI-METHOD TIMESERIES
###############################################################################
# tbc

###############################################################################
# PLOT THE VALIDATION PLOT
###############################################################################
print('Creating Fig 3.8 Validation Plot')
bar_plot_vars = ['Tot', 'Ant', 'GHG', 'OHF', 'Nat']
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=4)
ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 4), rowspan=1, colspan=1)

gr.Fig_3_8_validation_plot(ax2, ['Ant'], '2017',
                           dict_IPCC_hl, dict_updates_hl,
                           source_markers, var_colours)
gr.Fig_3_8_validation_plot(ax1, bar_plot_vars, '2010-2019',
                           dict_IPCC_hl, dict_updates_hl,
                           source_markers, var_colours)

# set the ax2 ylims to be equal to the ax1 ylims
ax1.set_ylim(-1.0, 2.0)
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(-0.2, 1.1)
# Hide the labels on the y axis of ax2
ax2.set_yticklabels([])

# set the y axis label
ax1.set_ylabel('Attributable change in surface temperature since 1850-1900 (°C)')

gr.overall_legend(fig, 'lower center', 5)
fig.suptitle('Validation of updates (right bar)' +
             'vs Results from IPCC (left bar)')
fig.savefig(f'{plot_folder}/3_WG1_Ch3_Validation.png')

###############################################################################
# Plot the headline SPM2-esque figure
###############################################################################
print('Creating SPM.2-esque figure')
fig = plt.figure(figsize=(12, 8))
ax0 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=1)
ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 1), rowspan=1, colspan=2)
ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 3), rowspan=1, colspan=2)
gr.Fig_SPM2_plot(ax0, ['Obs'], ['2010-2019', '2013-2022'],
                 dict_IPCC_hl, dict_All_Obs, var_colours)
gr.Fig_SPM2_plot(ax1, ['Ant', 'GHG', 'OHF', 'Nat'], ['2010-2019', '2013-2022'],
                 dict_IPCC_hl, dict_updates_hl, var_colours)
gr.Fig_SPM2_plot(ax2, ['Ant', 'GHG', 'OHF', 'Nat'], ['2017', '2022'],
                 dict_IPCC_hl, dict_updates_hl, var_colours)

# Plot the text
fig.text(ax0.get_position().x0, ax0.get_position().y1+0.08,
         'Observed Warming',
         fontsize=matplotlib.rcParams['axes.titlesize'],
         fontweight='bold'
         )
fig.text(ax0.get_position().x0, ax0.get_position().y1+0.02,
         '(a) Observed warming\naveraged over 10-years',
         ha='left',
         fontsize=matplotlib.rcParams['font.size'],
         fontweight='regular',
        #  fontstyle='italic'
)
fig.text(ax1.get_position().x0, ax1.get_position().y1+0.08,
         ('Contributions to warming expressed in terms of two IPCC definitions'),
         fontsize=matplotlib.rcParams['axes.titlesize'],
         fontweight='bold'
         )
fig.text(ax1.get_position().x0, ax1.get_position().y1+0.02,
         ('(b) Average warming across 10-years (AR6 Definition)'
         '\nassessed from attribution studies'),
         fontsize=matplotlib.rcParams['font.size'],
         fontweight='regular'
         )
fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
         ('(c) Present-day warming (SR1.5 Definition)'
          '\nassessed from attribution studies'),
         fontsize=matplotlib.rcParams['font.size'],
         fontweight='regular'
         )
# Set the grid to the back for the fig
ax0.set_axisbelow(True);
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)


ax0.set_ylabel('Attributable change in surface temperature since 1850-1900 (°C)')
ax0.set_xlim(-0.7, 1.1)
# set the ax1 ylims to be equal to the ax2 ylims    
ax1.set_ylim(-1.2, 2.0)
ax2.set_ylim(ax1.get_ylim())
ax0.set_ylim(ax1.get_ylim())
ax1.set_yticklabels([])
ax2.set_yticklabels([])
# fig.suptitle('Assessed contributions to observed warming')  # SPM2 title
fig.savefig(f'{plot_folder}/4_SPM2_Results.png')