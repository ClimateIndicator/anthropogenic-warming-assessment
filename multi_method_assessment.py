import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from attribution_methods.GlobalWarmingIndex.src import graphing as gr
from attribution_methods.GlobalWarmingIndex.src import definitions as defs

###############################################################################
# Checklist of inputs to this script that need updating each year:
# 1. GWI Results (manually add csv to results/ directory)
# 2. KCC Results (manually add csv to results/ directory)
# 3. ROF Results (manually add csv to results/ directory)
# 4. Observed Warming Update (manually add in this script - get from IGCC team)
# 5. HadCRUT Observed Warming (manually add csv to
#    attribution_methods/GlobalWarmingIndex/data/Temp/HadCRUT)

# Checklist of inputs that don't need updating each year
# 1. CMIP6 PiControl simulations (update only when CMIP7 is ready)
# 2. IPCC Quoted reults (never update; they are quotes from IPCC AR6 and SR1.5)

if __name__ == '__main__':
    ###########################################################################
    # LOAD DATA ###############################################################
    ###########################################################################

    start_pi, end_pi = 1850, 1900
    start_yr, end_yr = 1850, 2023

    # Temperature dataset
    df_temp_Obs = defs.load_HadCRUT(start_pi, end_pi, start_yr, end_yr)
    n_yrs = df_temp_Obs.shape[0]
    timeframes = [1, 3, 30]
    df_temp_PiC = defs.load_PiC_CMIP6(n_yrs, start_pi, end_pi)
    df_temp_PiC = defs.filter_PiControl(df_temp_PiC, timeframes)
    df_temp_PiC.set_index(np.arange(n_yrs)+start_yr, inplace=True)

    # RESULTS FROM ANNUAL UPDATES #############################################
    # Combine dataframes of results from all attribution methods (Walsh (GWI),
    # Ribes (KCC), Gillett (ROF)) into one dictionary.
    dict_updates_hl = {}  # Headline results
    dict_updates_ts = {}  # Timeseries results
    files = os.listdir('results')  # Files in the results/ directory
    for method in ['Walsh', 'Ribes', 'Gillett']:
        file_ts = [f for f in files if f'{method}_GMST_timeseries' in f][0]
        file_hs = [f for f in files if f'{method}_GMST_headlines' in f][0]
        # skiprows = 1 if method == 'Gillett' else 0
        skiprows = 0
        df_method_ts = pd.read_csv(
            f'results/{file_ts}',
            index_col=0,  header=[0, 1], skiprows=skiprows)
        df_method_hl = pd.read_csv(
            f'results/{file_hs}',
            index_col=0,  header=[0, 1], skiprows=skiprows)
        if method == 'Walsh':
            n = file_ts.split('.csv')[0].split('_')[-1]
        dict_updates_hl[method] = df_method_hl
        dict_updates_ts[method] = df_method_ts

    # No Tot warming provided for ROF, so include an indicative approximation
    # as the sum of Ant and Nat warming
    dict_updates_ts['Gillett'].loc[:, ('Tot', '50')] = (
        dict_updates_ts['Gillett'].loc[:, ('Ant', '50')] +
        dict_updates_ts['Gillett'].loc[:, ('Nat', '50')]
        )

    # MULTI-METHOD ASSESSMENT - AR6 STYLE - TIMESERIES ########################
    # Conclusion: no uncertainty plumes available at time of writing for ROF
    # (Gillett) method, so a multi-method timeseries is not created here.
    # Instead, we plot the individual methods separately as an indicative
    # alternative later.

    # MULTI-METHOD ASSESSMENT - AR6 STYLE - HEADLINES #########################
    # Create a list of the variables in df_Walsh_hl
    list_of_dfs = []
    periods_to_assess = ['2010-2019',
                         '2014-2023',
                         '2017',
                         '2023',
                         '2017 (SR15 definition)',
                         '2023 (SR15 definition)']
    for period in periods_to_assess:
        dict_updates_Assessment = {}

        variables = ['Ant', 'GHG', 'OHF', 'Nat']

        for var in variables:
            # Find the highest 95%, lowest 5%, and all medians, across methods
            minimum = min([dict_updates_hl[method].loc[period, (var, '5')]
                           for method in dict_updates_hl.keys()])
            maximum = max([dict_updates_hl[method].loc[period, (var, '95')]
                           for method in dict_updates_hl.keys()])
            medians = [dict_updates_hl[method].loc[period, (var, '50')]
                       for method in dict_updates_hl.keys()]

            # Follow AR6 assessment method of best estimate being the
            # 0.01C-precision mean of the central estimates for each method,
            # and the likely range being the smallest 0.1C-precision range that
            # envelops the 5-95% range for each and every method.

            # Handle the multiple cases of some or all values being negative by
            # translating them all to being posisitve.
            minimum, maximum = minimum + 10, maximum + 10
            # round minimum value in minimum down to the lowest 0.1
            likely_min = (np.floor(minimum * 10) / 10 * np.sign(minimum))
            # round maximum value in maximum up to the highest 0.1
            likely_max = (np.ceil(maximum * 10) / 10 * np.sign(maximum))

            # subraction loses 0.1-precision from the above steps, so round.
            likely_min = np.round(likely_min - 10, 1)
            likely_max = np.round(likely_max - 10, 1)

            # calculate best estimate as mean across methods to 0.01 precision
            best_est = np.round(np.mean(medians), 2)
            # add dictionary of results for this variable to the dictionary for
            # the single-perdiod assessment
            dict_updates_Assessment.update(
                {(var, '50'): best_est,
                 (var,  '5'): likely_min,
                 (var, '95'): likely_max}
            )
        # Create a dataframe for assessment of this period
        df_updates_Assessment = pd.DataFrame(
            dict_updates_Assessment, index=[period])
        df_updates_Assessment.columns.names = ['variable', 'percentile']
        df_updates_Assessment.index.name = 'Year'
        # Add it to the list
        list_of_dfs.append(df_updates_Assessment)

    # Overall assessment dataframe is concatenation of dataframes for each
    # period
    dict_updates_hl['Assessment'] = pd.concat(list_of_dfs)
    dict_updates_hl['Assessment'].to_csv(
            'results/Assessment-Update-2023_GMST_headlines.csv')

    # OBSERVATIONS ############################################################
    # Add updated observation results from the annual updates paper section 4
    df_update_Obs_repeat = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        # 2010-2019 (2023 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 6
        # 2010-2019 (2022 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 4
        ('Obs', '50'): 1.07,
        ('Obs',  '5'): 0.89,
        ('Obs', '95'): 1.22
    }, index=['2010-2019'])
    df_update_Obs_repeat.columns.names = ['variable', 'percentile']
    df_update_Obs_repeat.index.name = 'Year'

    df_update_Obs_update = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        # 2014-2023 (2023 analysis): 1.19 [1.06-1.30] From Blair, paper Sect. 6
        # 2013-2022 (2022 analysis): 1.14 [1.00-1.25] From Blair, paper Sect. 4
        ('Obs', '50'): 1.19,
        ('Obs',  '5'): 1.06,
        ('Obs', '95'): 1.30,
    }, index=['2014-2023'])

    df_update_Obs_update.columns.names = ['variable', 'percentile']
    df_update_Obs_update.index.name = 'Year'

    # Add observations quoted from AR6
    df_AR6_Obs = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Obs', '50'): 1.06,  # AR6 3.3.1.1.2 p442 from observations
        ('Obs',  '5'): 0.88,  # AR6 3.3.1.1.2 p442 from observations
        ('Obs', '95'): 1.21,  # AR6 3.3.1.1.2 p442 from observations
    }, index=['2010-2019'])
    df_AR6_Obs.columns.names = ['variable', 'percentile']
    df_AR6_Obs.index.name = 'Year'

    # Combine all observations into one dictionary
    df_All_Obs = pd.concat([
                            # df_AR6_Obs,
                            df_update_Obs_repeat,
                            df_update_Obs_update
                            ])
    dict_updates_Obs_hl = {'Assessment': df_All_Obs}
    dict_IPCC_Obs_hl = {'Assessment': df_AR6_Obs}

    # QUOTED HEADLINE RESULTS FROM IPCC 6TH ASSESSMENT CYCLE ##################
    # Create dataframe of results from AR6 WG1 Ch.3
    df_AR6_assessment = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.07,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
        ('Ant',  '5'): 0.80,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
        ('Ant', '95'): 1.30,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
        ('GHG', '50'): 1.40,  # AR6 We introduce multi-method assessment here;
        # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of
        # likely range to give 1.5
        ('GHG',  '5'): 1.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('GHG', '95'): 2.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('Nat', '50'): 0.03,  # We introduce multi-method assessment here;
        # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of
        # likely range to give 0.0
        ('Nat',  '5'): -0.10,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('Nat', '95'): 0.10,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('OHF', '50'): -0.32,  # We introduce multi-method assessment here;
        # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of
        # likely range to give -0.4
        ('OHF',  '5'): -0.80,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('OHF', '95'): 0.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('Int', '50'): 0.00,  # AR6 3.3.1.1.2 had no value for this, and SPM2
        # just plotted midpoint of likely range to give 0.0, which is still
        # used here.
        ('Int',  '5'): -0.20,  # AR6 3.3.1.1.2 p443, SPM A.1.3
        ('Int', '95'): 0.20,  # AR6 3.3.1.1.2 p443, SPM A.1.3
    }, index=['2010-2019'])
    df_AR6_assessment.columns.names = ['variable', 'percentile']
    df_AR6_assessment.index.name = 'Year'

    # Create dataframe of results from SR15 Ch.1
    df_SR15_assessment = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.0,  # SR15 1.2.1.3
        ('Ant',  '5'): 0.8,  # SR15 1.2.1.3
        ('Ant', '95'): 1.2,  # SR15 1.2.1.3
    }, index=['2017'])
    df_SR15_assessment.columns.names = ['variable', 'percentile']
    df_SR15_assessment.index.name = 'Year'

    # Combine 6th assessment cycle results from both SR1.5 and AR6
    df_IPCC_assessment = pd.concat([df_AR6_assessment, df_SR15_assessment])
    df_IPCC_assessment.to_csv('results/Assessment-6thIPCC_headlines.csv')

    # QUOTED RESULTS FROM INDIVIDUAL AR6 ATTRIBUTION METHODS ##################
    # These results for each method are quoted here from the AR6 assessment.
    # Data available from https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3_nathan/esmvaltool/diag_scripts/ipcc_ar6/fig3_8.py

    # Haustein 2017 (GWI)
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

    # Ribes (KCC)
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

    # Gillet (ROF)
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

    # Smith (AR6 WGI Chapter 7)
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

    # Combine all IPCC-quoted results (not the updated results) into one
    # dictionary
    dict_IPCC_hl = {
        'Assessment': df_IPCC_assessment,
        'Haustein': df_IPCC_Haustein,
        'Ribes': df_IPCC_Ribes,
        'Gillett': df_IPCC_Gillett,
        'Smith': df_IPCC_Smith,
    }

    ###########################################################################
    # CREATE PLOTS ############################################################
    ###########################################################################

    # Plotting colours
    var_colours = {
        'Tot': '#d7827e',
        'Ant': '#b4637a',
        'Nat': '#56949f',
        'GHG': '#907aa9',
        'OHF': '#ea9d34',
        'Res': '#9893a5',
        'Obs': '#797593',
        'PiC': '#cecacd'}
    var_names = {
        'Obs': 'Observed Warming',
        'Ant': 'Total Human-induced Warming',
        'GHG': 'Well-mixed Greenhouse Gases',
        'OHF': 'Other Human Forcings',
        'Nat': 'Natural Forcings',
        }

    source_markers = {
        'Haustein': 'o',  # Walsh and Haustein are both GWI so get same symbol.
        'Walsh': 'o',
        'Ribes': 'v',
        'Gillett': 's',
        'Smith': 'D'}

    labels = {
        'Haustein': 'Global Warming Index',
        'Walsh': 'Global Warming Index',
        'Ribes': 'Kriging for Climate Change',
        'Gillett': 'Regularised Optimal Fingerprinting',
        'Smith': 'AR6 WG1 Chapter 7',
        }

    plot_folder = f'./plots/{end_yr}/'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # PLOT TIMESERIES FOR EACH METHOD #########################################
    for method in dict_updates_ts.keys():
        print(f'Creating {method} Simple Plot...')
        plot_vars = ['Ant', 'GHG', 'Nat', 'OHF']
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
        gr.gwi_timeseries(
            ax, df_temp_Obs, df_temp_PiC, dict_updates_ts[method],
            plot_vars, var_colours)
        ax.set_ylim(-1, 2)
        ax.set_xlim(start_yr, end_yr)
        ax.text(1875, -0.85, '1850-1900\nPreindustrial Baseline', ha='center')
        gr.overall_legend(fig, 'lower center', 6)
        fig.suptitle(f'{method} Timeseries Plot')
        fig.savefig(f'{plot_folder}/2_{method}_timeseries.png')
        fig.savefig(f'{plot_folder}/2_{method}_timeseries.svg')

    # PLOT THE MULTI-METHOD TIMESERIES IN SINGLE FIGURE #######################
    print('Creating Multi-Method Stacked Plot...')
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

    # Plot simplified (5-95% only) plumes for GWI method.
    gr.gwi_timeseries(ax, df_temp_Obs, df_temp_PiC, dict_updates_ts['Walsh'],
                      plot_vars, var_colours, sigmas=['5', '95', '50'],
                      labels=True)
    # Plot the median best-estimate for each method on top of the GWI plumes.
    for m, l in zip(['Walsh', 'Ribes', 'Gillett'], ['-', '--', ':']):
        for v in plot_vars:
            ax.plot(
                dict_updates_ts[m].index,
                dict_updates_ts[m].loc[:, (v, '50')].values,
                color=var_colours[v],
                ls=l, lw=2, alpha=0.7)
        # Plot arbitrary lines so that separate black lines for each method
        # appear in the legend.
        ax.plot([100, 100], [100, 100], color='black', lw=2, alpha=0.7, ls=l,
                label=f'{m}: {labels[m]}')

    ax.set_ylim(-1, 2)
    ax.set_xlim(start_yr, end_yr)
    ax.text(1875, -0.85, '1850-1900\nPreindustrial Baseline', ha='center')
    fig.suptitle('Timeseries for each attribution method used '
                 'in the assessment of contributions to observed warming')
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    gr.overall_legend(fig, 'lower center', 3,
                      reorder=[8, 0, 1, 2, 3, 4, 5, 6, 7])
    fig.savefig(f'{plot_folder}/2_stacked-multi_method_timeseries.png')
    fig.savefig(f'{plot_folder}/2_stacked-multi_method_timeseries.svg')

    # PLOT THE MULTI-METHOD TIMESERIES IN MULTI-FIGURE ########################
    print('Creating Multi-Method Aligned Plot...')
    fig = plt.figure(figsize=(16, 6))
    methods = ['Walsh', 'Ribes', 'Gillett']
    subs = ['(a)', '(b)', '(c)']

    for m in methods:
        ax = plt.subplot2grid(shape=(1, 3), loc=(0, methods.index(m)),
                              rowspan=1, colspan=1)
        PiC = df_temp_PiC if m == 'Walsh' else None
        gr.gwi_timeseries(ax, df_temp_Obs, PiC, dict_updates_ts[m],
                          ['Tot', 'GHG', 'Nat', 'OHF'], var_colours)
        ax.set_ylim(-1, 2)
        ax.set_xlim(1900, end_yr)
        if methods.index(m) > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.set_title(f'{subs[methods.index(m)]} {m}: {labels[m]}')
    gr.overall_legend(fig, 'lower center', 6)
    # fig.suptitle('Testing one two three how do we think this looks?')
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.94))
    fig.suptitle('Timeseries for each attribution method used '
                 'in the assessment of contributions to observed warming')
    fig.savefig(f'{plot_folder}/2_aligned-multi_method_timeseries.png')
    fig.savefig(f'{plot_folder}/2_aligned-multi_method_timeseries.svg')

    # PLOT THE VALIDATION PLOT ################################################
    print('Creating Fig 3.8 Validation Plot')
    bar_plot_vars = ['Ant', 'GHG', 'OHF', 'Nat']
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=4)
    ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 4), rowspan=1, colspan=1)

    gr.Fig_3_8_validation_plot(ax2, ['Ant'], '2017',
                               dict_IPCC_hl, dict_updates_hl,
                               dict_IPCC_Obs_hl, dict_updates_Obs_hl,
                               source_markers, var_colours, labels)
    gr.Fig_3_8_validation_plot(ax1, bar_plot_vars, '2010-2019',
                               dict_IPCC_hl, dict_updates_hl,
                               dict_IPCC_Obs_hl, dict_updates_Obs_hl,
                               source_markers, var_colours, labels)
    # set the ax2 ylims to be equal to the ax1 ylims
    ax1.set_ylim(-1.0, 2.0)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(-0.2, 1.1)
    # Hide the labels on the y axis of ax2
    ax2.set_yticklabels([])
    # set the y axis label
    ax1.set_ylabel('Attributable change in surface temperature '
                   'since 1850-1900 (°C)')

    # create a one datapoint at 100, 100 for each method:
    for m in sorted(labels.keys()):
        ax2.errorbar(2., 1., yerr=0.1, xerr=None,
                     label=labels[m], fmt=source_markers[m],
                     color='gray', ms=7, lw=2,
                     )
    gr.overall_legend(fig, 'lower center', 4)
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.88))

    fig.suptitle('Validation of updated lines of evidence for assessing '
                 'contributions to observed warming')
    fig.text(ax1.get_position().x0, ax1.get_position().y1+0.02,
             '(a) 2010-2019 AR6 WG1 Ch.3 (left)\nvs 2010-2019 repeat (right)',
             ha='left', fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='regular',
             #  fontstyle='italic'
             )
    fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
             '(b) 2017 SR1.5 Ch.1 (left)\nvs 2017 repeat (right)',
             ha='left', fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='regular',
             #  fontstyle='italic'
             )
    fig.savefig(f'{plot_folder}/3_WG1_Ch3_Validation.png')
    fig.savefig(f'{plot_folder}/3_WG1_Ch3_Validation.svg')

    # Plot the headline SPM2-esque figure #####################################
    print('Creating SPM.2-esque figure')
    text_toggle = False
    fig = plt.figure(figsize=(12, 10))
    ax0 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 1), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 3), rowspan=1, colspan=2)
    gr.Fig_SPM2_plot(
        ax0, ['Obs'], ['2010-2019', '2014-2023'],
        dict_IPCC_hl, dict_updates_Obs_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax1, ['Ant', 'GHG', 'OHF', 'Nat'], ['2010-2019', '2014-2023'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax2, ['Ant', 'GHG', 'OHF', 'Nat'], ['2017', '2023'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)

    # Set the grid to the back for the fig
    ax0.set_axisbelow(True)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    ax0.set_ylabel('Attributable change in global mean surface temperature '
                   'since 1850-1900 (°C)')
    ax0.set_xlim(-0.7, 1.1)
    ax1.set_ylim(-1.0 - text_toggle * 0.2, 2.0)
    ax2.set_ylim(ax1.get_ylim())
    ax0.set_ylim(ax1.get_ylim())
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])

    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.88))

    # Add text
    fig.text(ax0.get_position().x0, ax0.get_position().y1+0.08,
             'Observed Warming',
             fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='bold'
             )
    fig.text(ax0.get_position().x0, ax0.get_position().y1+0.02,
             '(a) Decade-average warming\ngiven by observations',
             ha='left',
             fontsize=matplotlib.rcParams['font.size'],
             fontweight='regular',
             #  fontstyle='italic'
             )
    fig.text(ax1.get_position().x0, ax1.get_position().y1+0.08,
             ('Contributions to observed warming '
             'expressed in terms of two IPCC warming definitions'),
             fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='bold'
             )
    fig.text(ax1.get_position().x0, ax1.get_position().y1+0.02,
             ('(b) AR6 Update: Decade-average warming contributions'
             '\nassessed from attribution studies'),
             fontsize=matplotlib.rcParams['font.size'],
             fontweight='regular'
             )
    fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
             ('(c) SR1.5 Update: Present-day warming contributions'
             '\nassessed from attribution studies'),
             fontsize=matplotlib.rcParams['font.size'],
             fontweight='regular'
             )

    # Add arrow from Other Human Forcing to Total Human-induced Warming
    for ax in [ax1, ax2]:
        # get bounds of ax1
        x0 = ax.get_position().x0
        y0 = ax.get_position().y0
        wi = ax.get_position().width
        xcoords = [[wi/8, wi/8], [3*wi/8, 3*wi/8],
                   [5*wi/8, 5*wi/8], [wi/8, 5*wi/8]]
        ycoords = [[y0-0.225, y0-0.26], [y0-0.225, y0-0.26],
                   [y0-0.171, y0-0.26], [y0-0.26, y0-0.26]]
        arrow = ['<-', ']-', ']-', '-']
        for i in range(4):
            plt.annotate('',
                         arrowprops=dict(arrowstyle=arrow[i],
                                         shrinkA=0, shrinkB=0,
                                         color='gainsboro',
                                         lw=1),
                         xy=(xcoords[i][1]+x0, ycoords[i][1]),
                         xycoords='figure fraction',
                         xytext=(xcoords[i][0]+x0, ycoords[i][0]),
                         textcoords='figure fraction'
                         )

    # fig.suptitle('Assessed contributions to observed warming')  # SPM2 title
    fig.savefig(f'{plot_folder}/4_SPM2_Results.png')
    fig.savefig(f'{plot_folder}/4_SPM2_Results.svg')

    # Create appendix-layout tables for results.
    print('Creating tables for appendix')
    with open('results/Table_GMST_all_methods.csv', 'w') as f:
        times = ['2010-2019', '2014-2023',
                 '2017', '2023',
                 '2017 (SR15 definition)', '2023 (SR15 definition)']
        f.write('variable, method, ' + ', '.join(times) + '\n')
        for v in ['Ant', 'GHG', 'OHF', 'Nat']:
            for m in ['Walsh', 'Ribes', 'Gillett', 'Assessment']:
                # print(
                #     dict_updates_hl[m]
                # )
                line = [v, m]
                if m == 'Assessment':
                    data = ["{:0.2f} ({:0.1f} to {:0.1f})".format(
                        dict_updates_hl[m].loc[t, (v, '50')],
                        dict_updates_hl[m].loc[t, (v, '5')],
                        dict_updates_hl[m].loc[t, (v, '95')]
                        )
                            for t in times]
                else:
                    data = ["{:0.2f} ({:0.2f} to {:0.2f})".format(
                        dict_updates_hl[m].loc[t, (v, '50')],
                        dict_updates_hl[m].loc[t, (v, '5')],
                        dict_updates_hl[m].loc[t, (v, '95')]
                        )
                            for t in times]

                line.extend(data)
                line = ', '.join([str(x) for x in line]) + '\n'
                f.write(line)

    # Load the Gillet dataset called results/Gillett_GSAT_headlines.csv to
    # pandas dataframe
    Gillet_GSAT = pd.read_csv(
            'results/Gillett_GSAT_headlines.csv',
            index_col=0,  header=[0, 1], skiprows=skiprows)

    with open('results/Table_GSAT_ROF_method.csv', 'w') as f:
        times = ['2010-2019', '2014-2023',
                 '2017 (SR15 definition)', '2023 (SR15 definition)']
        f.write('variable, ' + ', '.join(times) + '\n')
        for v in ['Ant', 'GHG', 'OHF', 'Nat']:
            line = [v]
            data = ["{:0.2f} ({:0.2f} to {:0.2f})".format(
                Gillet_GSAT.loc[t, (v, '50')],
                Gillet_GSAT.loc[t, (v, '5')],
                Gillet_GSAT.loc[t, (v, '95')]
                )
                    for t in times]

            line.extend(data)
            line = ', '.join([str(x) for x in line]) + '\n'
            f.write(line)

    ###########################################################################
    # PLOT THE RATES ##########################################################
    ###########################################################################
    print('Creating rate plots...')
    fig = plt.figure(figsize=(14, 7))
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)

    sigmas = [[17, 83], [5, 95]]
    sigmas_all = list(
        np.concatenate((np.sort(np.ravel(sigmas)), [50]), axis=0)
        )
    rate_vars = ['Ant', 'GHG', 'OHF', 'Nat']

    # Plot attributed warming rates ###########################################
    # Import the rates csv file (files are a list to facilitate comparing
    # multiple samplings by plotting them atop each other).
    files = ['./results/Walsh_GMST_rates.csv']

    for file in files:
        df_rates_GWI = pd.read_csv(file, index_col=0,  header=[0, 1], skiprows=0)
        times = [int(y.split(' ')[0].split('-')[1]) for y in df_rates_GWI.index]
        for var in rate_vars:
            ax1.plot(times, df_rates_GWI[(var, '50')]*10,
                     label=var_names[var], color=var_colours[var])
            for s in range(len(sigmas_all)//2):
                ax1.fill_between(
                    times,
                    df_rates_GWI[(var, str(sigmas_all[s]))]*10,
                    df_rates_GWI[(var, str(sigmas_all[-(s+2)]))]*10,
                    alpha=0.2, color=var_colours[var], linewidth=0.0
                    )
    # # plot the olf GWI trend values from Chris' plot
    # This code is copied from https://github.com/ClimateIndicator/forcing-timeseries/blob/main/notebooks/decadal-trends.ipynb
    # Read in the CSV file with the temperature data
    # GWI_df = pd.read_csv(
    #     'https://raw.githubusercontent.com/ClimateIndicator/data/v2023.05.01/'
    #     + 'data/anthropogenic_warming/Walsh_GMST_timeseries_6048000.csv')
    # GWI_df = GWI_df.rename(columns={'timebound_lower': 'Year'})
    # # Set the 'Year' column as the index
    # GWI_df['Year'] = GWI_df['Year'].astype(int)
    # GWI_df.set_index('Year', inplace=True)
    # rolling_mean_GWI = GWI_df.rolling(window=10).mean()
    # GWI_trend = (rolling_mean_GWI - rolling_mean_GWI.shift(10))
    # y_GWI = GWI_trend['anthropogenic_p50'].values
    # x_GWI = GWI_trend.index.values
    # ax1.plot(x_GWI, y_GWI, marker='+', linestyle='None',)
    # ax1.plot(x_GWI[-3:], y_GWI[-3:],
    #          marker='+', color='red', linestyle='None')

    # Plot HadCRUT5 rates #####################################################
    # Check whether './results/HadCRUT_rates_Obs.csv' exists:
    # If it does, read it in. If it doesn't, calculate the rates and save them.
    if os.path.exists('./results/Rates_Obs_HadCRUT5.csv'):
        # print('Reading existing HadCRUT rate dataset.')
        df_rates_GWI = pd.read_csv('./results/Rates_Obs_HadCRUT5.csv',
                               index_col=0,  header=[0, 1], skiprows=0)
    else:
        print('Calculating HadCRUT rate dataset.')
        df_rates_GWI = defs.rate_HadCRUT5(
            start_pi, end_pi, start_yr, end_yr, sigmas_all)
        df_rates_GWI.to_csv('./results/Rates_Obs_HadCRUT5.csv')

    # Plot the observed rates
    err_pos = df_rates_GWI[('Obs', '95')]*10 - df_rates_GWI[('Obs', '50')]*10
    err_neg = df_rates_GWI[('Obs', '50')]*10 - df_rates_GWI[('Obs', '5')]*10
    ax1.errorbar(times,  df_rates_GWI[('Obs', '50')]*10,
                 yerr=(err_neg, err_pos),
                 fmt='o', color=var_colours['Obs'], ms=2.5, lw=1,
                 label='Reference Observations: HadCRUT5')

    # # Plot the Ribes rates as scatter:
    # df_rates_Ribes = pd.read_csv(
    #     './results/Ribes_GMST_rates.csv',
    #     index_col=0,  header=[0, 1], skiprows=0)
    # times = [int(y.split(' ')[0].split('-')[1]) for y in df_rates_Ribes.index]
    # for var in rate_vars:
    #     err_pos = (df_rates_Ribes[(var, '95')] * 10 -
    #                df_rates_Ribes[(var, '50')] * 10)
    #     err_neg = (df_rates_Ribes[(var, '50')] * 10 -
    #                df_rates_Ribes[(var, '5')] * 10)
    #     ax1.errorbar(times, df_rates_Ribes[(var, '50')]*10,
    #                  yerr=(err_neg, err_pos),
    #                  # make it a square marker for the scatter
    #                  fmt='s',
    #                  color=var_colours[var], ms=5, lw=2,
    #                  label=var_names[var])

    # Add a line along the y=0 line
    ax1.axhline(0, color='black', lw=0.5)
    ax1.set_xlim([1950, end_yr+1])
    ax1.set_title('(a) Attributed Global Warming',
                  loc='left',
                  fontweight='regular',
                  fontsize=matplotlib.rcParams['font.size'],
                  y=1.02
                  )
    ax1.set_ylabel('Decadal trend (°C decade$^{-1}$)')
    ax1.set_xlabel('End year of trend decade')
    ax1.set_ylim(-0.3, 0.5)
    rate_ticks = list(np.arange(1950, end_yr+1, 20))
    rate_ticks.append(end_yr)
    ax1.set_xticks(rate_ticks)

    # Plot ERF Rates ##########################################################
    # Check whether './results/ERF_rates_results.csv' exists:
    # If it does, read it in
    # If it doesn't, calculate the rates and save them
    if os.path.exists('./results/Rates_results_ERF.csv'):
        # print('Reading existing ERF rate dataset.')
        df_forc_rates = pd.read_csv(
            './results/Rates_results_ERF.csv',
            index_col=0,  header=[0, 1], skiprows=0)
    else:
        print('Calculating ERF rate dataset.')
        # Load the ERF dataset
        df_forc_rates = defs.rate_ERF(end_yr, sigmas_all, rate_vars)
        df_forc_rates.to_csv('./results/Rates_results_ERF.csv')

    times = [int(y.split(' ')[0].split('-')[1])
             for y in df_forc_rates.index]

    for var in rate_vars:
        ax2.plot(times, df_forc_rates[(var, '50')]*10,
                 label=var_names[var], color=var_colours[var])
        for s in range(len(sigmas_all)//2):
            ax2.fill_between(
                times,
                df_forc_rates[(var, str(sigmas_all[s]))]*10,
                df_forc_rates[(var, str(sigmas_all[-(s+2)]))]*10,
                alpha=0.2, color=var_colours[var], linewidth=0.0)

    # PLOT THE 2022 results for comparison ####################################
    # This code is copied from https://github.com/ClimateIndicator/forcing-timeseries/blob/main/notebooks/decadal-trends.ipynb
    # # Read in the CSV file with the ERF data
    # erf_df = pd.read_csv(
    #     'https://raw.githubusercontent.com/ClimateIndicator/' +
    #     'forcing-timeseries/main/output/ERF_best_aggregates_1750-2023.csv')
    # erf_df = erf_df.rename(columns={'Unnamed: 0': 'Year'})
    # # Set the 'Year' column as the index
    # erf_df.set_index('Year', inplace=True)
    # rolling_mean = erf_df.rolling(window=10).mean()
    # erf_trend = (rolling_mean - rolling_mean.shift(10))
    # y=erf_trend['anthro'].values
    # x=erf_trend.index.values
    # # Plot each series on its respective axis
    # ax2.plot(x,y,marker='o', linestyle='None',label='2022 analysis')
    # ax2.plot(x[-4:], y[-4:], marker='o', color='red', linestyle='None')

    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_xlim([1950, end_yr+1])
    ax2.set_ylim([-1.5, 2.5])
    ax2.set_title('(a) Effective Radiative Forcing',
                  loc='left',
                  fontweight='regular',
                  fontsize=matplotlib.rcParams['font.size'],
                  y=1.02
                  )

    ax2.set_ylabel('Decadal trend (Wm$^{-2}$decade$^{-1}$)')
    ax2.set_xlabel('End year of trend decade')
    ax2.set_xticks(rate_ticks)

    gr.overall_legend(fig, 'lower center', ncol=5)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.90))

    fig.text(
        ax1.get_position().x0,
        ax1.get_position().y1+0.08,
        ('Decadal rates of change for contributions to ' +
         'Attributed Global Warming and Effective Radiative Forcing'),
        fontweight='bold',
        fontsize=matplotlib.rcParams['axes.titlesize'],
        )
    fig.savefig(f'{plot_folder}/5_Rates_timeseries.png')
    fig.savefig(f'{plot_folder}/5_Rates_timeseries.svg')

    ###########################################################################
    # Plot definition diagram #################################################
    ###########################################################################
    # Load the assessment results
    df_headlines = pd.read_csv(
        "results/Assessment-Update-2023_GMST_headlines.csv",
        index_col=0,  header=[0, 1], skiprows=0
    )

    # Plot the GWI 'Ant' 50th percentile
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    gr.definition_diagram(
        ax1, end_yr,
        df_headlines, df_temp_Obs, dict_updates_ts['Walsh'],
        var_colours)
    ax1.set_ylabel('Temperature anomaly, relative to 1850-1900 baseline (°C)')
    ax1.set_ylim(0.7, 1.5)
    ticks = list(np.arange(start_yr, end_yr, 5))
    ticks.append(end_yr)
    ax1.set_xticks(ticks, ticks)
    ax1.set_yticks([1.0, 1.5])
    ax1.set_xlim(2002.5, end_yr + 1)
    # gr.overall_legend(fig, loc='lower center', ncol=4)
    fig.suptitle('Anthropogenic Warming Assessment Period Definitions')
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    fig.savefig(f'{plot_folder}/1_definition_diagram.png')
    fig.savefig(f'{plot_folder}/1_definition_diagram.svg')
