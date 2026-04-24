import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src import graphing as gr
from src import definitions as defs

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--refresh-ancillary-data',
        dest='refresh_ancillary_data',
        action='store_true',
        help=(
            'Recompute and overwrite all files in results/ancillary that this '
            'script manages.'
        )
    )
    args = parser.parse_args()
    refresh_ancillary_data = args.refresh_ancillary_data

    ###########################################################################
    # LOAD DATA ###############################################################
    ###########################################################################

    start_pi, end_pi = 1850, 1900
    start_yr, end_yr = 1850, 2025

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
        skiprows = 0
        df_method_ts = pd.read_csv(
            f'results/{file_ts}',
            index_col=0,  header=[0, 1], skiprows=skiprows)
        df_method_hl = pd.read_csv(
            f'results/{file_hs}',
            index_col=0,  header=[0, 1], skiprows=skiprows)
        if method == 'Walsh':
            n = file_ts.split('.csv')[0].split('_')[-1]
        df_method_hl = defs.en_dash_ify(df_method_hl)

        dict_updates_hl[method] = df_method_hl
        dict_updates_ts[method] = df_method_ts

    # No Tot warming provided for ROF, so include an indicative approximation
    # as the sum of Ant and Nat warming
    dict_updates_ts['Gillett'].loc[:, ('Tot', '50')] = (
        dict_updates_ts['Gillett'].loc[:, ('Ant', '50')] +
        dict_updates_ts['Gillett'].loc[:, ('Nat', '50')]
        )

    # Select only the following variables for each method:
    # This is required in case some methods have extra variables, such as
    # full component-wise warming
    assess_vars = ['Tot', 'Ant', 'GHG', 'Nat', 'OHF']
    for method in dict_updates_ts.keys():
        _df = dict_updates_ts[method]
        _df = _df.loc[:, (assess_vars, slice(None))]
    for method in dict_updates_hl.keys():
        _df = dict_updates_hl[method]
        _df = _df.loc[:, (assess_vars, slice(None))]

    # MULTI-METHOD ASSESSMENT - AR6 STYLE - TIMESERIES ########################
    # Conclusion: no uncertainty plumes available at time of writing for ROF
    # (Gillett) method, so a multi-method timeseries is not created here.
    # Instead, we plot the individual methods separately as an indicative
    # alternative later.

    # MULTI-METHOD ASSESSMENT - AR6 STYLE - HEADLINES #########################
    # Create a list of the variables in df_Walsh_hl
    list_of_dfs = []
    periods_to_assess = ['2010\N{EN DASH}2019',
                         '2016\N{EN DASH}2025',
                         '2017',
                         '2025',
                         '2017 (SR15 definition)',
                         '2025 (SR15 definition)']
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
    unendashed_assessment = defs.un_en_dash_ify(
        dict_updates_hl['Assessment'].copy())
    unendashed_assessment.to_csv(
            'results/Assessment-Update-2025_GMST_headlines.csv')

    # OBSERVATIONS ############################################################
    # Add updated observation results from the annual updates paper section 4
    df_update_Obs_repeat = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        # 2010-2019 (2025 analysis): 1.055 [0.89-1.22] From Blair, paper Sect. 7
        # 2010-2019 (2024 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 7
        # 2010-2019 (2023 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 6
        # 2010-2019 (2022 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 4
        ('Obs', '50'): 1.055,
        ('Obs',  '5'): 0.89,  # TODO Blair checking whether this needs updating
        ('Obs', '95'): 1.22   # TODO Blair checking whether this needs updating
    }, index=['2010\N{EN DASH}2019'])
    df_update_Obs_repeat.columns.names = ['variable', 'percentile']
    df_update_Obs_repeat.index.name = 'Year'

    df_update_Obs_update = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        # 2016-2025 (2025 analysis): 1.26 [1.13-1.36] From Blair, paper Sect. 7
        # 2015-2024 (2024 analysis): 1.24 [1.11-1.35] From Blair, paper Sect. 7
        # 2014-2023 (2023 analysis): 1.19 [1.06-1.30] From Blair, paper Sect. 6
        # 2013-2022 (2022 analysis): 1.14 [1.00-1.25] From Blair, paper Sect. 4
        ('Obs', '50'): 1.26,
        ('Obs',  '5'): 1.13,
        ('Obs', '95'): 1.36,
    }, index=['2016\N{EN DASH}2025'])

    df_update_Obs_update.columns.names = ['variable', 'percentile']
    df_update_Obs_update.index.name = 'Year'

    # Add observations quoted from AR6
    df_AR6_Obs = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Obs', '50'): 1.06,  # AR6 3.3.1.1.2 p442 from observations
        ('Obs',  '5'): 0.88,  # AR6 3.3.1.1.2 p442 from observations
        ('Obs', '95'): 1.21,  # AR6 3.3.1.1.2 p442 from observations
    }, index=['2010\N{EN DASH}2019'])
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
    }, index=['2010\N{EN DASH}2019'])
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
    unendashed_assessment = defs.un_en_dash_ify(df_IPCC_assessment.copy())
    unendashed_assessment.to_csv('results/Assessment-6thIPCC_headlines.csv')

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
    }, index=['2010\N{EN DASH}2019'])
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
    }, index=['2010\N{EN DASH}2019'])
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
    }, index=['2010\N{EN DASH}2019'])
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
    }, index=['2010\N{EN DASH}2019'])
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
    var_names = defs.VAR_NAMES.copy()

    # Colour-code sub-variables using the corresponding aggregate category.
    flatten_sub_vars = {
        sub: parent
        for parent, children in defs.SUB_VAR_MAPPING.items()
        for sub in children
    }
    for sub_var, parent_var in flatten_sub_vars.items():
        # Preserve original aggregate colours; only assign inherited colours
        # to variables that do not already have an explicit palette entry.
        if (sub_var not in var_colours) and (parent_var in var_colours):
            var_colours[sub_var] = var_colours[parent_var]

    def get_plot_colour(var):
        if var in var_colours:
            return var_colours[var]
        mapped = defs.map_var_to_regression_aggregate(
            [var], regress_vars=['GHG', 'OHF', 'Nat']
        ).get(var, var)
        return var_colours.get(mapped, var_colours['Res'])

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
    main_plot_vars = ['Ant', 'GHG', 'Nat', 'OHF']
    for method in dict_updates_ts.keys():
        print(f'Creating {method} Simple Plot...')
        df_method_ts = dict_updates_ts[method]
        all_data_vars = (
            df_method_ts.columns.get_level_values(0).unique().to_list()
        )
        all_plot_vars = [v for v in all_data_vars if v != 'Obs']
        plume_vars = [v for v in ['Tot', 'Ant', 'GHG', 'OHF', 'Nat', 'Res']
                      if v in all_data_vars]

        plot_configs = [
            {
                'name': 'main',
                'vars': [v for v in main_plot_vars if v in all_data_vars],
                'suffix': '',
                'legend_loc': 'lower center',
                'legend_ncol': 6,
                'linestyle': 'solid',
            },
            {
                'name': 'all-vars',
                'vars': all_plot_vars,
                'suffix': '_all-vars',
                'legend_loc': 'center right',
                'legend_ncol': 1,
                'linestyle': gr.get_dynamic_linestyles(all_plot_vars),
            },
        ]

        for cfg in plot_configs:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1,
                                  colspan=1)

            if cfg['name'] == 'all-vars':
                label_map = gr.get_subvariable_indented_labels(var_names)
                label_map.update({
                    'Obs': 'Reference temperatures: HadCRUT5'
                })
                legend_order_vars = gr.get_full_variable_legend_order(
                    cfg['vars'] + ['Obs']
                )
            else:
                label_map = var_names
                legend_order_vars = None

            gr.gwi_timeseries(
                ax, df_temp_Obs, None, df_method_ts,
                cfg['vars'],
                {var: get_plot_colour(var) for var in cfg['vars'] + ['Obs']},
                sigmas=['5', '95', '50'],
                labels=label_map,
                linestyle=cfg['linestyle'],
                plume_vars=plume_vars,
            )

            ax.set_ylim(-1, 2)
            ax.set_xlim(start_yr, end_yr)
            ax.text(1875, -0.85, '1850\N{EN DASH}1900\nPreindustrial Baseline',
                    ha='center')

            if cfg['name'] == 'all-vars':
                label_to_var = {v: k for k, v in label_map.items()}
                gr.overall_legend(
                    fig, cfg['legend_loc'], cfg['legend_ncol'],
                    reorder=gr.get_legend_reorder_indices(
                        fig,
                        label_to_var=label_to_var,
                        ordered_vars=legend_order_vars
                    )
                )
                fig.tight_layout(rect=(0.02, 0.02, 0.74, 0.96))
                fig.suptitle(f'{method} Timeseries Plot (All Variables)')
            else:
                gr.overall_legend(fig, cfg['legend_loc'], cfg['legend_ncol'])
                fig.suptitle(f'{method} Timeseries Plot')

            fig.savefig(
                f'{plot_folder}/2_{method}_timeseries{cfg["suffix"]}.png'
            )
            fig.savefig(
                f'{plot_folder}/2_{method}_timeseries{cfg["suffix"]}.pdf'
            )

    plot_vars = main_plot_vars

    # PLOT THE MULTI-METHOD TIMESERIES IN SINGLE FIGURE #######################
    print('Creating Multi-Method Stacked Plot...')
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

    # Plot simplified (5-95% only) plumes for GWI method.
    gr.gwi_timeseries(ax, df_temp_Obs, None, dict_updates_ts['Walsh'],
                      ['Ant', 'GHG', 'Nat', 'OHF'],
                      var_colours, sigmas=['5', '95', '50'],
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
    ax.text(1875, -0.85, '1850\N{EN DASH}1900\nPreindustrial Baseline',
            ha='center')
    fig.suptitle('Timeseries for each attribution method used '
                 'in the assessment of contributions to observed warming')
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    gr.overall_legend(fig, 'lower center', 3,
                      reorder=[0, 1, 2, 3, 4, 5, 6, 7])
    fig.savefig(f'{plot_folder}/2_stacked-multi_method_timeseries.png')
    fig.savefig(f'{plot_folder}/2_stacked-multi_method_timeseries.pdf')

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
    fig.savefig(f'{plot_folder}/2_aligned-multi_method_timeseries.pdf')

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
    gr.Fig_3_8_validation_plot(ax1, bar_plot_vars, '2010\N{EN DASH}2019',
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
                   'since 1850\N{EN DASH}1900 (°C)')

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
             ('(a) 2010\N{EN DASH}2019 AR6 WG1 Ch.3 (left)\n' +
              '      vs 2010\N{EN DASH}2019 repeat (right)'),
             ha='left', fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='regular',
             #  fontstyle='italic'
             )
    fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
             '(b) 2017 SR1.5 Ch.1 (left)\n      vs 2017 repeat (right)',
             ha='left', fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='regular',
             #  fontstyle='italic'
             )
    fig.savefig(f'{plot_folder}/3_WG1_Ch3_Validation.png')
    fig.savefig(f'{plot_folder}/3_WG1_Ch3_Validation.pdf')

    # Plot the headline SPM2-esque figure #####################################
    print('Creating SPM.2-esque figure')
    text_toggle = True
    fig = plt.figure(figsize=(12, 10))
    ax0 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 1), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 3), rowspan=1, colspan=2)
    gr.Fig_SPM2_plot(
        ax0, ['Obs'], ['2010\N{EN DASH}2019', '2016\N{EN DASH}2025'],
        dict_IPCC_hl, dict_updates_Obs_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax1,
        ['Ant', 'GHG', 'OHF', 'Nat'],
        ['2010\N{EN DASH}2019', '2016\N{EN DASH}2025'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax2,
        ['Ant', 'GHG', 'OHF', 'Nat'],
        ['2017 (SR15 definition)', '2025 (SR15 definition)'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)

    # Set the grid to the back for the fig
    ax0.set_axisbelow(True)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    ax0.set_ylabel('Attributable change in global mean surface temperature '
                   'since 1850\N{EN DASH}1900 (°C)')
    ax0.set_xlim(-0.7, 1.1)
    ax1.set_ylim(-1.0 - text_toggle * 0.5, 2.0)
    ax2.set_ylim(ax1.get_ylim())
    ax0.set_ylim(ax1.get_ylim())
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])

    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.88))

    # Add text
    fig.text(ax0.get_position().x0, ax0.get_position().y1+0.08,
             'Observed Warming',
             fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='bold',
             )
    fig.text(ax0.get_position().x0, ax0.get_position().y1+0.02,
             '(a) Decade-average warming\n      given by observations',
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
             '\n      assessed from attribution studies'),
             fontsize=matplotlib.rcParams['font.size'],
             fontweight='regular'
             )
    fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
             ('(c) SR1.5 Update: Present-day warming contributions'
             '\n      assessed from attribution studies'),
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
    fig.savefig(f'{plot_folder}/4_SPM2_Results.pdf')

    # Plot the headline SPM2-esque figure without IPCC comparisons ############
    print('Creating SPM.2-esque figure without IPCC comparison bars')
    text_toggle = False
    fig = plt.figure(figsize=(12, 10))
    ax0 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 1), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 3), rowspan=1, colspan=2)
    gr.Fig_SPM2_plot(
        ax0, ['Obs'], ['2016\N{EN DASH}2025'],
        dict_IPCC_hl, dict_updates_Obs_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax1,
        ['Ant', 'GHG', 'OHF', 'Nat'],
        ['2016\N{EN DASH}2025'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax2,
        ['Ant', 'GHG', 'OHF', 'Nat'],
        ['2025 (SR15 definition)'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)

    # Set the grid to the back for the fig
    ax0.set_axisbelow(True)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    ax0.set_ylabel('Attributable change in global mean surface temperature '
                   'since 1850\N{EN DASH}1900 (°C)')
    ax0.set_xlim(-1, 1)
    ax1.set_xlim(-0.5, 3.5)
    ax2.set_xlim(ax1.get_xlim())
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
             '(a) Decade-average warming'
             '\n      '
             'given by observations',
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
             ('(b) AR6 Update: 2016\N{EN DASH}2025 decade-average warming'
              '\n      '
              'contributions assessed from attribution studies'),
             fontsize=matplotlib.rcParams['font.size'],
             fontweight='regular'
             )
    fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
             ('(c) SR1.5 Update: 2025 present-day warming'
              '\n      '
              'contributions assessed from attribution studies'),
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

    fig.savefig(f'{plot_folder}/4_SPM2_Results_Updates-Only.png')
    fig.savefig(f'{plot_folder}/4_SPM2_Results_Updates-Only.pdf')

    # CREATE APPENDIX-LAYOUT ABLES FOR RESULTS ################################
    if not os.path.exists('./results/ancillary'):
        os.makedirs('./results/ancillary')
    # 1. Table for all methods
    print('Creating tables for appendix')
    # Check if the table file already exists and remove it
    table_gmst_path = './results/ancillary/Table_GMST_all_methods.csv'
    write_table_gmst = (
        refresh_ancillary_data or (not os.path.exists(table_gmst_path))
    )
    if write_table_gmst:
        if refresh_ancillary_data and os.path.exists(table_gmst_path):
            os.remove(table_gmst_path)
        with open(table_gmst_path, 'w+') as f:
            times = ['2010\N{EN DASH}2019', '2016\N{EN DASH}2025',
                     '2017', '2025',
                     '2017 (SR15 definition)', '2025 (SR15 definition)']
            f.write('variable, method, ' + ', '.join(times) + '\n')
            for v in ['Ant', 'GHG', 'OHF', 'Nat']:
                for m in ['Walsh', 'Ribes', 'Gillett', 'Assessment']:
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
    else:
        print('Reusing existing ancillary table:', table_gmst_path)

    # 2. Table for ROF GSAT only
    # Load the Gillet dataset called results/Gillett_GSAT_headlines.csv to
    # pandas dataframe
    Gillet_GSAT = pd.read_csv(
            'results/Gillett_GSAT_headlines.csv',
            index_col=0,  header=[0, 1], skiprows=skiprows)
    Gillet_GSAT = defs.en_dash_ify(Gillet_GSAT)

    table_gsat_path = './results/ancillary/Table_GSAT_ROF_method.csv'
    write_table_gsat = (
        refresh_ancillary_data or (not os.path.exists(table_gsat_path))
    )
    if write_table_gsat:
        if refresh_ancillary_data and os.path.exists(table_gsat_path):
            os.remove(table_gsat_path)
        with open(table_gsat_path, 'w+') as f:
            times = ['2010\N{EN DASH}2019', '2016\N{EN DASH}2025',
                     '2017 (SR15 definition)', '2025 (SR15 definition)']
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
    else:
        print('Reusing existing ancillary table:', table_gsat_path)

    ###########################################################################
    # PLOT THE RATES ##########################################################
    ###########################################################################
    print('Creating rate plots...')
    sigmas = [[17, 83], [5, 95]]
    sigmas_all = list(
        np.concatenate((np.sort(np.ravel(sigmas)), [50]), axis=0)
        )
    main_rate_vars = ['Ant', 'GHG', 'OHF', 'Nat']
    # Toggle uncertainty shading in the all-variable rate plots.
    show_all_vars_rate_plumes = False

    def plot_rate_panel(
        ax,
        df_rates,
        plot_vars,
        linestyle_map,
        plume_vars,
        plume_sigma_pairs=None,
        label_map=None
    ):
        times_local = [int(y.split(' ')[0].split('-')[1]) for y in df_rates.index]

        if plume_sigma_pairs is None:
            plume_sigma_pairs = [
                (str(sigmas_all[s]), str(sigmas_all[-(s+2)]))
                for s in range(len(sigmas_all)//2)
            ]

        reg_map = defs.map_var_to_regression_aggregate(
            plot_vars, regress_vars=['GHG', 'OHF', 'Nat']
        )
        for var in plot_vars:
            if (var, '50') not in df_rates.columns:
                continue

            colour_var = var_colours.get(var)
            if colour_var is None:
                colour_var = var_colours.get(reg_map.get(var, var), '#444444')

            ax.plot(
                times_local,
                df_rates[(var, '50')] * 10,
                label=(label_map or var_names).get(var, var),
                color=colour_var,
                linestyle=linestyle_map.get(var, 'solid')
            )

            if var in plume_vars:
                for low, high in plume_sigma_pairs:
                    if ((var, low) in df_rates.columns
                            and (var, high) in df_rates.columns):
                        ax.fill_between(
                            times_local,
                            df_rates[(var, low)] * 10,
                            df_rates[(var, high)] * 10,
                            alpha=0.2,
                            color=colour_var,
                            linewidth=0.0
                        )

        return times_local

    def full_erf_schema_valid(df_erf_rates):
        all_vars = set(df_erf_rates.columns.get_level_values(0).unique())
        required_aggregates = {'GHG', 'OHF', 'Nat', 'Ant', 'Tot'}
        expected_subs = {
            sub_var
            for parent, children in defs.SUB_VAR_MAPPING.items()
            if parent in {'GHG', 'OHF', 'Nat'}
            for sub_var in children
        }
        return (required_aggregates.issubset(all_vars)
                and not all_vars.isdisjoint(expected_subs))

    # Plot attributed warming rates ###########################################
    # Import the rates csv file (files are a list to facilitate comparing
    # multiple samplings by plotting them atop each other).
    df_rates_walsh = pd.read_csv(
        './results/Walsh_GMST_rates.csv',
        index_col=0, header=[0, 1], skiprows=0
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
    # Check whether './results/ancillary/HadCRUT_rates_Obs.csv' exists:
    # If it does, read it in. If it doesn't, calculate the rates and save them.
    hadcrut_rate_cache = './results/ancillary/Rates_Obs_HadCRUT5.csv'
    if (not refresh_ancillary_data) and os.path.exists(hadcrut_rate_cache):
        # print('Reading existing HadCRUT rate dataset.')
        df_rates_GWI = pd.read_csv(
            hadcrut_rate_cache,
            index_col=0,  header=[0, 1], skiprows=0)
    else:
        print('Calculating HadCRUT rate dataset.')
        df_rates_GWI = defs.rate_HadCRUT5(
            start_pi, end_pi, start_yr, end_yr, sigmas_all)
        df_rates_GWI.to_csv(hadcrut_rate_cache)

    # Calculate the rates for IGCC temperature dataset ########################
    # Import IGCC temperature data
    # Check whether './results/ancillary/IGCC_rates_Obs.csv' exists:
    # If it does, read it in. If it doesn't, calculate the rates and save them.
    igcc_rate_cache = './results/ancillary/Rates_Obs_IGCC.csv'
    if (not refresh_ancillary_data) and os.path.exists(igcc_rate_cache):
        # print('Reading existing IGCC rate dataset.')
        df_Obs_IGCC = pd.read_csv(
            igcc_rate_cache,
            index_col=0,  header=[0, 1], skiprows=0)
    else:
        print('Calculating IGCC rate dataset.')
        df_Obs_IGCC = defs.load_Temp_IGCC(start_pi, end_pi, end_yr)
        df_Obs_IGCC_rate = defs.rate_IGCC(start_pi, end_pi, start_yr, end_yr)
        df_Obs_IGCC_rate.to_csv(igcc_rate_cache)

    # ax1.plot(times,  df_Obs_IGCC_rate[('Obs', '50')]*10,
    #          color='black',
    #          label='Reference Observations: IGCC',
    #          lw=2)

    rate_ticks = list(np.arange(1950, end_yr+1, 20))
    rate_ticks.append(end_yr)

    # Plot ERF Rates ##########################################################
    erf_cache_main = './results/ancillary/Rates_results_ERF.csv'
    erf_cache_full = './results/ancillary/Rates_results_ERF_full-vars.csv'

    if (not refresh_ancillary_data) and os.path.exists(erf_cache_main):
        df_forc_rates_main = pd.read_csv(
            erf_cache_main, index_col=0, header=[0, 1], skiprows=0)
    else:
        print('Calculating ERF aggregate rate dataset.')
        df_forc_rates_main = defs.rate_ERF(
            end_yr, sigmas_all, variable_mode='aggregate'
        )
        df_forc_rates_main.to_csv(erf_cache_main)

    regenerate_full_erf = refresh_ancillary_data
    if (not regenerate_full_erf) and os.path.exists(erf_cache_full):
        df_forc_rates_full = pd.read_csv(
            erf_cache_full, index_col=0, header=[0, 1], skiprows=0)
        regenerate_full_erf = not full_erf_schema_valid(df_forc_rates_full)
    elif (not regenerate_full_erf) and (not os.path.exists(erf_cache_full)):
        regenerate_full_erf = True

    if regenerate_full_erf:
        print('Calculating ERF full-variable rate dataset.')
        df_forc_rates_full = defs.rate_ERF(
            end_yr, sigmas_all, variable_mode='all'
        )
        df_forc_rates_full.to_csv(erf_cache_full)

    rate_plot_configs = [
        {
            'name': 'main',
            'suffix': '',
            'gwi_vars': [
                var for var in main_rate_vars
                if (var, '50') in df_rates_walsh.columns
            ],
            'erf_rates_df': df_forc_rates_main,
            'erf_vars': [
                var for var in main_rate_vars
                if (var, '50') in df_forc_rates_main.columns
            ],
            'legend_loc': 'lower center',
            'legend_ncol': 5,
        },
        {
            'name': 'all-vars',
            'suffix': '_all-vars',
            'gwi_vars': [
                var for var in df_rates_walsh.columns.get_level_values(0).unique()
                if var != 'Obs'
            ],
            'erf_rates_df': df_forc_rates_full,
            'erf_vars': [
                var for var in df_forc_rates_full.columns.get_level_values(0).unique()
                if var != 'Obs'
            ],
            'legend_loc': 'center right',
            'legend_ncol': 1,
        },
    ]

    for cfg in rate_plot_configs:
        fig = plt.figure(figsize=(14, 7))
        ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        if cfg['name'] == 'all-vars':
            rate_label_map = gr.get_subvariable_indented_labels(var_names)
            rate_label_map.update({
                'Obs': 'Reference temperatures: HadCRUT5'
            })
            legend_order_vars = gr.get_full_variable_legend_order(
                cfg['gwi_vars'] + cfg['erf_vars'] + ['Obs']
            )
        else:
            rate_label_map = var_names
            legend_order_vars = None

        main_plume_vars = ['Tot', 'Ant', 'GHG', 'OHF', 'Nat']
        if cfg['name'] == 'all-vars':
            if show_all_vars_rate_plumes:
                plume_sigma_pairs = [('5', '95')]
                gwi_plume_vars = [
                    v for v in main_plume_vars if v in cfg['gwi_vars']
                ]
                erf_plume_vars = [
                    v for v in main_plume_vars if v in cfg['erf_vars']
                ]
            else:
                plume_sigma_pairs = None
                gwi_plume_vars = []
                erf_plume_vars = []
        else:
            plume_sigma_pairs = None
            gwi_plume_vars = [
                v for v in ['Tot', 'Ant', 'GHG', 'OHF', 'Nat', 'Res']
                if v in cfg['gwi_vars']
            ]
            erf_plume_vars = [v for v in main_plume_vars if v in cfg['erf_vars']]
        gwi_linestyle = gr.get_dynamic_linestyles(cfg['gwi_vars'])
        erf_linestyle = gr.get_dynamic_linestyles(cfg['erf_vars'])

        times_gwi = plot_rate_panel(
            ax1,
            df_rates_walsh,
            cfg['gwi_vars'],
            gwi_linestyle,
            gwi_plume_vars,
            plume_sigma_pairs,
            rate_label_map
        )

        # Plot the observed rates
        err_pos = df_rates_GWI[('Obs', '95')] * 10 - df_rates_GWI[('Obs', '50')] * 10
        err_neg = df_rates_GWI[('Obs', '50')] * 10 - df_rates_GWI[('Obs', '5')] * 10
        obs_label = rate_label_map.get('Obs', 'Reference Observations: HadCRUT5')
        ax1.errorbar(
            times_gwi, df_rates_GWI[('Obs', '50')] * 10,
            yerr=(err_neg, err_pos),
            fmt='o', color=var_colours['Obs'], ms=2.5, lw=1,
            label=obs_label
        )

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
        ax1.set_xticks(rate_ticks)

        plot_rate_panel(
            ax2,
            cfg['erf_rates_df'],
            cfg['erf_vars'],
            erf_linestyle,
            erf_plume_vars,
            plume_sigma_pairs,
            rate_label_map
        )

        ax2.axhline(0, color='black', lw=0.5)
        ax2.set_xlim([1950, end_yr+1])
        ax2.set_ylim([-1.5, 2.5])
        ax2.set_title('(b) Effective Radiative Forcing',
                      loc='left',
                      fontweight='regular',
                      fontsize=matplotlib.rcParams['font.size'],
                      y=1.02
                      )

        ax2.set_ylabel('Decadal trend (Wm$^{-2}$decade$^{-1}$)')
        ax2.set_xlabel('End year of trend decade')
        ax2.set_xticks(rate_ticks)

        if cfg['name'] == 'all-vars':
            label_to_var = {v: k for k, v in rate_label_map.items()}
            gr.overall_legend(
                fig, cfg['legend_loc'], ncol=cfg['legend_ncol'],
                reorder=gr.get_legend_reorder_indices(
                    fig,
                    label_to_var=label_to_var,
                    ordered_vars=legend_order_vars
                )
            )
            fig.tight_layout(rect=(0.02, 0.06, 0.80, 0.90))
        else:
            gr.overall_legend(fig, cfg['legend_loc'], ncol=cfg['legend_ncol'])
            fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.90))

        fig.text(
            ax1.get_position().x0,
            ax1.get_position().y1+0.08,
            ('Decadal rates of change for contributions to ' +
             'Attributed Global Warming and Effective Radiative Forcing'),
            fontweight='bold',
            fontsize=matplotlib.rcParams['axes.titlesize'],
            )
        fig.savefig(f'{plot_folder}/5_Rates_timeseries{cfg["suffix"]}.png')
        fig.savefig(f'{plot_folder}/5_Rates_timeseries{cfg["suffix"]}.pdf')

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

    ###########################################################################
    # Plot definition diagram #################################################
    ###########################################################################
    # Load the assessment results
    df_headlines = pd.read_csv(
        "results/Assessment-Update-2025_GMST_headlines.csv",
        index_col=0,  header=[0, 1], skiprows=0
    )
    df_headlines = defs.en_dash_ify(df_headlines)

    # Plot the GWI 'Ant' 50th percentile
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    gr.GWI_definition_diagram(
        ax1, end_yr,
        dict_updates_hl['Walsh'], df_temp_Obs, dict_updates_ts['Walsh'],
        var_colours)
    ax1.set_ylabel(
        'Global mean surface temperature,\n' +
        'relative to 1850\N{EN DASH}1900 baseline (°C)'
        )

    ax1.set_ylim(0.75, 1.6)
    ticks = list(np.arange(start_yr, end_yr, 5))
    ticks.append(end_yr)
    ax1.set_xticks(ticks, ticks)
    ax1.set_yticks([1.0, 1.5])
    ax1.set_xlim(2002.5, end_yr + 1)
    # gr.overall_legend(fig, loc='lower center', ncol=4)
    # fig.suptitle(
    #     'Period Definitions for the IPCC Anthropogenic Warming Assesments',
    # )

    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.92))
    # fig.text(
    #     ax1.get_position().x0,
    #     ax1.get_position().y1+0.06,
    #     ('Period definitions for the IPCC assesments of ' +
    #      'anthropogenic global warming'),
    #     fontweight=matplotlib.rcParams['figure.titleweight'],
    #     fontsize=matplotlib.rcParams['figure.titlesize'],
    #     )

    fig.savefig(f'{plot_folder}/1_definition_diagram_GWI.png')
    fig.savefig(f'{plot_folder}/1_definition_diagram_GWI.pdf')

    ###########################################################################
    # Plot the comparison figures #############################################
    ###########################################################################
    dict_analysis_ts = {}
    compare_years = [str(end_yr), str(end_yr-1)]
    linestyles = {'Walsh':   '-', 'Ribes':   '--', 'Gillett': ':'}

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

    for method in dict_updates_ts.keys():
        dict_analysis_ts[method] = {}
        for year in compare_years:
            if year == str(end_yr):
                # If current analysis year, then just use the results already
                # loaded into the dict_updates_ts dictionary earlier in script.
                dict_analysis_ts[method][year] = dict_updates_ts[method]
            else:
                # If previous analysis year, then download csv from previous
                # GitHub release.
                directory = f'https://raw.githubusercontent.com/ClimateIndicator/anthropogenic-warming-assessment/IGCC-{year}/results/'
                file_ts = f'{method}_GMST_timeseries.csv'
                file_location = f'{directory}{file_ts}'
                df_ = defs.read_csv_with_retries(
                    file_location,
                    index_col=0,
                    header=[0, 1],
                )
                if method == 'Gillett':
                    # No Tot warming provided for ROF, so include an indicative
                    # approximation as the sum of Ant and Nat warming
                    df_.loc[:, ('Tot', '50')] = (
                        df_.loc[:, ('Ant', '50')] + df_.loc[:, ('Nat', '50')])
                dict_analysis_ts[method][year] = df_

        # Plot the ('Ant', '50') columns of timeseries against each other
        for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
            selected_years = (
                dict_analysis_ts[method][max(compare_years)].index <=
                int(min(compare_years)))
            ax.plot(
                (dict_analysis_ts[method][max(compare_years)].loc[
                     selected_years, (var, '50')] -
                 dict_analysis_ts[method][min(compare_years)][(var, '50')]
                 ),
                label=f'{method} {var}',
                color=var_colours[var],
                linestyle=linestyles[method])
    gr.overall_legend(fig, 'lower center', 3)
    fig.tight_layout(rect=[0.1, 0.15, 0.9, 0.9])
    ax.set_xlim(1850, int(min(compare_years)))
    # Set the xticks to be at 50 year intervals
    ax.set_xticks(np.append(np.arange(1850, int(min(compare_years))+1, 50),
                  int(min(compare_years))))
    ax.set_yticks(np.arange(-0.05, 0.05, 0.005))

    ax.fill_between([1850, 1900], [-5, -5], [+5, +5], color='#f4f2f1')
    ax.text(1875, -0.055, '1850\N{EN DASH}1900\nPreindustrial Baseline',
            ha='center')
    ax.set_ylim(-0.06, 0.08)
    ax.set_ylabel(
        f'{max(compare_years)} analysis minus {min(compare_years)} analysis,' +
        ' 50th percentiles, °C'
        )
    plt.suptitle('Difference between analysis years')
    plt.savefig(f'{plot_folder}/6_analysis_components_comparison.png')
    plt.savefig(f'{plot_folder}/6_analysis_components_comparison.pdf')

    # Print out the average of 1850-1900 for the 50th percentile of each variable each method and variable
    for method in sorted(dict_analysis_ts.keys()):
        print(f'Average of 1850-1900 for {method}:')
        for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
            avg_1850_1900 = dict_analysis_ts[method][max(compare_years)].loc[
                (dict_analysis_ts[method][max(compare_years)].index >= 1850) &
                (dict_analysis_ts[method][max(compare_years)].index <= 1900),
                (var, '50')
            ].mean()
            print(f'  {var}: {avg_1850_1900:.8f} °C')

    ###########################################################################
    # Multi-method timeseries to see where changes come from each year ########
    ###########################################################################
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    dict_analysis_ts['Average'] = {}
    method_names = {
        'Walsh': 'GWI',
        'Ribes': 'KCC',
        'Gillett': 'ROF',
        'Average': 'Multi-method Average'
    }
    year_colours = {
        '2025': 'xkcd:teal',
        '2024': 'xkcd:tomato',
    }
    for year in compare_years:
        # TIMESERIES
        # Create an empty timeseries
        Ant_average = dict_analysis_ts['Walsh'][year][('Ant', '50')].copy()
        Ant_average[:] = 0

        for method in methods:
            Ant_average += dict_analysis_ts[method][year][('Ant', '50')]
            ax.plot(dict_analysis_ts[method][year][('Ant', '50')],
                    year_colours[year], linestyle=linestyles[method],
                    label=method_names[method])
        Ant_average /= len(methods)
        dict_analysis_ts['Average'][year] = Ant_average

        ax.plot(Ant_average, label='Multi-method Average', color='black')
    fig.suptitle('Anthropogenic warming best estimate: ' +
                 'three attribution methods and their multi-method average')
    ax.set_ylabel('Ant 50th percentile, °C')
    ax.set_xlim(2000, end_yr+1)
    ax.set_ylim(0.6, 1.6)
    gr.overall_legend(fig, 'lower center', 4)
    fig.savefig(f'{plot_folder}/7_Compare_{"-".join(compare_years)}.png')
    fig.savefig(f'{plot_folder}/7_Compare_{"-".join(compare_years)}.pdf')

    print('\n')
    print (dict_analysis_ts['Average'].keys())

    for method in sorted(dict_analysis_ts.keys()):
        print(f'Comparison for: {method}')
        # print(dict_analysis_ts[method])
        print('  Comparing', ' and '.join(dict_analysis_ts[method].keys()), ':')

        print(f'  {compare_years[-1]} analysis gives results for year {compare_years[-1]}:', end=' ')
        if method == 'Average':
            a = dict_analysis_ts[method][compare_years[-1]][int(compare_years[-1])]
        else:
            a = dict_analysis_ts[method][compare_years[-1]].loc[int(compare_years[-1]), ('Ant', '50')]
        print(a)

        print(f'  {compare_years[-2]} analysis gives results for year {compare_years[-1]}:', end=' ')
        if method == 'Average':
            b = dict_analysis_ts[method][compare_years[-2]][int(compare_years[-1])]
        else:
            b = dict_analysis_ts[method][compare_years[-2]].loc[int(compare_years[-1]), ('Ant', '50')]
        print(b)

        print(f'  {compare_years[0]} analysis gives results for year {compare_years[0]}:', end=' ')
        if method == 'Average':
            c = dict_analysis_ts[method][compare_years[-2]][int(compare_years[-2])]
        else:
            c = dict_analysis_ts[method][compare_years[-2]].loc[int(compare_years[-2]), ('Ant', '50')]
        print(c)

        print(f'  Therefore the {compare_years[-1]} revision is: {b-a}')
        print(f'  Therefore the {compare_years[-2]} forced increase is: {c-b}')
        print(f'  Therefore the overall year-on-year change is: {c-a}')

    print('\n')

    ###########################################################################
    # Calculate linear extrapolation for next year ############################
    ###########################################################################
    extrap_times = [
        f'{end_yr}',
        f'{end_yr} (SR15 definition)',
        f'{end_yr-9}\N{EN DASH}{end_yr}'
        ]
    extrap_times_new = [
        f'{end_yr+1}',
        f'{end_yr+1} (SR15 definition)',
        f'{end_yr+1-9}\N{EN DASH}{end_yr+1}'
        ]
    extrap_var = 'Ant'
    extrap_sigmas = ['5', '50', '95']
    dict_extrap = {}

    for method in methods:
        print(f'Extrapolating {method}')
        # Obtain the 5, 95, and 50th percentile values for the 2023 year
        # for 'Ant'
        df_extrap = dict_updates_hl[method].loc[
            extrap_times, (extrap_var, extrap_sigmas)].copy()
        # print(df_extrap)
        for t in extrap_times:
            for p in extrap_sigmas:
                current = dict_updates_hl[method].loc[t, (extrap_var, p)]
                # print(f'{method} {extrap_var} {t} {p}th percentile: {current}')
                df_rate = pd.read_csv(
                    f'./results/{method}_GMST_rates.csv',
                    index_col=0, header=[0, 1], skiprows=0)
                rate = df_rate.loc[f'{end_yr-9}-{end_yr} (AR6 rate definition)',
                                   (extrap_var, p)]
                # print(f'{method} {extrap_var} {t} {p}th percentile rate: {rate}')
                extrap_result = current + rate
                # print(f'{method} {extrap_var} {t} {p}th percentile extrapolated: {extrap_result}')
                df_extrap.loc[
                    extrap_times_new[extrap_times.index(t)],
                    (extrap_var, p)] = extrap_result
        print(df_extrap)
        dict_extrap[method] = df_extrap

    # MULTI-METHOD ASSESSMENT - AR6 STYLE - HEADLINES #####################
    list_extrap_dfs = []
    extrap_assess_times = extrap_times + extrap_times_new
    # print(extrap_assess_times)
    for period in extrap_assess_times:
        # print(dict_extrap.keys())
        dict_extrap_Assessment = {}

        # Find the highest 95%, lowest 5%, and all medians, across methods
        minimum = min([dict_extrap[method].loc[period, (extrap_var, '5')]
                       for method in dict_extrap.keys()])
        maximum = max([dict_extrap[method].loc[period, (extrap_var, '95')]
                       for method in dict_extrap.keys()])
        medians = [dict_extrap[method].loc[period, (extrap_var, '50')]
                   for method in dict_extrap.keys()]

        minimum, maximum = minimum + 10, maximum + 10
        likely_min = (np.floor(minimum * 10) / 10 * np.sign(minimum))
        likely_max = (np.ceil(maximum * 10) / 10 * np.sign(maximum))
        likely_min = np.round(likely_min - 10, 1)
        likely_max = np.round(likely_max - 10, 1)
        best_est = np.round(np.mean(medians), 2)
        dict_extrap_Assessment.update(
            {(extrap_var, '50'): best_est,
                (extrap_var,  '5'): likely_min,
                (extrap_var, '95'): likely_max}
        )
        df_updates_Assessment = pd.DataFrame(
            dict_extrap_Assessment, index=[period])
        df_updates_Assessment.columns.names = ['variable', 'percentile']
        df_updates_Assessment.index.name = 'Year'
        list_extrap_dfs.append(df_updates_Assessment)
    dict_extrap['Assessment'] = pd.concat(list_extrap_dfs)
    print('Extrapolated assessment')
    print(dict_extrap['Assessment'])
    unendashed_assessment = defs.un_en_dash_ify(
        dict_extrap['Assessment'].copy())
    unendashed_assessment.to_csv(
            f'results/Assessment-Extrapolation-{end_yr+1}_GMST_headlines.csv')

    ###########################################################################
    # Create PLOT OF RAW ERFS #################################################
    ###########################################################################
    df_forc = defs.load_ERF_CMIP6(end_yr)
    # PLot the 0.05, 0.5, 0.95 quantile for each variable:
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    for var in ['GHG', 'OHF', 'Nat']:
        quantile_value = 0.5  # Change this to the quantile you want
        ax.plot(df_forc.index, df_forc[var].quantile(quantile_value, axis=1),
                label=var_names[var], color=var_colours[var])
        ax.fill_between(
            df_forc.index,
            df_forc[var].quantile(0.05, axis=1),
            df_forc[var].quantile(0.95, axis=1),
            alpha=0.2, color=var_colours[var], linewidth=0.0)
    fig.savefig(f'{plot_folder}/0_ERF_plottest.png')
    fig.savefig(f'{plot_folder}/0_ERF_plottest.pdf')
