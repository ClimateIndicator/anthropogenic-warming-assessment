import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src import graphing as gr
from src import definitions as defs
import data.Temp.IGCC.IGCC_Obs as IGCC_Obs_data
import data.Temp.IPCC.IPCC_Obs as IPCC_Obs_data
import data.Temp.IPCC.IPCC_AGW as IPCC_AGW_data


def parse_arguments():
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
    args.refresh_ancillary_data
    refresh_ancillary_data = args.refresh_ancillary_data
    return refresh_ancillary_data


def load_filtered_PiC(START_YR, IGCC_YR, START_PI, END_PI, n_yrs):
    timeframes = [1, 3, 30]
    df_temp_PiC = defs.load_PiC_CMIP6(n_yrs, START_PI, END_PI)
    df_temp_PiC = defs.filter_PiControl(df_temp_PiC, timeframes)
    df_temp_PiC.set_index(np.arange(n_yrs)+START_YR, inplace=True)
    return df_temp_PiC


def load_attribution_results(assess_vars):
    # Combine dataframes of results from all attribution methods (Walsh (GWI),
    # Ribes (KCC), Gillett (ROF)) into one dictionary.
    dict_updates_hl = {}  # Headline results
    dict_updates_ts = {}  # Timeseries results
    files = os.listdir('results')  # Files in the results/ directory
    for method in ['Walsh', 'Ribes', 'Gillett']:
        file_ts = [f for f in files if f'{method}_GMST_timeseries' in f][0]
        file_hs = [f for f in files if f'{method}_GMST_headlines' in f][0]
        df_method_ts = pd.read_csv(
            f'results/{file_ts}',
            index_col=0,  header=[0, 1], skiprows=0)
        df_method_hl = pd.read_csv(
            f'results/{file_hs}',
            index_col=0,  header=[0, 1], skiprows=0)

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
    for method in dict_updates_ts.keys():
        _df = dict_updates_ts[method]
        # Use get_level_values(0).isin(...) to avoid Pandas MultiIndex
        # FutureWarning when a specific variable from assess_vars might be
        # missing in the dataframe.
        # Also assign the filtered dataframe back to the dictionary.
        dict_updates_ts[method] = _df.loc[
            :,
            _df.columns.get_level_values(0).isin(assess_vars)]
    for method in dict_updates_hl.keys():
        _df = dict_updates_hl[method]
        dict_updates_hl[method] = _df.loc[
            :,
            _df.columns.get_level_values(0).isin(assess_vars)]

    return dict_updates_hl, dict_updates_ts


def multi_method_timeseries(assess_vars):
    # MULTI-METHOD ASSESSMENT - AR6 STYLE - TIMESERIES ########################
    # Conclusion: no uncertainty plumes available at time of writing for ROF
    # (Gillett) method, so a multi-method timeseries is not created here.
    # Instead, we plot the individual methods separately as an indicative
    # alternative later.
    return


def multi_method_headlines(
        dict_updates_hl,
        assess_vars,
        IGCC_YR,
        SR15_YR,
        AR6_YR):

    # MULTI-METHOD ASSESSMENT - AR6 STYLE - HEADLINES #########################
    list_of_dfs = []
    periods_to_assess = [f'{AR6_YR-9}\N{EN DASH}{AR6_YR}',
                         f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}',
                         f'{SR15_YR}',
                         f'{IGCC_YR}',
                         f'{SR15_YR} (SR15 definition)',
                         f'{IGCC_YR} (SR15 definition)']
    for period in periods_to_assess:

        dict_updates_Assessment = {}

        for var in assess_vars:
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
            f'results/Assessment-Update-{IGCC_YR}_GMST_headlines.csv')

    return df_updates_Assessment


def figure_individual_timeseries(
        plot_vars,
        dict_updates_ts,
        df_temp_Obs
        ):
    """PLOT TIMESERIES FOR EACH METHOD"""

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
                'vars': [v for v in plot_vars if v in all_data_vars],
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
                {var: defs.get_plot_colour(var)
                 for var in cfg['vars'] + ['Obs']},
                sigmas=['5', '95', '50'],
                labels=label_map,
                linestyle=cfg['linestyle'],
                plume_vars=plume_vars,
            )

            ax.set_ylim(-1, 2)
            ax.set_xlim(START_YR, IGCC_YR)
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
                f'{PLOT_FOLDER}/2_{method}_timeseries{cfg["suffix"]}.png'
            )
            fig.savefig(
                f'{PLOT_FOLDER}/2_{method}_timeseries{cfg["suffix"]}.pdf'
            )


def figure_stacked_method_timeseries(
        plot_vars,
        dict_updates_ts,
        df_temp_Obs,
        var_colours
        ):
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
    ax.set_xlim(START_YR, IGCC_YR)
    ax.text(1875, -0.85, '1850\N{EN DASH}1900\nPreindustrial Baseline',
            ha='center')
    fig.suptitle('Timeseries for each attribution method used '
                 'in the assessment of contributions to observed warming')
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    gr.overall_legend(fig, 'lower center', 3,
                      reorder=[0, 1, 2, 3, 4, 5, 6, 7])
    fig.savefig(f'{PLOT_FOLDER}/2_stacked-multi_method_timeseries.png')
    fig.savefig(f'{PLOT_FOLDER}/2_stacked-multi_method_timeseries.pdf')


def figure_aligned_method_timeseries(
        plot_vars,
        df_updates_ts,
        df_temp_Obs
        ):
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
        ax.set_xlim(1900, IGCC_YR)
        if methods.index(m) > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.set_title(f'{subs[methods.index(m)]} {m}: {labels[m]}')
    gr.overall_legend(fig, 'lower center', 6)
    # fig.suptitle('Testing one two three how do we think this looks?')
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.94))
    fig.suptitle('Timeseries for each attribution method used '
                 'in the assessment of contributions to observed warming')
    fig.savefig(f'{PLOT_FOLDER}/2_aligned-multi_method_timeseries.png')
    fig.savefig(f'{PLOT_FOLDER}/2_aligned-multi_method_timeseries.pdf')


def plot_validation_plot(
        dict_IPCC_hl,
        dict_updates_hl,
        dict_IPCC_Obs_hl,
        dict_updates_Obs_hl,
        source_markers,
        var_colours,
        labels
):
    """PLOT THE VALIDATION PLOT."""
    print('Creating Fig 3.8 Validation Plot')
    bar_plot_vars = ['Ant', 'GHG', 'OHF', 'Nat']
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=4)
    ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 4), rowspan=1, colspan=1)

    gr.Fig_3_8_validation_plot(ax2, ['Ant'], f'{SR15_YR}',
                               dict_IPCC_hl, dict_updates_hl,
                               dict_IPCC_Obs_hl, dict_updates_Obs_hl,
                               source_markers, var_colours, labels)
    gr.Fig_3_8_validation_plot(ax1, bar_plot_vars,
                               f'{AR6_YR-9}\N{EN DASH}{AR6_YR}',
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
             (f'(a) {AR6_YR-9}\N{EN DASH}{AR6_YR} AR6 WG1 Ch.3 (left)\n' +
              f'      vs {AR6_YR-9}\N{EN DASH}{AR6_YR} repeat (right)'),
             ha='left', fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='regular',
             #  fontstyle='italic'
             )
    fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
             f'(b) {SR15_YR} SR1.5 Ch.1 (left)'
             '\n      '
             f'vs {SR15_YR} repeat (right)',
             ha='left', fontsize=matplotlib.rcParams['axes.titlesize'],
             fontweight='regular',
             #  fontstyle='italic'
             )
    fig.savefig(f'{PLOT_FOLDER}/3_WG1_Ch3_Validation.png')
    fig.savefig(f'{PLOT_FOLDER}/3_WG1_Ch3_Validation.pdf')


def figure_SPM2_full(
        dict_IPCC_hl,
        dict_updates_hl,
        dict_IPCC_Obs_hl,
        dict_updates_Obs_hl,
        var_colours,
        var_names,
        labels,
        assess_vars
):

    # Plot the headline SPM2-esque figure #####################################
    print('Creating SPM.2-esque figure')
    text_toggle = True
    fig = plt.figure(figsize=(12, 10))
    ax0 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 1), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 3), rowspan=1, colspan=2)
    gr.Fig_SPM2_plot(
        ax0,
        ['Obs'],
        [f'{AR6_YR-9}\N{EN DASH}{AR6_YR}', f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}'],
        dict_IPCC_hl, dict_updates_Obs_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax1,
        assess_vars,
        [f'{AR6_YR-9}\N{EN DASH}{AR6_YR}', f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax2,
        assess_vars,
        [f'{SR15_YR} (SR15 definition)', f'{IGCC_YR} (SR15 definition)'],
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
    fig.savefig(f'{PLOT_FOLDER}/4_SPM2_Results.png')
    fig.savefig(f'{PLOT_FOLDER}/4_SPM2_Results.pdf')


def figure_SPM2_updates_only(
        dict_IPCC_hl,
        dict_updates_hl,
        dict_IPCC_Obs_hl,
        dict_updates_Obs_hl,
        var_colours,
        var_names,
        labels,
        assess_vars
):

    # Plot the headline SPM2-esque figure without IPCC comparisons ############
    print('Creating SPM.2-esque figure without IPCC comparison bars')
    text_toggle = False
    fig = plt.figure(figsize=(12, 10))
    ax0 = plt.subplot2grid(shape=(1, 5), loc=(0, 0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid(shape=(1, 5), loc=(0, 1), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid(shape=(1, 5), loc=(0, 3), rowspan=1, colspan=2)
    gr.Fig_SPM2_plot(
        ax0, ['Obs'], [f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}'],
        dict_IPCC_hl, dict_updates_Obs_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax1,
        assess_vars,
        [f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}'],
        dict_IPCC_hl, dict_updates_hl,
        var_colours, var_names, labels, text_toggle)
    gr.Fig_SPM2_plot(
        ax2,
        assess_vars,
        [f'{IGCC_YR} (SR15 definition)'],
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
             (f'(b) AR6 Update: {IGCC_YR-9}\N{EN DASH}{IGCC_YR} '
              'decade-average warming'
              '\n      '
              'contributions assessed from attribution studies'),
             fontsize=matplotlib.rcParams['font.size'],
             fontweight='regular'
             )
    fig.text(ax2.get_position().x0, ax2.get_position().y1+0.02,
             (f'(c) SR1.5 Update: {IGCC_YR} present-day warming'
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

    fig.savefig(f'{PLOT_FOLDER}/4_SPM2_Results_Updates-Only.png')
    fig.savefig(f'{PLOT_FOLDER}/4_SPM2_Results_Updates-Only.pdf')


def tables_for_supplement(
        IGCC_YR, SR15_YR, AR6_YR,
        dict_updates_hl, assess_vars):
    """ CREATE APPENDIX-LAYOUT TABLES FOR RESULTS."""

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
            times = [f'{AR6_YR-9}\N{EN DASH}{AR6_YR}',
                     f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}',
                     f'{SR15_YR}', f'{IGCC_YR}',
                     f'{SR15_YR} (SR15 definition)',
                     f'{IGCC_YR} (SR15 definition)']
            f.write('variable, method, ' + ', '.join(times) + '\n')
            for v in assess_vars:
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
            index_col=0,  header=[0, 1], skiprows=0)
    Gillet_GSAT = defs.en_dash_ify(Gillet_GSAT)

    table_gsat_path = './results/ancillary/Table_GSAT_ROF_method.csv'
    write_table_gsat = (
        refresh_ancillary_data or (not os.path.exists(table_gsat_path))
    )
    if write_table_gsat:
        if refresh_ancillary_data and os.path.exists(table_gsat_path):
            os.remove(table_gsat_path)
        with open(table_gsat_path, 'w+') as f:
            times = [
                f'{AR6_YR-9}\N{EN DASH}{AR6_YR}',
                f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}',
                f'{SR15_YR} (SR15 definition)',
                f'{IGCC_YR} (SR15 definition)']
            f.write('variable, ' + ', '.join(times) + '\n')
            for v in assess_vars:
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


def rate_calculate_ERF(
        sigmas_all
        ):

    """ CALCULATE ERF RATES FOR PLOTTING."""
    # Define file paths for caching the ERF calculations
    erf_cache_main = './results/ancillary/Rates_results_ERF.csv'
    erf_cache_full = './results/ancillary/Rates_results_ERF_full-vars.csv'

    # AGGREGATE VARIABLES ERF RATES #######################################
    # Check if we can use the cached main ERF dataset or if we need to
    # recalculate it
    if (not refresh_ancillary_data) and os.path.exists(erf_cache_main):
        # Load the existing aggregate rate dataset to save computation time
        df_forc_rates_main = pd.read_csv(
            erf_cache_main, index_col=0, header=[0, 1], skiprows=0)
    else:
        # Calculate the ERF aggregate rate dataset from scratch
        print('Calculating ERF aggregate rate dataset.')
        df_forc_rates_main = defs.rate_ERF(
            IGCC_YR, sigmas_all, variable_mode='aggregate'
        )
        # Save the newly calculated dataset to cache for future runs
        df_forc_rates_main.to_csv(erf_cache_main)

    # FULL-VARIABLE ERF RATES #############################################
    # Determine if we need to recalculate the full-variable ERF dataset
    regenerate_full_erf = refresh_ancillary_data

    # If a manual refresh isn't requested, check if the cached full dataset
    # exists
    if (not regenerate_full_erf) and os.path.exists(erf_cache_full):
        # Load the existing full dataset
        df_forc_rates_full = pd.read_csv(
            erf_cache_full, index_col=0, header=[0, 1], skiprows=0)
        # Ensure the dataset matches the expected schema; flag for
        # regeneration if invalid
        regenerate_full_erf = not full_erf_schema_valid(df_forc_rates_full)
    elif (not regenerate_full_erf) and (not os.path.exists(erf_cache_full)):
        # If the cache file does not exist, we must generate it
        regenerate_full_erf = True

    # Generate the full-variable ERF dataset if required
    if regenerate_full_erf:
        print('Calculating ERF full-variable rate dataset.')
        df_forc_rates_full = defs.rate_ERF(
            IGCC_YR, sigmas_all, variable_mode='all'
        )
        # Cache the new result so it can be reused later
        df_forc_rates_full.to_csv(erf_cache_full)

    # Return both the main (aggregate) and full-variable ERF rate
    # dataframes for plotting
    return df_forc_rates_main, df_forc_rates_full


def figure_rates(
        assess_vars,
        var_colours,
        var_names,
):
    print('Creating rate plots...')

    sigmas = [[17, 83], [5, 95]]
    sigmas_all = list(
        np.concatenate((np.sort(np.ravel(sigmas)), [50]), axis=0)
        )

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
        times_local = [
            int(y.split(' ')[0].split('-')[1])
            for y in df_rates.index
            ]

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

    # Plot attributed warming rates ###########################################
    # Import the rates csv file (files are a list to facilitate comparing
    # multiple samplings by plotting them atop each other).
    df_rates_walsh = pd.read_csv(
        './results/Walsh_GMST_rates.csv',
        index_col=0, header=[0, 1], skiprows=0
    )

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
            START_PI, END_PI, START_YR, IGCC_YR, sigmas_all)
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
        df_Obs_IGCC = defs.load_Temp_IGCC(START_PI, END_PI, IGCC_YR)
        df_Obs_IGCC_rate = defs.rate_IGCC(START_PI, END_PI, START_YR, IGCC_YR)
        df_Obs_IGCC_rate.to_csv(igcc_rate_cache)

    # ax1.plot(times,  df_Obs_IGCC_rate[('Obs', '50')]*10,
    #          color='black',
    #          label='Reference Observations: IGCC',
    #          lw=2)

    rate_ticks = list(np.arange(1950, IGCC_YR+1, 20))
    rate_ticks.append(IGCC_YR)

    df_forc_rates_main, df_forc_rates_full = rate_calculate_ERF(
        sigmas_all
    )

    # Plot ERF Rates ##########################################################

    rate_plot_configs = [
        {
            'name': 'main',
            'suffix': '',
            'gwi_vars': [
                var for var in assess_vars
                if (var, '50') in df_rates_walsh.columns
            ],
            'erf_rates_df': df_forc_rates_main,
            'erf_vars': [
                var for var in assess_vars
                if (var, '50') in df_forc_rates_main.columns
            ],
            'legend_loc': 'lower center',
            'legend_ncol': 5,
        },
        {
            'name': 'all-vars',
            'suffix': '_all-vars',
            'gwi_vars': [
                var
                for var
                in df_rates_walsh.columns.get_level_values(0).unique()
                if var != 'Obs'
            ],
            'erf_rates_df': df_forc_rates_full,
            'erf_vars': [
                var
                for var
                in df_forc_rates_full.columns.get_level_values(0).unique()
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
            erf_plume_vars = [
                v for v in main_plume_vars if v in cfg['erf_vars']]
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
        err_pos = (df_rates_GWI[('Obs', '95')] - df_rates_GWI[('Obs', '50')])
        err_neg = (df_rates_GWI[('Obs', '50')] - df_rates_GWI[('Obs', '5')])
        err_pos *= 10
        err_neg *= 10
        obs_label = rate_label_map.get(
            'Obs', 'Reference Observations: HadCRUT5')
        ax1.errorbar(
            times_gwi, df_rates_GWI[('Obs', '50')] * 10,
            yerr=(err_neg, err_pos),
            fmt='o', color=var_colours['Obs'], ms=2.5, lw=1,
            label=obs_label
        )

        # Add a line along the y=0 line
        ax1.axhline(0, color='black', lw=0.5)
        ax1.set_xlim([1950, IGCC_YR+1])
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
        ax2.set_xlim([1950, IGCC_YR+1])
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
        fig.savefig(f'{PLOT_FOLDER}/5_Rates_timeseries{cfg["suffix"]}.png')
        fig.savefig(f'{PLOT_FOLDER}/5_Rates_timeseries{cfg["suffix"]}.pdf')


def figure_definition_diagram(
        IGCC_YR, START_YR,
        dict_updates_hl, df_temp_Obs, dict_updates_ts,
        var_colours
):

    # Load the assessment results
    df_headlines = pd.read_csv(
        f"results/Assessment-Update-{IGCC_YR}_GMST_headlines.csv",
        index_col=0,  header=[0, 1], skiprows=0
    )
    df_headlines = defs.en_dash_ify(df_headlines)

    # Plot the GWI 'Ant' 50th percentile
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    gr.GWI_definition_diagram(
        ax1, IGCC_YR,
        dict_updates_hl['Walsh'], df_temp_Obs, dict_updates_ts['Walsh'],
        var_colours)
    ax1.set_ylabel(
        'Global mean surface temperature,\n' +
        'relative to 1850\N{EN DASH}1900 baseline (°C)'
        )

    ax1.set_ylim(0.75, 1.6)
    ticks = list(np.arange(START_YR, IGCC_YR, 5))
    ticks.append(IGCC_YR)
    ax1.set_xticks(ticks, ticks)
    ax1.set_yticks([1.0, 1.5])
    ax1.set_xlim(2002.5, IGCC_YR + 1)
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

    fig.savefig(f'{PLOT_FOLDER}/1_definition_diagram_GWI.png')
    fig.savefig(f'{PLOT_FOLDER}/1_definition_diagram_GWI.pdf')


def figure_COP_context(
        IGCC_YR, START_PI, END_PI,
        dict_updates_ts,
        dict_updates_hl,
        metoffice_proj_next_yr,
        ant_sr15_proj_increment,
):
    """Create the COP communication figure.

    Shows observed GMST and the SR1.5-definition human-induced warming
    timeseries, a forward projection for the upcoming year, and a
    decade-average band. Designed for COP plenary readability.

    Parameters
    ----------
    metoffice_proj_next_yr : float
        Met Office HadCRUT projection for IGCC_YR+1. Update each year.
    ant_sr15_proj_increment : float
        Projected 1-year increase in multi-method mean Ant SR15 warming.
        Update each year (e.g. from the annual attribution rate).
    """
    print('Creating COP context figure...')

    # -------------------------------------------------------------------------
    # COP-specific matplotlib styling — applied via rc_context so it does not
    # bleed into other figures. Adjust here to tune the look for COP presentations.
    # -------------------------------------------------------------------------
    _font = 'Arial'
    cop_style = {
        'font.family': _font,
        'font.size': 14,
        'mathtext.fontset': 'custom',
        'mathtext.rm': _font,
        'mathtext.bf': f'{_font}:bold',
        'mathtext.cal': _font,
        'legend.frameon': False,
        'axes.spines.bottom': True,
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 2,
        'axes.facecolor': 'white',
        'axes.titleweight': 'regular',
        'axes.edgecolor': '#D1CFCD',
        'xtick.color': '#D1CFCD',
        'xtick.labelcolor': '#6F6F6F',
        'ytick.color': '#6F6F6F',
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': '#D1CFCD',
        'grid.linewidth': 1.5,
        'ytick.major.size': 0,
        'ytick.major.width': 0,
    }

    # Colour scheme for this communication figure (bold red/black)
    cop_colours = {
        'Obs': 'black',
        'Ant': '#C71518',
        'Ant decade': '#C71518',
    }

    # -------------------------------------------------------------------------
    # Data computation (outside rc_context)
    # -------------------------------------------------------------------------
    df_Obs_IGCC = defs.load_Temp_IGCC(START_PI, END_PI, IGCC_YR)

    # Compute SR1.5-definition warming per method: for each year, fit a
    # 15-year linear trend to the Ant timeseries and record the final-year
    # value of that trend (the SR1.5 "present-day warming" definition).
    dict_updates_SR15 = {}
    hl_years = np.arange(1990, IGCC_YR + 1)
    for method in ['Walsh', 'Ribes', 'Gillett']:
        trunc_Yrs = dict_updates_ts[method].index.values
        df_method_SR15 = dict_updates_ts[method].copy()
        df_method_SR15[:] = np.zeros(df_method_SR15.shape)
        for year in hl_years:
            mask = (year - 14 <= trunc_Yrs) & (trunc_Yrs <= year)
            trend_end = np.apply_along_axis(
                func1d=defs.final_value_of_trend,
                axis=0,
                arr=dict_updates_ts[method].values[mask, :]
            )
            df_method_SR15.loc[year, :] = trend_end
        dict_updates_SR15[method] = df_method_SR15

    sr15_mmm = (
        dict_updates_SR15['Walsh'] +
        dict_updates_SR15['Ribes'] +
        dict_updates_SR15['Gillett']
    ) / 3

    df_headlines = dict_updates_hl['Assessment']  # already en-dashed in memory
    decade_label = f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}'

    obs_year = int(df_Obs_IGCC.index[-1])
    obs_value = float(df_Obs_IGCC['GMST'].iloc[-1])
    ant_sr15_now = float(sr15_mmm[('Ant', '50')].loc[IGCC_YR])
    decade_avg_value = float(df_headlines[('Ant', '50')].loc[decade_label])
    human_label_value = float(
        df_headlines[('Ant', '50')].loc[f'{IGCC_YR} (SR15 definition)'])

    # -------------------------------------------------------------------------
    # Annotation spacing — adjust these to tune label positions
    # -------------------------------------------------------------------------
    arrow_offset = 0.1   # gap between data point and arrowhead (years)
    arm_length = 2.5     # length of horizontal level arrow (years)
    FLICK_X = 1.0        # horizontal component of 45° flick to text (years)
    FLICK_Y = 0.05       # vertical component of 45° flick (°C); tune for visual angle
    arrow_width = 1.2
    arrow_head_size = 20

    # -------------------------------------------------------------------------
    # Figure — everything inside rc_context uses the COP-specific styling
    # -------------------------------------------------------------------------
    with matplotlib.rc_context(cop_style):
        fig = plt.figure(figsize=(12, 7))
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))
        marker_size = 3
        marker_size_highlight = 7

        # Observed GMST solid line
        ax.plot(
            df_Obs_IGCC.index, df_Obs_IGCC['GMST'],
            color=cop_colours['Obs'],
            linewidth=1.2, alpha=1.0,
            marker='o', markersize=marker_size,
            label='Observed warming'
        )
        # Human-induced (SR1.5-definition, multi-method mean) solid line
        ax.plot(
            sr15_mmm.index, sr15_mmm[('Ant', '50')],
            color=cop_colours['Ant'],
            linewidth=2.5, alpha=1.0,
            marker='o', markersize=marker_size,
            label='Human-induced warming'
        )

        # Highlighted dots at IGCC_YR
        ax.scatter(
            IGCC_YR, ant_sr15_now,
            color=cop_colours['Ant'], s=marker_size_highlight * 8,
            edgecolor=cop_colours['Ant'], marker='o', lw=0
        )
        ax.scatter(
            IGCC_YR, obs_value,
            color=cop_colours['Obs'], s=marker_size_highlight * 8,
            edgecolor=cop_colours['Obs'], marker='o', lw=0
        )

        # Key policy / scientific events — update the COP entry each year
        key_dates = {
            2010: 'Cancún Agreement',
            2015: 'Paris Agreement',
            2021: 'IPCC AR6 WGI',
            2026: 'COP31 Antalya',  # ← upc date label (and year if needed) each year
        }
        for year, label in key_dates.items():
            ax.axvline(
                x=year, color='#cfd1d0', linestyle='-', linewidth=0.5, zorder=0)
            ax.text(
                x=year - 0.6, y=0.5 + 0.015, s=label,
                rotation=90, verticalalignment='bottom',
                color='#6F6F6F', fontsize=12
            )

        # Decade-average human-induced warming band (last 10 years of assessment)
        ax.plot(
            df_Obs_IGCC.index[-10:],
            [decade_avg_value] * 10,
            color=cop_colours['Ant decade'], alpha=0.5, linestyle='-', lw=0.5
        )
        ax.fill_between(
            df_Obs_IGCC.index[-10:],
            [decade_avg_value] * 10,
            sr15_mmm[('Ant', '50')].iloc[-10:],
            color=cop_colours['Ant decade'], alpha=0.05, lw=0
        )
        ax.scatter(
            IGCC_YR - 9 + 4.5, decade_avg_value,
            color=cop_colours['Ant decade'], s=marker_size_highlight * 8,
            edgecolor=cop_colours['Ant decade'], marker='o', alpha=0.5, lw=0
        )

        # Annotations: horizontal arm from the data point, then a short 45°
        # flick leading to the separated text label. A single annotate call
        # with connectionstyle="angle" guarantees a gap-free elbow.
        # angleB=0 keeps the arm horizontal at the data end.
        # angleA must exit the text end leftward (toward data): 135° for
        # upward flick (text above arm), 225° for downward (text below arm),
        # i.e. angleA = 180 - 45*flick_sign.
        # Red annotations flick downward; black (obs) flicks upward.
        # Tune arm_length, FLICK_X, FLICK_Y above to adjust layout.
        def _cop_annotate(data_x, data_y, flick_up, text_str, color, alpha=1.0):
            flick_sign = 1 if flick_up else -1
            text_x = data_x + arm_length + FLICK_X
            text_y = data_y + flick_sign * FLICK_Y
            ax.annotate(
                text_str,
                xy=(data_x + arrow_offset, data_y),
                xytext=(text_x, text_y),
                color=color, alpha=alpha, fontweight='regular',
                arrowprops=dict(
                    color=color,
                    arrowstyle='->',
                    mutation_scale=arrow_head_size,
                    linewidth=arrow_width,
                    alpha=alpha,
                    # tail attaches at left edge of text box, halfway up
                    relpos=(0, 0.5),
                    # shrinkA=5, shrinkB=0,
                    connectionstyle=f"angle,angleA={45 * flick_sign},angleB=0,rad=5",
                ),
                verticalalignment='center',
                horizontalalignment='left',
                # annotation_clip=False,
            )

        _cop_annotate(
            obs_year, obs_value, True,
            'Observed\n' + f'{obs_year}\n' +
            r"$\bf{" + f"{obs_value:.2f}" + "}$ °C",
            cop_colours['Obs'],
        )
        _cop_annotate(
            IGCC_YR, ant_sr15_now, False,
            'Human-induced\n' + f'{IGCC_YR}\n' +
            r"$\bf{" + f"{human_label_value:.2f}" + "}$ °C",
            cop_colours['Ant'],
        )
        _cop_annotate(
            IGCC_YR, decade_avg_value, False,
            'Human-induced\n' +
            f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR} average\n' +
            r"$\bf{" + f"{decade_avg_value:.2f}" + "}$ °C",
            cop_colours['Ant decade'],
            alpha=0.5,
        )

        # Axis formatting
        ax.set_ylabel('Global Surface Temperature Increase\n')
        ax.yaxis.label.set_size(16)
        # Fixed y-range for COP communication; update if warming exceeds 1.55 °C
        ax.set_ylim(0.5, 1.55)
        ax.set_yticks([0.5, 1.0, 1.5], ['0.5 °C ', '1.0 °C ', '1.5 °C '])
        ax.get_yticklabels()[2].set_color(cop_colours['Ant'])
        ax.get_ygridlines()[2].set_color(cop_colours['Ant'])

        # x-axis: 5-year ticks up to and including IGCC_YR; projection year
        # is shown via the dashed line but gets no tick to avoid crowding
        ticks = list(np.arange(1990, IGCC_YR, 5)) + [IGCC_YR]
        ax.set_xticks(ticks, ticks)
        ax.set_xlim(1999.5, IGCC_YR + 2.2)

        fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))

        # ---------------------------------------------------------------------
        # Version 1 — without next-year projection. Everything above is shared
        # by both versions; save the figure as-is here first.
        # ---------------------------------------------------------------------
        fig.savefig(f'{PLOT_FOLDER}/8_COP_context_no_projection.png', dpi=300)
        fig.savefig(f'{PLOT_FOLDER}/8_COP_context_no_projection.pdf')

        # ---------------------------------------------------------------------
        # Version 2 — add the dashed next-year projection segments on top of
        # the shared figure, then re-save. Keeping this as an additive block
        # means both versions stay in sync when updated in future years.
        # ---------------------------------------------------------------------
        # Dashed projection segments from IGCC_YR → IGCC_YR+1
        ax.plot(
            [IGCC_YR, IGCC_YR + 1],
            [df_Obs_IGCC['GMST'].loc[IGCC_YR], metoffice_proj_next_yr],
            color=cop_colours['Obs'],
            linewidth=1.2, linestyle='--',
            marker='o', markersize=marker_size,
            markerfacecolor='none',
            markeredgecolor=cop_colours['Obs'],
            markeredgewidth=2
        )
        ax.plot(
            [IGCC_YR, IGCC_YR + 1],
            [ant_sr15_now, ant_sr15_now + ant_sr15_proj_increment],
            color=cop_colours['Ant'],
            linewidth=1.2, linestyle='--',
            marker='o', markersize=marker_size,
            markerfacecolor='none',
            markeredgecolor=cop_colours['Ant'],
            markeredgewidth=2
        )
        fig.savefig(f'{PLOT_FOLDER}/8_COP_context.png', dpi=300)
        fig.savefig(f'{PLOT_FOLDER}/8_COP_context.pdf')


def compare_assess_years(
        compare_years, IGCC_YR, dict_updates_ts
):

    dict_analysis_ts = {}

    for method in dict_updates_ts.keys():
        dict_analysis_ts[method] = {}
        for year in compare_years:
            if year == str(IGCC_YR):
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
    return dict_analysis_ts


def figure_assessment_interannual_timeseries_difference(
    main_vars,
    START_PI, END_PI,
    IGCC_YR, START_YR,
    dict_updates_ts,
    var_colours,
    dict_analysis_ts,
    compare_years,
    linestyles
):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

    for method in dict_analysis_ts.keys():
        # Plot the ('Ant', '50') columns of timeseries against each other
        for var in main_vars:
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
    ax.set_xticks(np.append(np.arange(START_PI, int(min(compare_years))+1, 50),
                  int(min(compare_years))))
    ax.set_yticks(np.arange(-0.05, 0.05, 0.005))

    ax.fill_between([START_PI, END_PI], [-5, -5], [+5, +5], color='#f4f2f1')
    ax.text(1875, -0.055, '1850\N{EN DASH}1900\nPreindustrial Baseline',
            ha='center')
    ax.set_ylim(-0.06, 0.08)
    ax.set_ylabel(
        f'{max(compare_years)} analysis minus {min(compare_years)} analysis,' +
        ' 50th percentiles, °C'
        )
    plt.suptitle('Difference between analysis years')
    plt.savefig(f'{PLOT_FOLDER}/6_analysis_components_comparison.png')
    plt.savefig(f'{PLOT_FOLDER}/6_analysis_components_comparison.pdf')


def check_pi_average(
        main_vars, dict_analysis_ts, IGCC_YR):

    for method in sorted(dict_analysis_ts.keys()):
        print(f'Average of 1850-1900 for {method}:')
        for var in main_vars:
            avg_1850_1900 = dict_analysis_ts[method][str(IGCC_YR)].loc[
                (dict_analysis_ts[method][str(IGCC_YR)].index >= 1850) &
                (dict_analysis_ts[method][str(IGCC_YR)].index <= 1900),
                (var, '50')
            ].mean()
            print(f'  {var}: {avg_1850_1900:.8f} °C')


def figure_assessment_interannual_ant_update(
    START_PI, END_PI,
    IGCC_YR, START_YR,
    dict_analysis_ts,
    compare_years,
    var_colours,
    linestyles
):

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
        f'{IGCC_YR}': 'xkcd:teal',
        f'{IGCC_YR-1}': 'xkcd:tomato',
    }
    for year in compare_years:
        # TIMESERIES
        # Create an empty timeseries
        method_keys = list(dict_analysis_ts.keys())
        method_keys.remove('Average') if 'Average' in method_keys else None

        Ant_average = dict_analysis_ts['Walsh'][year][('Ant', '50')].copy()
        Ant_average[:] = 0

        for method in method_keys:
            Ant_average += dict_analysis_ts[method][year][('Ant', '50')]
            ax.plot(dict_analysis_ts[method][year][('Ant', '50')],
                    year_colours[str(year)], linestyle=linestyles[method],
                    label=method_names[method])
        Ant_average /= len(method_keys)
        dict_analysis_ts['Average'][year] = Ant_average

        ax.plot(Ant_average, label='Multi-method Average', color='black')
    fig.suptitle('Anthropogenic warming best estimate: ' +
                 'three attribution methods and their multi-method average')
    ax.set_ylabel('Ant 50th percentile, °C')
    ax.set_xlim(2000, IGCC_YR+1)
    ax.set_ylim(0.6, 1.6)
    gr.overall_legend(fig, 'lower center', 4)
    fig.savefig(f'{PLOT_FOLDER}/7_Compare_{"-".join(compare_years)}.png')
    fig.savefig(f'{PLOT_FOLDER}/7_Compare_{"-".join(compare_years)}.pdf')


def write_interannual_ant_changes(
        dict_analysis_ts, compare_years
):
    print('\n')
    print(dict_analysis_ts['Average'].keys())

    for method in sorted(dict_analysis_ts.keys()):
        print(f'Comparison for: {method}')
        # print(dict_analysis_ts[method])
        print('  Comparing',
              ' and '.join(dict_analysis_ts[method].keys()), ':')

        print(f'  {compare_years[-1]} analysis gives results '
              f'for year {compare_years[-1]}:', end=' ')
        if method == 'Average':
            a = dict_analysis_ts[method][compare_years[-1]][
                int(compare_years[-1])]
        else:
            a = dict_analysis_ts[method][compare_years[-1]].loc[
                int(compare_years[-1]), ('Ant', '50')]
        print(a)

        print(f'  {compare_years[-2]} analysis gives results '
              f'for year {compare_years[-1]}:', end=' ')
        if method == 'Average':
            b = dict_analysis_ts[method][compare_years[-2]][
                int(compare_years[-1])]
        else:
            b = dict_analysis_ts[method][compare_years[-2]].loc[
                int(compare_years[-1]), ('Ant', '50')]
        print(b)

        print(f'  {compare_years[0]} analysis gives results '
              f'for year {compare_years[0]}:', end=' ')
        if method == 'Average':
            c = dict_analysis_ts[method][compare_years[-2]][
                int(compare_years[-2])]
        else:
            c = dict_analysis_ts[method][compare_years[-2]].loc[
                int(compare_years[-2]), ('Ant', '50')]
        print(c)

        print(f'  Therefore the {compare_years[-1]} revision is: {b-a}')
        print(f'  Therefore the {compare_years[-2]} forced increase is: {c-b}')
        print(f'  Therefore the overall year-on-year change is: {c-a}')

    print('\n')


def figure_raw_erfs():
    df_forc = defs.load_ERF_CMIP6(IGCC_YR)
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
    fig.savefig(f'{PLOT_FOLDER}/0_ERF_plottest.png')
    fig.savefig(f'{PLOT_FOLDER}/0_ERF_plottest.pdf')


def extrapolate_assessment(
        dict_updates_hl, methods, IGCC_YR
):
    extrap_times = [
        f'{IGCC_YR}',
        f'{IGCC_YR} (SR15 definition)',
        f'{IGCC_YR-9}\N{EN DASH}{IGCC_YR}'
        ]

    extrap_times_new = [
        f'{IGCC_YR+1}',
        f'{IGCC_YR+1} (SR15 definition)',
        f'{IGCC_YR+1-9}\N{EN DASH}{IGCC_YR+1}'
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
                df_rate = pd.read_csv(
                    f'./results/{method}_GMST_rates.csv',
                    index_col=0, header=[0, 1], skiprows=0)
                rate = df_rate.loc[
                    f'{IGCC_YR-9}-{IGCC_YR} (AR6 rate definition)',
                    (extrap_var, p)]
                extrap_result = current + rate
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
            f'results/Assessment-Extrapolation-{IGCC_YR+1}_GMST_headlines.csv')


if __name__ == '__main__':

    ###########################################################################
    # Checklist of inputs to this script that need updating each year:
    # 1. GWI Results (manually add csv to results/ directory)
    # 2. KCC Results (manually add csv to results/ directory)
    # 3. ROF Results (manually add csv to results/ directory)
    # 4. Observed Warming Update (manually add in this script - get from team)
    # 5. HadCRUT Observed Warming (manually add csv to
    #    attribution_methods/GlobalWarmingIndex/data/Temp/HadCRUT)

    # Checklist of inputs that don't need updating each year
    # 1. CMIP6 PiControl simulations (update only when CMIP7 is ready)
    # 2. IPCC Quoted reults (never update; quotes from IPCC AR6 and SR1.5)

    # Define date constants for the assessment
    START_PI, END_PI = 1850, 1900
    START_YR = 1850
    IGCC_YR = 2025
    AR6_YR = 2019
    SR15_YR = 2017

    # =========================================================================
    # Annual manual inputs for figure_COP_context — update each year
    # =========================================================================
    # Met Office HadCRUT projection for IGCC_YR+1: loaded from
    # data/Temp/MetOffice/HadCRUT_annual_projections.csv — append a row there.
    METOFFICE_PROJ_NEXT_YR = defs.load_metoffice_projection(IGCC_YR)

    # Projected 1-year increase in multi-method mean Ant SR15 warming.
    # Derived from the annual rate of change in the attribution timeseries.
    ANT_SR15_PROJ_INCREMENT = 0.027  # TODO: update each year if needed

    # =========================================================================

    main_vars = ['Tot', 'Ant', 'GHG', 'Nat', 'OHF']
    assess_vars = ['Ant', 'GHG', 'Nat', 'OHF']
    var_names = defs.VAR_NAMES.copy()

    sigmas = [[17, 83], [5, 95]]
    sigmas_all = list(
        np.concatenate((np.sort(np.ravel(sigmas)), [50]), axis=0)
        )

    # Parse arguments to determine whether to refresh ancillary data files
    refresh_ancillary_data = parse_arguments()

    # Load attribution reference datasets #####################################
    # Load reference observations used in attribution (HadCRUT)
    # TODO: Update file each year
    df_temp_Obs = defs.load_HadCRUT(START_PI, END_PI, START_YR, IGCC_YR)

    # Load reference pi-control runs
    df_temp_PiC = load_filtered_PiC(
        START_YR, IGCC_YR, START_PI, END_PI, df_temp_Obs.shape[0])

    # LOAD IPCC-QUOTED RESULTS ################################################
    # Load IPCC-quoted observation results
    df_AR6_Obs = IPCC_Obs_data.load_observation_assessment()
    dict_IPCC_Obs_hl = {'Assessment': df_AR6_Obs}

    # Load IPCC-quoted attribution results
    dict_IPCC_hl = IPCC_AGW_data.load_IPCC_df()

    # Load IGCC results #######################################################
    # Load IGCC observation assessment (IGCC Obs team to provide this as a csv)
    # TODO: Update file each year, and update the load_observation_assessment
    df_IGCC_Obs = IGCC_Obs_data.load_observation_assessment(assess_yr=IGCC_YR)
    dict_updates_Obs_hl = {'Assessment': df_IGCC_Obs}

    # Load attribution results
    # TODO: Update files each year
    dict_updates_hl, dict_updates_ts = load_attribution_results(main_vars)

    # Create multi-method timeseries assessment
    # PLACEHOLDER
    _ = multi_method_timeseries(assess_vars)

    # Create multi-method headline assessment
    df_updates_Assessment = multi_method_headlines(
        dict_updates_hl, assess_vars, IGCC_YR, SR15_YR, AR6_YR)

    # Definte colours, labels, and symbols for graphing
    var_colours = defs.get_var_colours()

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

    PLOT_FOLDER = f'./plots/{IGCC_YR}/'
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)

    # PLOT RAW ERFS
    figure_raw_erfs()

    # PLOT TIMESERIES FOR EACH METHOD
    figure_individual_timeseries(
        plot_vars=assess_vars,
        dict_updates_ts=dict_updates_ts,
        df_temp_Obs=df_temp_Obs
        )

    # PLOT STACKED MULTI-METHOD TIMESERIES
    figure_stacked_method_timeseries(
        plot_vars=assess_vars,
        dict_updates_ts=dict_updates_ts,
        df_temp_Obs=df_temp_Obs,
        var_colours=var_colours
        )

    # PLOT ALIGNED MULTI-METHOD TIMESERIES
    figure_aligned_method_timeseries(
        plot_vars=assess_vars,
        df_updates_ts=dict_updates_ts,
        df_temp_Obs=df_temp_Obs
        )

    # PLOT VALIDATION PLOT
    plot_validation_plot(
        dict_IPCC_hl=dict_IPCC_hl,
        dict_updates_hl=dict_updates_hl,
        dict_IPCC_Obs_hl=dict_IPCC_Obs_hl,
        dict_updates_Obs_hl=dict_updates_Obs_hl,
        source_markers=source_markers,
        var_colours=var_colours,
        labels=labels
    )

    # PLOT SPM2-ESQUE FIGURE
    figure_SPM2_full(
        dict_IPCC_hl=dict_IPCC_hl,
        dict_updates_hl=dict_updates_hl,
        dict_IPCC_Obs_hl=dict_IPCC_Obs_hl,
        dict_updates_Obs_hl=dict_updates_Obs_hl,
        var_colours=var_colours,
        var_names=var_names,
        labels=labels,
        assess_vars=assess_vars
    )

    # PLOT SPM2-ESQUE FIGURE WITH UPDATES ONLY
    figure_SPM2_updates_only(
        dict_IPCC_hl=dict_IPCC_hl,
        dict_updates_hl=dict_updates_hl,
        dict_IPCC_Obs_hl=dict_IPCC_Obs_hl,
        dict_updates_Obs_hl=dict_updates_Obs_hl,
        var_colours=var_colours,
        var_names=var_names,
        labels=labels,
        assess_vars=assess_vars
    )

    # CREATE TABLES FOR SUPPLEMENT
    tables_for_supplement(
        IGCC_YR=IGCC_YR,
        SR15_YR=SR15_YR,
        AR6_YR=AR6_YR,
        dict_updates_hl=dict_updates_hl,
        assess_vars=assess_vars
    )

    # Calculate ERF rates
    df_forc_rates_full, df_forc_rates_main = rate_calculate_ERF(
        sigmas_all
    )

    # PLOT RATES FIGURES
    figure_rates(
        assess_vars=assess_vars,
        var_colours=var_colours,
        var_names=var_names,
    )

    # PLOT DEFINITION DIAGRAM
    figure_definition_diagram(
        IGCC_YR=IGCC_YR,
        START_YR=START_YR,
        dict_updates_hl=dict_updates_hl,
        df_temp_Obs=df_temp_Obs,
        dict_updates_ts=dict_updates_ts,
        var_colours=var_colours
    )

    # PLOT COP CONTEXT FIGURE
    figure_COP_context(
        IGCC_YR=IGCC_YR,
        START_PI=START_PI,
        END_PI=END_PI,
        dict_updates_ts=dict_updates_ts,
        dict_updates_hl=dict_updates_hl,
        metoffice_proj_next_yr=METOFFICE_PROJ_NEXT_YR,
        ant_sr15_proj_increment=ANT_SR15_PROJ_INCREMENT,
    )

    compare_years = [str(IGCC_YR), str(IGCC_YR-1)]
    linestyles = {'Walsh':   '-', 'Ribes':   '--', 'Gillett': ':'}

    # Compare assessment years
    dict_analysis_ts = compare_assess_years(
        compare_years=compare_years,
        IGCC_YR=IGCC_YR,
        dict_updates_ts=dict_updates_ts
    )

    # Check PI average
    check_pi_average(
        main_vars=main_vars,
        dict_analysis_ts=dict_analysis_ts,
        IGCC_YR=IGCC_YR
    )

    # Plot comparison figure
    figure_assessment_interannual_timeseries_difference(
        main_vars=main_vars,
        START_PI=START_PI,
        END_PI=END_PI,
        IGCC_YR=IGCC_YR,
        START_YR=START_YR,
        dict_updates_ts=dict_updates_ts,
        var_colours=var_colours,
        dict_analysis_ts=dict_analysis_ts,
        compare_years=compare_years,
        linestyles=linestyles
    )

    # Plot the anthropogenic warming best estimate comparison figure
    figure_assessment_interannual_ant_update(
        START_PI=START_PI,
        END_PI=END_PI,
        IGCC_YR=IGCC_YR,
        START_YR=START_YR,
        dict_analysis_ts=dict_analysis_ts,
        compare_years=compare_years,
        var_colours=var_colours,
        linestyles=linestyles
    )

    # Write out the interannual changes in the anthropogenic warming best
    # estimate
    write_interannual_ant_changes(
        dict_analysis_ts=dict_analysis_ts,
        compare_years=compare_years
    )
