import os
import sys
import datetime as dt
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src import graphing as gr
from src import definitions as defs



font_family = 'Arial'

matplotlib.rcParams.update({
    # General figure
    'figure.dpi': 300,
    'figure.figsize': (15, 10),
    'figure.titlesize': 17,
    'figure.titleweight': 'light',
    'legend.frameon': False,
    # General fonts
    'font.family':  font_family,
    # 'font.weight': 'light',
    'font.size': 14,
    # 'pdf.fonttype': 42,  # Switch from default 3 to 42 to use TrueType fonts
    # for published PDF figures
    # Mathtext fonts
    'mathtext.fontset': 'custom',
    'mathtext.rm': font_family,
    'mathtext.bf': f'{font_family}:bold',
    'mathtext.cal': font_family,  # To pre-emptively stop the matplotlib error of
    # being unable to find a calligraphic/cursive font on the linux cluster,
    # specigy to just use font_family font choice for these cases.
    # Axis box
    'axes.spines.bottom': True,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 2,
    # 'axes.facecolor': '#f9f8f7',  # AR6 at 50% opacity.
    'axes.facecolor': 'white',
    # Axis labels
    'axes.titleweight': 'regular',
    # 'axes.labelcolor': 'gray',
    # AXes color
    'axes.edgecolor': '#D1CFCD',
    # Axes tick color
    'xtick.color': '#D1CFCD',
    'xtick.labelcolor': "#6F6F6F",
    'ytick.color': '#6F6F6F',
    # Axis grid
    'axes.grid': True,
    'axes.grid.axis': 'y',
    # 'grid.color': '#cfd1d0',  # AR6
    'grid.color': '#D1CFCD',
    'grid.linewidth': 1.5,
    # 'axes.axisbelow': True,
    # Axis  ticks
    'ytick.major.size': 0,
    'ytick.major.width': 0,
    # 'ytick.color': 'gray',
})


var_colours = {
    'Obs': 'black',
    'Ant': "#C71518",  # Red
    'Ant decade': "#C71518",  # Light red
    # 'Obs decade': '#C6C4C3'
    'Obs decade': 'xkcd:slate grey'
}

if __name__ == '__main__':

    start_pi = 1850
    end_pi = 1900

    # Load IGCC Warming Observation Data
    df_Obs_IGCC = defs.load_Temp_IGCC(start_pi, end_pi)

    # Load the headline assessment from IGCC update
    df_headlines = pd.read_csv(
        "results/Assessment-Update-2024_GMST_headlines.csv",
        index_col=0,  header=[0, 1], skiprows=0
    )
    df_headlines = defs.en_dash_ify(df_headlines)

    # Load IGCC Warming Attribution Data
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
        df_method_hl - defs.en_dash_ify(df_method_hl)

        dict_updates_hl[method] = df_method_hl
        dict_updates_ts[method] = df_method_ts

    # No Tot warming provided for ROF, so include an indicative approximation
    # as the sum of Ant and Nat warming
    dict_updates_ts['Gillett'].loc[:, ('Tot', '50')] = (
        dict_updates_ts['Gillett'].loc[:, ('Ant', '50')] +
        dict_updates_ts['Gillett'].loc[:, ('Nat', '50')]
        )

    # Calculate the rolling SR1.5 definition warming on the Ant-50 timeseries
    # Calculate the linear trend of the final 15 years of the timeseries
    # and use this to calculate the present-day warming
    print('Calculating SR15-definition temps', end=' ')
    dict_updates_SR15 = {}

    hl_years = np.arange(1990, 2024+1)
    for method in dict_updates_ts.keys():
        trunc_Yrs = dict_updates_ts[method].index.values
        df_method_SR15 = dict_updates_ts[method].copy()
        df_method_SR15[:] = np.zeros(df_method_SR15.shape)

        for year in hl_years:
            years_SR15 = ((year-14 <= trunc_Yrs) * (trunc_Yrs <= year))
            temp_Att_Results_SR15_recent = dict_updates_ts[
                method].values[years_SR15, :]
            # Use Numpy to apply the function defs.final_value_of_trend along
            # the 0 axis of temp_Att_Results_SR15_recent array
            trend_SR15_recent = np.apply_along_axis(
                func1d=defs.final_value_of_trend,
                axis=0,
                arr=temp_Att_Results_SR15_recent
                )
            df_method_SR15.loc[year, :] = trend_SR15_recent
        dict_updates_SR15[method] = df_method_SR15


    # Calculate multi-method mean of the timeseries results
    dict_updates_ts['Multi-method mean'] = (
        dict_updates_ts['Walsh'] +
        dict_updates_ts['Ribes'] +
        dict_updates_ts['Gillett']
    ) / 3
    # Calculate the multi-method mean of the SR15 results
    dict_updates_SR15['Multi-method mean'] = (
        dict_updates_SR15['Walsh'] +
        dict_updates_SR15['Ribes'] +
        dict_updates_SR15['Gillett']
    ) / 3

    # # Calculate the extrapolated 2025 value for 'Multi-method mean' SR15
    # temp = dict_updates_SR15['Multi-method mean'][('Ant', '50')].values[-10:]
    # print(temp)
    # time = dict_updates_SR15['Multi-method mean'].index.values[-10:]
    # print(time)
    # fit = np.poly1d(np.polyfit(time, temp, 1))
    # value = fit(2025)
    # print(value)

    # Create figure
    fig = plt.figure(figsize=(12, 7))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))
    # Size for marker dots on the lines (change this to control dot size)
    marker_size = 3
    marker_size_highlight = 7

    ax.plot(
        df_Obs_IGCC.index,
        df_Obs_IGCC['GMST'],
        color=var_colours['Obs'],
        linewidth=1.2,
        alpha=1.0,
        marker='o',
        markersize=marker_size,
        label='Observed warming'
        )

    ax.plot(
        dict_updates_SR15['Multi-method mean'].index,
        dict_updates_SR15['Multi-method mean'][('Ant', '50')],
        color=var_colours['Ant'],
        linewidth=2.5,
        alpha=1.0,
        marker='o',
        markersize=marker_size,
        label='Human-induced warming'
    )

    # Plot the predictions for 2025
    ax.plot(
        [2024, 2025],
        [
            df_Obs_IGCC['GMST'].loc[2024],
            1.45  # FROM NICK DUNSTONE: MET OFFICE PREDICTION FOR 2025
         ],
        color=var_colours['Obs'],
        linewidth=1.2,
        linestyle='--',
        # Make marker empty circle
        marker='o',
        markersize=marker_size,
        markerfacecolor='none',
        markeredgecolor=var_colours['Obs'],
        markeredgewidth=2
    )
    ax.plot(
        [2024, 2025],
        [dict_updates_SR15['Multi-method mean'][('Ant', '50')].loc[2024],
         dict_updates_SR15['Multi-method mean'][('Ant', '50')].loc[2024] + 0.027],
        color=var_colours['Ant'],
        linewidth=1.2,
        linestyle='--',
        # Make marker empty circle
        marker='o',
        markersize=marker_size,
        markerfacecolor='none',
        markeredgecolor=var_colours['Ant'],
        markeredgewidth=2
    )

    ax.scatter(
        2024, dict_updates_SR15['Multi-method mean'][('Ant', '50')].loc[2024],
        color=var_colours['Ant'],
        s=marker_size_highlight*8,
        edgecolor=var_colours['Ant'],
        marker='o',
        lw=0
    )
    ax.scatter(
        2024, df_Obs_IGCC['GMST'].loc[2024],
        color=var_colours['Obs'],
        s=marker_size_highlight*8,
        edgecolor=var_colours['Obs'],
        marker='o',
        lw=0
    )

    print(dict_updates_ts['Multi-method mean'][('Ant', '50')].loc[2024])
    print(dict_updates_SR15['Multi-method mean'][('Ant', '50')].loc[2024])

    # Key dates/events
    key_dates = {
        2015: 'Paris Agreement',
        2010: 'Cancún Agreement',
        2021: 'IPCC AR6 WGI',
        # 2014: 'IPCC AR5',
        # 2018: 'IPCC SR1.5',
        # 2007: 'IPCC AR4',
        # 2001: 'IPCC TAR',
        2025: 'COP30 Belém',
        # 2025: r"$\bf{" + "COP30 Belém" + "}$"
        # 1992: 'Rio Summit: UNFCCC'
    }
    for year, label in key_dates.items():
        # Add an arrow from the end of the label to the SR1.5 datapoint
        # ax.annotate(
        #     '',
        #     xy=(year,
        #         dict_updates_SR15['Multi-method mean'][('Ant', '50')].loc[year]),
        #     xycoords='data',
        #     xytext=(year, 1.4), textcoords='data',
        #     arrowprops=dict(arrowstyle='->', color='grey', lw=1)
        # )

        ax.axvline(
            x=year,
            color='#cfd1d0', linestyle='-', linewidth=0.5, zorder=0
            )
        ax.text(
            x=year - 0.6,
            y=0.5 + 0.015,
            s=label,
            rotation=90, verticalalignment='bottom', color='#6F6F6F',
            fontsize=12
            )

    # Plot the decade average warming from df_headlines
    ax.plot(
        df_Obs_IGCC.index[-10:],
        [df_headlines[('Ant', '50')].loc['2015\N{EN Dash}2024'] for _ in range(10)],
        color=var_colours['Ant decade'], alpha=0.5, linestyle='-',
        lw=0.5
    )
    ax.fill_between(
        df_Obs_IGCC.index[-10:],
        [df_headlines[('Ant', '50')].loc['2015\N{EN Dash}2024'] for _ in range(10)],
        dict_updates_SR15['Multi-method mean'][('Ant', '50')].iloc[-10:],
        color=var_colours['Ant decade'], alpha=0.05, lw=0
    )
    ax.scatter(
        2015+4.5,
        df_headlines[('Ant', '50')].loc['2015\N{EN Dash}2024'],
        color=var_colours['Ant decade'],
        s=marker_size_highlight*8,
        edgecolor=var_colours['Ant decade'],
        marker='o',
        alpha=0.5,
        lw=0
    )

    # # Do the same for the observations
    # ax.plot(
    #     df_Obs_IGCC.index[-10:],
    #     [1.24 for _ in range(10)],
    #     color=var_colours['Obs decade'], alpha=1, linestyle='--'  
    # )
    # ax.fill_between(
    #     df_Obs_IGCC.index[-10:],
    #     [1.24 for _ in range(10)],
    #     df_Obs_IGCC['GMST'].iloc[-10:],
    #     color=var_colours['Obs decade'], alpha=0.08, lw=0
    # )
    # plt.scatter(
    #     2015+4.5,
    #     1.24,
    #     color=var_colours['Obs decade'],
    #     s=marker_size*8,
    #     edgecolor=var_colours['Obs decade'],
    #     # linewidth=1,
    #     marker='o'
    # )

    # for year in df_Obs_IGCC.index[-10:]:
    #     ax.plot(
    #         [year-0.01, year-0.01],
    #         [
    #             1.24,
    #             df_Obs_IGCC['GMST'].loc[year]
    #         ],
    #         color=var_colours['Obs'], alpha=0.1
    #     )


    # Annotate the assessed observed end value (no uncertainty) just off the
    # right-hand side of the plot. Use a small horizontal offset so the text
    # sits outside the plotting area with an arrow pointing to the final point.
    arrow_width = 1.2
    arrow_alpha = 1.0
    arrow_offset = 0.1
    arrow_head_size = 20
    arrow_head_style = '->'
    text_offset = 3.5  # years to shift the annotation text to the right
    
    obs_year = df_Obs_IGCC.index[-1]
    obs_value = df_Obs_IGCC['GMST'].iloc[-1]
    obs_label = (
        'Observed\n' +
        # r"$\bf{" + "2024" + "}$" + "\n" +
        '2024\n' +
        # f"\n{obs_value:.2f} °C"
        r"$\bf{" + f"{obs_value:.2f}" + "}$ °C"
    )
    ax.annotate(
        obs_label,
        xy=(obs_year + arrow_offset, obs_value),
        xytext=(obs_year + text_offset, obs_value),
        color=var_colours['Obs'],
        fontweight='regular',
        arrowprops=dict(
            color=var_colours['Obs'],
            arrowstyle=arrow_head_style,
            mutation_scale=arrow_head_size,
            connectionstyle="arc3,rad=0.0",
            linewidth=arrow_width,
            alpha=arrow_alpha
        ),
        verticalalignment='center'
    )

    # Do the same for the end of the Ant warming
    human_year = 2024
    human_value = dict_updates_SR15['Multi-method mean'][('Ant', '50')].loc[human_year]
    human_label_value = df_headlines[('Ant', '50')].loc['2024 (SR15 definition)']
    human_label = (
        "Human-induced\n" +
        # r"$\bf{" + "2024" + "}$" + "\n" +
        '2024\n' +
        r"$\bf{" + f"{human_label_value:.2f}" + "}$ °C"
        )

    ax.annotate(
        human_label,
        xy=(human_year + arrow_offset, human_value),
        xytext=(human_year + text_offset, human_value),
        color=var_colours['Ant'],
        fontweight='regular',
        arrowprops=dict(
            color=var_colours['Ant'],
            arrowstyle=arrow_head_style,
            mutation_scale=arrow_head_size,
            connectionstyle="arc3,rad=0.0",
            linewidth=arrow_width,
            alpha=arrow_alpha
        ),
        verticalalignment='center'
    )

    # # Add an annotation for the decade average warming level, with the arrow
    # # pointing to the decade average line, halfway along the decade
    # decade_avg_year = 2015 + 4.5
    # decade_avg_value = 1.24
    # decade_avg_label = (
    #     # "\n\n" + r"$\bf{" + 'Observed' + "}$" +
    #     "\n\nObserved average" +
    #     "\n" + 
    #     # r"$\bf{" + '2015\N{EN DASH}2024' + "}$" + "\n" +
    #     '2015\N{EN DASH}2024\n' +
    #     r"$\bf{" + f"{decade_avg_value:.2f}" + "}$ °C"
    # )
    # ax.annotate(
    #     decade_avg_label,
    #     xy=(2024 + arrow_offset, decade_avg_value),
    #     # xytext=(decade_avg_year + 4.5 + text_offset, decade_avg_value-0.2),
    #     xytext=(2024 + text_offset, decade_avg_value),
    #     color=var_colours['Obs decade'],
    #     fontweight='regular',
    #     arrowprops=dict(
    #         color=var_colours['Obs decade'],
    #         arrowstyle=arrow_head_style,
    #         mutation_scale=arrow_head_size/1.5,
    #         # connectionstyle="angle,angleA=0,angleB=-80,rad=5",
    #         connectionstyle="arc3,rad=0.0",
    #         linewidth=arrow_width/1.5,
    #         alpha=arrow_alpha,
    #         linestyle='--'
    #     ),
    #     verticalalignment='center'
    # )

    # Add an annotation for the decade average HIW level, with the arrow
    # pointing to the decade average line, halfway along the decade
    decade_avg_year = 2015 + 4.5
    decade_avg_value = df_headlines[('Ant', '50')].loc['2015\N{EN Dash}2024']
    decade_avg_label = (
        # "\n\n" + r"$\bf{" + 'Observed' + "}$" +
        "\n\nHuman-induced" +
        "\n" + 
        # r"$\bf{" + '2015\N{EN DASH}2024' + "}$" + "\n" +
        '2015\N{EN DASH}2024 average\n' +
        r"$\bf{" + f"{decade_avg_value:.2f}" + "}$ °C"
    )
    ax.annotate(
        decade_avg_label,
        xy=(2024 + arrow_offset, decade_avg_value),
        # xytext=(decade_avg_year + 4.5 + text_offset, decade_avg_value-0.2),
        xytext=(2024 + text_offset, decade_avg_value),
        color=var_colours['Ant decade'],
        fontweight='regular',
        arrowprops=dict(
            color=var_colours['Ant decade'],
            arrowstyle=arrow_head_style,
            mutation_scale=arrow_head_size,
            # connectionstyle="angle,angleA=0,angleB=-80,rad=5",
            connectionstyle="arc3,rad=0.0",
            linewidth=arrow_width,
            alpha=arrow_alpha/2,
            linestyle='-'
        ),
        verticalalignment='center'
    )

    ax.set_ylabel(
            'Global Surface Temperature Increase\n'
            # + 'relative to 1850\N{EN DASH}1900 baseline (°C)'
            )
    # Change the size of the y axis label
    ax.yaxis.label.set_size(16)

    ax.set_ylim(0.5, 1.55)
    ticks = list(np.arange(1990, 2025, 5))
    ticks.append(2025)
    ax.set_xticks(ticks, ticks)
    ax.set_yticks([0.5, 1.0, 1.5], ['0.5 °C ', '1.0 °C ', '1.5 °C '])
    # Set the ytick color for 1.5C to the Ant colour
    plt.gca().get_yticklabels()[2].set_color(var_colours['Ant'])
    # Make the 1.5°C ytick label bold text formatting
    # plt.gca().get_yticklabels()[2].set_fontweight('bold')
    # Remove the horizontal axis gridline at 1.5°C
    # ax.get_ygridlines()[2].set_visible(False)
    ax.get_ygridlines()[2].set_color(var_colours['Ant'])

    ax.set_xlim(1999.5, 2025.2)

    # gr.overall_legend(fig, 'lower center', 2)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))
    fig.savefig('COP30-figure.png', dpi=300)
