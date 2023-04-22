"""Script to produce plots for gwi.py."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
from gwi import moving_average


matplotlib.rcParams.update(
    {'font.size': 11,
     'font.family': 'Roboto',
     'font.weight': 'light',
     'axes.linewidth': 0.5, 'axes.titleweight': 'regular',
     'axes.grid': True, 'grid.linewidth': 0.5,
     'grid.color': 'gainsboro',
     'figure.dpi': 200, 'figure.figsize': (15, 10),
     'figure.titlesize': 17,
     'figure.titleweight': 'light',
     'legend.frameon': False}
)
# Fontweights~: light, regular, normal,


def overall_legend(fig, loc, ncol, nrow=False):
    """Add a clean legend to a figure with multiple subplots."""
    handles, labels = [], []

    for ax in fig.axes:
        hs, ls = ax.get_legend_handles_labels()
        if hs not in handles:
            handles.extend(hs)
        if ls not in labels:
            labels.extend(ls)

    by_label = dict(zip(labels, handles))

    fig.legend(by_label.values(), by_label.keys(),
               loc=loc, ncol=ncol)

    ## rect = (left, bottom, right, top)
    # if loc == 'right':
    #     fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.94))
    # elif loc == 'lower center':
    #     # fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.94))
    #     fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))


def running_mean_internal_variability(
    timeframes, df_temp_PiC, temp_Obs_IV
):
    """Plot running means of internal variability ensemble."""
    lims = [0.6, 0.4, 0.15]
    for t in range(len(timeframes)):
        axA = plt.subplot2grid(
            shape=(len(timeframes), 4), loc=(t, 0), rowspan=1, colspan=3)
        axB = plt.subplot2grid(
            shape=(len(timeframes), 4), loc=(t, 3), rowspan=1, colspan=1)

        axA.set_ylim(-lims[t], lims[t])
        cut_beg = timeframes[t]//2
        cut_end = timeframes[t]-1-timeframes[t]//2
        _time = df_temp_PiC.index.to_numpy()

        if timeframes[t] == 1:
            _time_sliced = _time[:]
        else:
            _time_sliced = _time[cut_beg: -cut_end]

        for ens in df_temp_PiC.columns:
            _data = moving_average(df_temp_PiC[ens], timeframes[t])
            axA.plot(_time_sliced, _data, label='CMIP6 PiControl',
                     color='gray', alpha=0.3)

            density = ss.gaussian_kde(_data)
            x = np.linspace(axA.get_ylim()[0], axA.get_ylim()[1], 50)
            y = density(x)
            axB.plot(y, x, color='gray', alpha=0.3)

        _data = moving_average(temp_Obs_IV, timeframes[t])
        axA.plot(_time_sliced, _data, label='HadCRUT5 median',
                 color='xkcd:teal', alpha=1)
        density = ss.gaussian_kde(_data)
        x = np.linspace(axA.get_ylim()[0], axA.get_ylim()[1], 50)
        y = density(x)
        axB.plot(y, x, color='xkcd:teal', alpha=1)

        # axes[i].set_ylim([-0.6, +0.6])
        axA.set_xlim(1845, 2030)
        axB.set_ylim(axA.get_ylim())
        # axB.get_yaxis().set_visible(False)
        axB.get_xaxis().set_visible(False)
        axA.set_ylabel('Internal Variability (°C) \n'+
                       f'({timeframes[t]}-year moving mean)')


def plot_internal_variability_sample(
        ax1, ax2,
        df_temp_PiC, df_temp_Obs, temp_Obs_IV, sigmas, sigmas_all):
    """Plot internal variability: sigma timeseries and box plot."""
    for p in range(len(sigmas)):
        ax1.fill_between(
            df_temp_PiC.index,
            df_temp_PiC.quantile(q=sigmas_all[p]/100, axis=1),
            df_temp_PiC.quantile(q=sigmas_all[-(p+2)]/100, axis=1),
            color='gray', alpha=0.2)
        ax2.fill_between(
            x=[3, 4],
            y1=df_temp_PiC.quantile(
                q=sigmas_all[p]/100, axis=1).mean()*np.ones(2),
            y2=df_temp_PiC.quantile(
                q=sigmas_all[-(p+2)]/100, axis=1).mean()*np.ones(2),
            color='gray', alpha=0.2)

    ax1.plot(df_temp_PiC.index, df_temp_PiC.quantile(q=0.5, axis=1),
             color='gray', alpha=0.7, label='CMIP6 piControl')
    ax2.plot([3, 4], df_temp_PiC.quantile(q=0.5, axis=1).mean()*np.ones(2),
             color='gray', alpha=0.7)

    # HadCRUT5 median
    ax1.plot(df_temp_Obs.index,
             (np.percentile(temp_Obs_IV, sigmas_all) *
              np.ones((len(df_temp_Obs.index), len(sigmas_all)))),
             color='xkcd:teal', alpha=0.7, ls='--')
    ax1.plot(df_temp_Obs.index, temp_Obs_IV,
             color='xkcd:teal', alpha=0.7, label='HadCRUT5 median')
    for p in sigmas:
        ax2.fill_between(x=[1, 2],
                         y1=np.percentile(temp_Obs_IV, p[0])*np.ones(2),
                         y2=np.percentile(temp_Obs_IV, p[1])*np.ones(2),
                         color='xkcd:teal', alpha=0.2)
    ax2.plot([1, 2], np.percentile(temp_Obs_IV, 50)*np.ones(2),
             color='xkcd:teal', alpha=0.7)

    ax1.set_ylabel('Internal Variability (°C)')
    ax1.set_ylim(-0.6, 0.6)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(0, 5)
    ax2.get_xaxis().set_visible(False)


def gwi_timeseries(ax, df_temp_Obs, df_temp_PiC, df_Results_ts,
                   plot_vars, plot_cols):
    """Plot the GWI timeseries for the given variables."""
    ax.set_ylabel('Warming Anomaly (°C)')
    fill_alpha = 0.2
    line_alpha = 0.7
    sigmas = df_Results_ts.columns.get_level_values('percentile').unique()
    # Plot the observations
    err_pos = (df_temp_Obs.quantile(q=0.95, axis=1) -
               df_temp_Obs.quantile(q=0.5, axis=1))
    err_neg = (df_temp_Obs.quantile(q=0.5, axis=1) -
               df_temp_Obs.quantile(q=0.05, axis=1))
    ax.errorbar(df_temp_Obs.index, df_temp_Obs.quantile(q=0.5, axis=1),
                yerr=(err_neg, err_pos),
                fmt='o', color=plot_cols['Obs'], ms=2.5, lw=1,
                label='Reference Temp: HadCRUT5')
    for s in range(len(sigmas)//2):
        # Plot the PiControl ensemble
        ax.fill_between(
            df_temp_PiC.index,
            df_temp_PiC.quantile(q=float(sigmas[s])/100, axis=1),
            df_temp_PiC.quantile(q=float(sigmas[-(s+2)])/100, axis=1),
            color=plot_cols['PiC'], alpha=fill_alpha)
        ax.plot(df_temp_PiC.index, df_temp_PiC.quantile(q=0.5, axis=1),
                color=plot_cols['PiC'], alpha=line_alpha,
                label='CMIP6 piControl')

        # Plot the GWI timeseries
        for var in plot_vars:
            ax.fill_between(
                df_Results_ts.index,
                df_Results_ts.loc[:, (var, sigmas[s])].values,
                df_Results_ts.loc[:, (var, sigmas[-(s+2)])].values,
                color=plot_cols[var], alpha=fill_alpha)
            ax.plot(df_Results_ts.index,
                    df_Results_ts.loc[:, (var, sigmas[-1])].values,
                    color=plot_cols[var], alpha=line_alpha, label=var)


def gwi_residuals(ax, df_Results_ts):
    """Plot the regression residuals of the GWI timeseries."""
    ax.set_ylabel('Regression Residuals (°C)')
    fill_alpha = 0.2
    line_alpha = 0.7
    sigmas = df_Results_ts.columns.get_level_values('percentile').unique()
    for s in range(len(sigmas)//2):
        ax.fill_between(
            df_Results_ts.index,
            df_Results_ts.loc[:, ('Res', sigmas[s])].values,
            df_Results_ts.loc[:, ('Res', sigmas[-(s+2)])].values,
            color='gray', alpha=fill_alpha)
        ax.plot(df_Results_ts.index,
                df_Results_ts.loc[:, ('Res', sigmas[-1])].values,
                label='HadCRUT5 Residuals',
                color='gray', alpha=line_alpha)
    ax.plot(df_Results_ts.index, np.zeros(len(df_Results_ts.index)),
            # label='Reference Temp: HadCRUT5',
            color='xkcd:magenta', alpha=0.7,
            )


def gwi_tot_vs_ant(ax, df_Results_ts):
    """Plot the TOT vs. ANT timeseries."""
    ax.plot([-0.2, 1.5], [-0.2, 1.5], color='gray', alpha=0.7)
    ax.plot(df_Results_ts.loc[:, ('Ant', '50')].values,
            df_Results_ts.loc[:, ('Tot', '50')].values,
            color='xkcd:teal', alpha=0.7,)
    ax.set_xlabel('Ant')
    ax.set_ylabel('TOT')
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.5)


def gwi_pdf(ax):
    """Plot the GWI PDFs for the given variables in end year."""
    # NOTE THAT THIS IS THE OLD CODE FOR THE PDF PLOT USING NUMPY ARRAYS
    # INSTEAD OF PANDAS DATAFRAMES FOR THE RESULTS.
    # Distributions ###############################################################
    for c in range(len(gwi_plot_names)):
        if gwi_plot_names[c] in {'Ant', 'GHG', 'Nat', 'OHF'}:
            # binwidth = 0.01
            # bins = np.arange(np.min(temp_Att_Results[-1, gwi_plot_components[i], :]),
            #                  np.max(temp_Att_Results[-1, gwi_plot_components[i], :]) + binwidth,
            #                  binwidth)
            # ax2.hist(temp_Att_Results[-1, gwi_plot_components[i], :], bins=bins,
            #          density=True, orientation='horizontal',
            #          color=gwi_plot_colours[i],
            #          alpha=0.3
            #          )
            density = ss.gaussian_kde(
                temp_Att_Results[-1, gwi_plot_components[c], :])
            x = np.linspace(
                temp_Att_Results[-1, gwi_plot_components[c], :].min(),
                temp_Att_Results[-1, gwi_plot_components[c], :].max(),
                100)
            y = density(x)
            # ax2.plot(y, x, color=gwi_plot_colours[i], alpha=0.7)
            ax.fill_betweenx(x, np.zeros(len(y)), y,
                            color=gwi_plot_colours[c], alpha=0.3)

    # Add PiC PDF
    density = ss.gaussian_kde(
                temp_PiC_unique[-1, :])
    x = np.linspace(
        temp_PiC_unique[-1, :].min(),
        temp_PiC_unique[-1, :].max(),
        100)
    y = density(x)
    ax.fill_betweenx(x, np.zeros(len(y)), y,
                        color='gray', alpha=0.3)

    # bins = np.arange(np.min(temp_Att_Results[-1, -5, :]),
    #                  np.max(temp_Att_Results[-1, -5, :]) + binwidth,
    #                  binwidth)
    # ax2.hist(temp_Att_Results[-1, -5, :], bins=bins,
    #          density=True, orientation='horizontal',
    #          color='pink', alpha=0.3
    #          )

    # bins = np.arange(np.min(temp_Att_Results[-1, -4, :]),
    #                  np.max(temp_Att_Results[-1, -4, :]) + binwidth,
    #                  binwidth)
    # ax2.hist(temp_Att_Results[-1, -4, :], bins=bins,
    #          density=True, orientation='horizontal',
    #          color='gray', alpha=0.3
    #          )


def Fig_SPM2_validation_plot(ax, period, variables, dict_updates_hl, source_cols):
    """Plot the SPM2 figure."""
    bar_width = (1.0-0.4)/len(dict_updates_hl.keys())
    methods = sorted(list(dict_updates_hl.keys()))
    for var in variables:
        for source in methods:
            med = dict_updates_hl[source].loc[period, (var, '50')]
            neg = med - dict_updates_hl[source].loc[period, (var, '5')]
            pos = dict_updates_hl[source].loc[period, (var, '95')] - med

            bar = ax.bar(variables.index(var) + bar_width*methods.index(source),
                         med,
                         yerr=([neg], [pos]),
                         label=source,
                         width=bar_width, color=source_cols[source], alpha=1.0)
            # ax.errorbar(variables.index(var) + bar_width*methods.index(source),
            #             med, yerr=([neg], [pos]),
                        # fmt='none', color='black')
            # ax.bar_label(bar, padding=10, fmt='%.2f')
            med_r = np.around(med, decimals=2)
            pos_r = np.around(pos, decimals=2)
            neg_r = np.around(neg, decimals=2)
            str_Result = r'${%s}^{+{%s}}_{-{%s}}$' % (med_r, pos_r, neg_r)
            # if med >= 0:
            #     # Automatically place the label above the bar
            #     padding = 10 + 15*(methods.index(source))
            # else:
            #     # Manually set the padding for negative values to put them
            #     # above the x axis
            #     padding = -80 - 15*(methods.index(source))
            ax.bar_label(bar, labels=[str_Result], padding=10 + 15*(methods.index(source)))
            ax.set_ylim(-1.5, 2.5)

    ax.set_xticks(np.arange(len(variables)), variables)
    ax.set_ylabel(f'Contributions to {period} warming relative to 1850-1900')
    ax.xaxis.grid(False)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def Fig_SPM2_results_plot(ax, periods, variables, dict_updates_hl, period_cols):
    """Plot the SPM2 figure."""
    bar_width = (1.0-0.4)/(len(dict_updates_hl.keys())*len(periods))
    methods = sorted(list(dict_updates_hl.keys()))
    labels = {
        '2010-2019': '2010-2019 (AR6 definition)',
        '2013-2022': '2013-2022 (AR6 definition)',
        '2022 (SR15 definition)': '2022 (SR15 definition)',
        '2017 (SR15 definition)': '2017 (SR15 definition)',
        '2017': '2017',
        '2022': '2022',
    }
    for var in variables:
        for period in periods:
            for source in methods:
                med = dict_updates_hl[source].loc[period, (var, '50')]
                neg = med - dict_updates_hl[source].loc[period, (var, '5')]
                pos = dict_updates_hl[source].loc[period, (var, '95')] - med
                bar_loc_offset = bar_width * (methods.index(source) +
                                              periods.index(period))
                bar = ax.bar(variables.index(var) + bar_loc_offset,
                             med,
                             yerr=([neg], [pos]),
                             label=labels[period],
                             width=bar_width,
                             color=period_cols[period],
                             alpha=0.9)
                med_r = np.around(med, decimals=2)
                pos_r = np.around(pos, decimals=2)
                neg_r = np.around(neg, decimals=2)
                str_Result = r'${%s}^{+{%s}}_{-{%s}}$' % (med_r, pos_r, neg_r)
                if med > 0:
                    # Automatically place the label above the bar
                    padding = 10 + 15*(periods.index(period))
                else:
                    # Manually set the padding for negative values to put them
                    # above the x axis
                    padding = -80 - 15*(periods.index(period))
                ax.bar_label(bar, labels=[str_Result], padding=padding)

    # remove the vertical guidelines in the plot
    ax.xaxis.grid(False)
    # add grid line for x axis
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xticks(np.arange(len(variables)), variables)
    ax.set_ylabel(f'Warming contribution (°C)')
    ax.set_ylim(-0.75, 2.0)


def Fig_3_8_validation_plot(
        ax, variables, dict_AR6_hl, dict_updates_hl, source_markers, var_colours):
    """Plot AR6 WG1 Ch.3 Fig.3.8"""

    bar_width = 0.4

    cycles = [dict_AR6_hl, dict_updates_hl]
    for var in variables:
        for cycle in cycles:
            # Plot the multi-method assessed results for the 2010-2019 period
            med_assess = cycle['Assessment'].loc['2010-2019', (var, '50')]
            min_assess = cycle['Assessment'].loc['2010-2019', (var, '5')]
            max_assess = cycle['Assessment'].loc['2010-2019', (var, '95')]

            ax.fill_between(
                [variables.index(var) + 0.45*cycles.index(cycle),
                 variables.index(var) + 0.45*cycles.index(cycle) + bar_width],
                min_assess, max_assess,
                color=(var_colours['Obs']
                       if (var == 'Tot' and cycles.index(cycle)==0)
                       else var_colours[var]),
                alpha=0.5
                # label=var
                )
            ls = (':' if (cycles.index(cycle) == 0 and
                           var in ['GHG', 'OHF', 'Nat'])
                  else '-')
            ax.plot(
                [variables.index(var) + 0.45*cycles.index(cycle),
                 variables.index(var) + 0.45*cycles.index(cycle) + bar_width],
                [med_assess, med_assess],
                color=(var_colours['Obs']
                       if (var == 'Tot' and cycles.index(cycle)==0)
                       else var_colours[var]),
                # Depict that the nest estimates are a new inclusion.
                ls=ls,
                lw=2)

            # Write str_Result in the middle of the plot
            str_Result = r'${%s}^{{%s}}_{{%s}}$' % \
                (med_assess, max_assess, min_assess)
            if (var == 'Tot' and cycles.index(cycle) == 0):
                str_Result = str_Result + '\n(Obs)'
            ax.text(
                (variables.index(var) + cycles.index(cycle) * 0.45
                 + bar_width / 2),
                0.6,
                str_Result,
                ha='center', va='center', color='black')

            # Plot the individual methods' results
            methods = sorted(list(cycle.keys()))[::-1]
            methods.remove('Assessment')
            for method in methods:
                if var in cycle[method].columns.get_level_values('variable'):
                    med_meth = cycle[method].loc['2010-2019', (var, '50')]
                    min_meth = cycle[method].loc['2010-2019', (var, '5')]
                    max_meth = cycle[method].loc['2010-2019', (var, '95')]
                    test = ax.errorbar(
                        variables.index(var) + 0.45*cycles.index(cycle) + 0.1 + methods.index(method)*(bar_width-0.2)/(len(methods)-1),
                        ([med_meth]),
                        yerr=([med_meth-min_meth], [max_meth-med_meth]),
                        color=var_colours[var], ms=7, lw=2,
                        label=method,
                        fmt=source_markers[method],
                        )
                    # Chris Smith's Chapter 7 results aren't included in the
                    # multimethod assessment, so plot with dashed line
                    if method == 'Smith':
                        test[-1][0].set_linestyle('--')

    # set the y axis label
    ax.set_ylabel('Attributable change in surface temperature\n' +
                  '2010-2019 vs 1850-1900 (°C)')
    # Remove the ticks from the x axis
    ax.xaxis.grid(False)
    # Set new custom x ticks
    ax.set_xticks(np.arange(len(variables)) + 0.425)
    # Set the x tick labels
    ax.set_xticklabels(variables)
    # Create a title for the plot
    ax.set_title(
        'Results from AR6 WG1 Ch.3 (left) vs Repeat using updates (right)')

    # Plot a middle-line
    ax.axhline(y=0, color='gray', linestyle='-',
            #    linewidth=0.5, alpha=0.7
               )
