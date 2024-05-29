"""Script to produce plots for gwi.py."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
# from src.definitions import moving_average
import sys

font_family = 'Roboto'

matplotlib.rcParams.update({
    # General figure
    'figure.dpi': 300,
    'figure.figsize': (15, 10),
    'figure.titlesize': 17,
    'figure.titleweight': 'light',
    'legend.frameon': False,
    # General fonts
    'font.family':  font_family,
    'font.weight': 'light',
    'font.size': 11,
    'pdf.fonttype': 42,  # Switch from default 3 to 42 to use TrueType fonts
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
    'axes.linewidth': 0.5,
    'axes.facecolor': '#f9f8f7',  # AR6 at 50% opacity.
    # 'axes.facecolor': 'white',
    # Axis labels
    'axes.titleweight': 'regular',
    # 'axes.labelcolor': 'gray',
    # Axis grid
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#cfd1d0',  # AR6
    'grid.linewidth': 0.5,
    # 'axes.axisbelow': True,
    # Axis  ticks
    'ytick.major.size': 0,
    'ytick.major.width': 0,
    # 'ytick.color': 'gray',
})
# Fontweights~: light, regular, normal,


def overall_legend(fig, loc, ncol, nrow=False, reorder=None):
    """Add a clean legend to a figure with multiple subplots."""
    handles, labels = [], []

    for ax in fig.axes:
        hs, ls = ax.get_legend_handles_labels()
        if hs not in handles:
            handles.extend(hs)
        if ls not in labels:
            labels.extend(ls)

    by_label = dict(zip(labels, handles))
    if reorder is None:
        order = np.arange(len(by_label))
    else:
        order = reorder

    fig.legend([list(by_label.values())[i] for i in order],
               [list(by_label.keys())[i] for i in order],
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
            _data = np.convolve(
                df_temp_PiC[ens],
                np.ones(timeframes[t]),
                'valid') / timeframes[t]
            # _data = moving_average(df_temp_PiC[ens], timeframes[t])
            axA.plot(_time_sliced, _data, label='CMIP6 PiControl',
                     color='gray', alpha=0.3)

            density = ss.gaussian_kde(_data)
            x = np.linspace(axA.get_ylim()[0], axA.get_ylim()[1], 50)
            y = density(x)
            axB.plot(y, x, color='gray', alpha=0.3)

        # _data = moving_average(temp_Obs_IV, timeframes[t])
        _data = np.convolve(
            temp_Obs_IV,
            np.ones(timeframes[t]),
            'valid') / timeframes[t]
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
                   plot_vars, plot_cols, sigmas='all', labels=True):
    """Plot the GWI timeseries for the given variables."""
    ax.set_ylabel(
        'Attributable change in surface temperature since 1850\N{EN DASH}1900 (°C)'
        )
    fill_alpha = 0.25
    line_alpha = 0.7
    if sigmas == 'all':
        sigmas = df_Results_ts.columns.get_level_values('percentile').unique()
    # Shade the pre-industrial period
    ax.fill_between([1850, 1900], [-5, -5], [+5, +5],
                    color='#f4f2f1')

    # Plot the observations
    err_pos = (df_temp_Obs.quantile(q=0.95, axis=1) -
               df_temp_Obs.quantile(q=0.5, axis=1))
    err_neg = (df_temp_Obs.quantile(q=0.5, axis=1) -
               df_temp_Obs.quantile(q=0.05, axis=1))
    ax.errorbar(df_temp_Obs.index, df_temp_Obs.quantile(q=0.5, axis=1),
                yerr=(err_neg, err_pos),
                fmt='o', color=plot_cols['Obs'], ms=2.5, lw=1,
                label=labels*'Reference Temp: HadCRUT5')
    if df_temp_PiC is not None:
        if len(sigmas) > 1:
            for s in range(len(sigmas)//2):
                # Plot the PiControl ensemble
                ax.fill_between(
                        df_temp_PiC.index,
                        df_temp_PiC.quantile(q=float(sigmas[s])/100, axis=1),
                        df_temp_PiC.quantile(q=float(sigmas[-(s+2)])/100, axis=1),
                        color=plot_cols['PiC'], alpha=fill_alpha, linewidth=0.0)
        ax.plot(df_temp_PiC.index, df_temp_PiC.quantile(q=0.5, axis=1),
                color=plot_cols['PiC'], alpha=line_alpha,
                label=labels*'CMIP6 piControl')

    for s in range(max(len(sigmas)//2, 1)):  # max to enable 50% only
        # Plot the GWI timeseries
        for var in plot_vars:

            # Because ROF (Gillett) method has different percentile results
            # available for different variables (ie Tot only has 50th), check
            # for each variable first whether to plot plume.
            var_sigmas = df_Results_ts.iloc[\
                :, df_Results_ts.columns.get_level_values('variable') == var
                ].columns.get_level_values('percentile').unique()
            if len(var_sigmas) > 1:
                ax.fill_between(
                    df_Results_ts.index,
                    df_Results_ts.loc[:, (var, sigmas[s])].values,
                    df_Results_ts.loc[:, (var, sigmas[-(s+2)])].values,
                    color=plot_cols[var], alpha=fill_alpha, linewidth=0.0)
            ax.plot(df_Results_ts.index,
                    df_Results_ts.loc[:, (var, sigmas[-1])].values,
                    color=plot_cols[var], alpha=line_alpha, label=labels*var)

    ax.set_xticks([1850, 1900, 1950, 2000, df_temp_Obs.index[-1]],
                  [1850, 1900, 1950, 2000, df_temp_Obs.index[-1]])


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
            ax.bar_label(
                bar,
                labels=[str_Result],
                padding=10 + 15*(methods.index(source)))
            ax.set_ylim(-1.5, 2.5)

    ax.set_xticks(np.arange(len(variables)), variables)
    ax.set_ylabel(f'Contributions to {period} warming relative to 1850\N{EN DASH}1900')
    ax.xaxis.grid(False)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def Fig_3_8_validation_plot(
        ax, variables, period,
        dict_IPCC_hl, dict_updates_hl,
        dict_IPCC_Obs_hl, dict_updates_Obs_hl,
        source_markers, var_colours, labels):
    """Plot AR6 WG1 Ch.3 Fig.3.8"""

    bar_width = 0.4

    # Plot observations
    if period == '2010\N{EN DASH}2019':
        # Plot the IPCC-quoted results for 2010\N{EN DASH}2019 observations
        med_Obs = dict_IPCC_Obs_hl['Assessment'].loc[period, ('Obs', '50')]
        min_Obs = dict_IPCC_Obs_hl['Assessment'].loc[period, ('Obs', '5')]
        max_Obs = dict_IPCC_Obs_hl['Assessment'].loc[period, ('Obs', '95')]

        ax.fill_between(
            [-1 + 0 * 0.45, -1 + 0 + bar_width], min_Obs, max_Obs,
            color=var_colours['Obs'], alpha=0.4, linewidth=0)
        ax.plot(
            [-1 + 0 * 0.45, -1 + 0 + bar_width], [med_Obs, med_Obs],
            color=var_colours['Obs'], lw=2)
        str_Result = r'${%s}^{{%s}}_{{%s}}$' % (med_Obs, max_Obs, min_Obs)
        ax.text(
            (-1 + 0 * 0.45 + bar_width / 2), 0.6,
            str_Result,
            ha='center', va='center', color='black')
        
        # Plot the updated re-assessment for 2010-2019 observations
        med_Obs = dict_updates_Obs_hl['Assessment'].loc[period, ('Obs', '50')]
        min_Obs = dict_updates_Obs_hl['Assessment'].loc[period, ('Obs', '5')]
        max_Obs = dict_updates_Obs_hl['Assessment'].loc[period, ('Obs', '95')]
        ax.fill_between(
            [-1 + 1 * 0.45, -1 + 0.45 + bar_width], min_Obs, max_Obs,
            color=var_colours['Obs'], alpha=0.6, linewidth=0)
        ax.plot(
            [-1 + 1 * 0.45, -1 + 0.45 + bar_width], [med_Obs, med_Obs],
            color=var_colours['Obs'], lw=2)
        str_Result = r'${%s}^{{%s}}_{{%s}}$' % (med_Obs, max_Obs, min_Obs)
        ax.text(
            (-1 + 1 * 0.45 + bar_width / 2), 0.6,
            str_Result,
            ha='center', va='center', color='black')

    cycles = [dict_IPCC_hl, dict_updates_hl]
    for var in variables:
        for cycle in cycles:
            # Plot the multi-method assessed results for the 2010-2019 period
            med_assess = cycle['Assessment'].loc[period, (var, '50')]
            min_assess = cycle['Assessment'].loc[period, (var, '5')]
            max_assess = cycle['Assessment'].loc[period, (var, '95')]

            ax.fill_between(
                [variables.index(var) + 0.45*cycles.index(cycle),
                 variables.index(var) + 0.45*cycles.index(cycle) + bar_width],
                min_assess, max_assess,
                color=(var_colours['Obs']
                       if (var == 'Tot' and cycles.index(cycle)==0)
                       else var_colours[var]),
                alpha=0.4 if cycles.index(cycle) == 0 else 0.6,
                linewidth=0,
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
                # Depict that the best estimates are a new inclusion.
                ls=ls,
                lw=2)

            # Write str_Result in the middle of the plot
            str_Result = r'${%s}^{{%s}}_{{%s}}$' % \
                (med_assess, max_assess, min_assess)
            ax.text(
                (variables.index(var)
                 + cycles.index(cycle) * 0.45
                 + bar_width / 2),
                0.6,
                str_Result,
                ha='center', va='center', color='black')

            # Plot the individual methods' results

            # A little footwork to plot the methods in the alphabetical order
            # of their method names, not authors...
            inv_labels = {val: key for key, val in labels.items()
                                   if key in cycle.keys()}
            usable_authors = set.intersection(set(labels.keys()),
                                              set(cycle.keys()))
            usable_methods = sorted([labels[m] for m in usable_authors])
            for method in usable_methods:
                author = inv_labels[method]
                vt = var in cycle[author].columns.get_level_values('variable')
                pt = period in cycle[author].index
                if vt and pt:
                    med_meth = cycle[author].loc[period, (var, '50')]
                    min_meth = cycle[author].loc[period, (var, '5')]
                    max_meth = cycle[author].loc[period, (var, '95')]
                    erbr = ax.errorbar(
                        (variables.index(var)
                         + 0.45*cycles.index(cycle)
                         + 0.1
                         + (usable_methods.index(method)
                            * (bar_width-0.2)
                            / (len(usable_methods)-1))
                         ),
                        ([med_meth]),
                        yerr=([med_meth-min_meth], [max_meth-med_meth]),
                        color=var_colours[var], ms=7, lw=2,
                        label=labels[author],
                        fmt=source_markers[author],
                        )
                    # Chris Smith's Chapter 7 results aren't included in the
                    # multimethod assessment, so plot with dashed line
                    if author == 'Smith':
                        erbr[-1][0].set_linestyle('--')

    # Remove the ticks from the x axis
    ax.xaxis.grid(False)
    # Set new custom x ticks
    ax.set_xticks(np.arange(len(variables) + 1) - 1 + 0.425)
    # Set the x tick labels
    ax.set_xticklabels(['Obs'] + variables)
    # Plot a middle-line
    ax.axhline(y=0, color='gray', linestyle='-',
            #    linewidth=0.5, alpha=0.7
               )

    # if period == '2010-2019':
    #     ax.set_title(f'AR6 WG1 Ch.3\n({period} warming)')
    # elif period == '2017':
    #     ax.set_title(f'SR1.5 Ch.1\n({period} warming)')


def Fig_SPM2_plot(
    ax, variables, periods,
    dict_IPCC_hl, dict_updates_hl,
    var_colours, var_names, labels,
    text_toggle
    ):
    """Plot AR6 WG1 SPM Fig.2-esque figure summarising assessed results."""
    # bar_width = (1.0-0.4)/(len(periods))
    bar_width = 0.3

    for var in variables:
        for period in periods:
            med = dict_updates_hl['Assessment'].loc[period, (var, '50')]
            neg = dict_updates_hl['Assessment'].loc[period, (var, '5')]
            pos = dict_updates_hl['Assessment'].loc[period, (var, '95')]

            bar_loc_offset = bar_width * periods.index(period)

            if var == 'Obs':
                colour = var_colours[var]
            elif med > 0:
                colour = '#d06e75'
                # colour = '#d7827e'
            elif med < 0:
                colour = '#7dbfd9'
                # colour = '#56949f'
            ax.bar(variables.index(var) + bar_loc_offset,
                   med,
                   yerr=([med-neg], [pos-med]),
                   error_kw=dict(lw=0.8, capsize=2, capthick=0.8),
                   width=bar_width,
                   color=colour,
                   alpha=1.0 if periods.index(period) == 1 else 0.7)
            if text_toggle:
                str_Result = r'${%s}^{{%s}}_{{%s}}$' % (med, pos, neg)
                ax.text(
                    variables.index(var) + bar_loc_offset,
                    -1.1,
                    str_Result,
                    ha='center', va='bottom', color='black',
                    rotation=90,
                    # fontsize=8
                    )
            if variables.index(var) == 0:
                ax.text(
                    variables.index(var) + bar_loc_offset,
                    0.05,
                    period,
                    ha='center', va='bottom', color='white',
                    rotation=90,
                    weight='regular'
                    # fontsize=8
                    )

    # add grid line for x axis
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    # Component labels
    tick_locs = np.arange(len(variables)) + (bar_width/2)*(len(periods)-1)
    # ax.text(tick_locs), -0.5,
    ax.set_xticks(tick_locs, [var_names[v] for v in variables], rotation=270,
                  weight='regular'
                  )


def definition_diagram(ax1, end_yr, df_headlines, df_temp_Obs, df_temp_Att,
                       var_colours):
    text_offset = 2
    rad = 5
    AR6_colour = '#67c1bf'
    SR15_colour = '#4f91cd'
    periods = {'single_period': str(end_yr),
               'trend_period': f'{end_yr} (SR15 definition)',
               'decade_period': f'{end_yr-9}\N{EN DASH}{end_yr}'}

    # Plot the observations ###################################################
    lower = df_temp_Obs.quantile(q=0.05, axis=1)
    upper = df_temp_Obs.quantile(q=0.95, axis=1)
    middle = df_temp_Obs.quantile(q=0.5, axis=1)
    err_neg = middle - lower
    err_pos = upper - middle
    ax1.errorbar(df_temp_Obs.index, middle,
                 yerr=(err_neg, err_pos),
                 fmt='o', color=var_colours['Obs'], ms=2.5, lw=1,
                 label='Reference Temp: HadCRUT5')
    value = (f"{middle[end_yr]:.2f} " +
             f"[{lower[end_yr]:.2f}\N{EN DASH}{upper[end_yr]:.2f}] °C")
    annotation = (r'$\bf{Observed \ single \ year}$' +
                  '\nHadCRUT5 reference' +
                  f'\n{end_yr} observation:\n{value}')
    ax1.annotate(
        annotation,
        xy=(df_temp_Obs.index[-1], middle[end_yr]),
        xytext=(df_temp_Obs.index[-1] + text_offset,
                middle[end_yr]),
        color=var_colours['Obs'],
        fontweight='regular',
        arrowprops=dict(
            color=var_colours['Obs'],
            arrowstyle='->',
            # Add a straight horizontal line between the xy and xytext using
            # connectionstyle=f"angle,angleA=0,angleB=0,rad={rad}"
            connectionstyle="arc3,rad=0.0"
            ),
        verticalalignment='center'
        )

    # Plot GWI timeseries #####################################################
    # Plot a line plot with scatter marks for Ant 50
    ax1.plot(df_temp_Obs.index, df_temp_Att['Ant', '50'],
             color=var_colours['Ant'], linestyle='-', linewidth=1)
    # Plot a scatter of the  'Ant', '50' GWI values
    ax1.scatter(x=df_temp_Obs.index,
                y=df_temp_Att['Ant', '50'],
                color=var_colours['Ant'], s=50,
                label='SR1.5 single year')
    # Plot the final value as single scatter point
    ax1.scatter(x=df_temp_Obs.index[-1],
                y=df_temp_Att.loc[end_yr, ('Ant', '50')],
                color=var_colours['Ant'], s=100,
                )
    # ax1.fill_between(
    #     df_temp_Obs.index, df_temp_Att['Ant', '5'], df_temp_Att['Ant', '95'],
    #     color=var_colours['Ant'], alpha=0.1, linewidth=0
    #     )
    value = (f"{df_headlines.loc[periods['single_period'], ('Ant', '50')]} " +
             f"[{df_headlines.loc[periods['single_period'], ('Ant', '5')]}" +
             "\N{EN DASH}" +
             f"{df_headlines.loc[periods['single_period'], ('Ant', '95')]}] " +
             "°C")
    annotation = (r'$\bf{SR1.5 \ single \ year}$' +
                  f'\n{end_yr} assessment:\n{value}')
    ax1.annotate(
        annotation,
        xy=(df_temp_Obs.index[-1] + 0.09,
            df_temp_Att.loc[end_yr, ('Ant', '50')] + 0.006),
        xytext=(df_temp_Obs.index[-1] + text_offset,
                df_temp_Att.loc[end_yr, ('Ant', '50')] + 0.05),
        color=var_colours['Ant'],
        fontweight='regular',
        arrowprops=dict(
            color=var_colours['Ant'],
            arrowstyle='->',
            connectionstyle=f"angle,angleA=0,angleB=45,rad={rad}"),
        verticalalignment='center',
        horizontalalignment='left'
        )

    # Calculate a trend line through the final 15 years of the GWI
    gwi_fit = np.polyfit(
        df_temp_Obs.index[-15:],
        df_temp_Att['Ant', '50'].iloc[-15:], 1)
    gwi_trend = np.poly1d(gwi_fit)
    gwi_trend = gwi_trend + (df_temp_Att.loc[end_yr, ('Ant', '50')] -
                             gwi_trend(df_temp_Obs.index[-1]))
    # Plot this line
    ax1.plot(df_temp_Obs.index[-15:], gwi_trend(df_temp_Obs.index[-15:]),
             color=SR15_colour, linestyle='--', linewidth=2)
    # Plot a scatter at the end of this line
    ax1.scatter(x=df_temp_Obs.index[-1],
                y=gwi_trend(df_temp_Obs.index[-1]),
                color=SR15_colour, s=45,
                label='SR1.5 trend-based')
    ax1.fill_between(
        df_temp_Obs.index[-15:],
        gwi_trend(df_temp_Obs.index[-15:]),
        df_temp_Att['Ant', '50'].iloc[-15:],
        color=SR15_colour, alpha=0.2
    )
    # Add an arrow pointing to trend-based scatter point
    value = (f"{df_headlines.loc[periods['trend_period'], ('Ant', '50')]} " +
             f"[{df_headlines.loc[periods['trend_period'], ('Ant', '5')]}" +
             "\N{EN DASH}" +
             f"{df_headlines.loc[periods['trend_period'], ('Ant', '95')]}] " +
             "°C")
    annotation = (r'$\bf{SR1.5 \ trend \ based}$' +
                  f'\n{end_yr} assessment:\n{value}')
    ax1.annotate(
        annotation,
        xy=(df_temp_Obs.index[-1] + 0.09,
            gwi_trend(df_temp_Obs.index[-1])-0.006),
        xytext=(df_temp_Obs.index[-1] + text_offset,
                gwi_trend(df_temp_Obs.index[-1]) - 0.05),
        color=SR15_colour,
        fontweight='regular',
        arrowprops=dict(
            color=SR15_colour,
            arrowstyle='->',
            connectionstyle=f"angle,angleA=0,angleB=-45,rad={rad}"),
        verticalalignment='center'
        )

    # find the average of the last 10 years of GWI
    decade_avg = df_temp_Att['Ant', '50'].iloc[-10:].mean()
    # decade_avg = df_headlines.loc['2014-2023', ('Ant', '50')]

    ax1.plot(df_temp_Obs.index[-10:],
             [decade_avg for _ in range(10)],
             color=AR6_colour,
             linestyle='--',
             linewidth=2
             )
    ax1.fill_between(
        df_temp_Obs.index[-10:],
        [decade_avg for _ in range(10)],
        df_temp_Att['Ant', '50'].iloc[-10:],
        color=AR6_colour, alpha=0.2
    )
    ax1.scatter(
        x=df_temp_Obs.index[-1] - 4.5,
        y=decade_avg,
        color=AR6_colour, s=50,
        label='AR6 decade-average'
    )

    value = (f"{df_headlines.loc[periods['decade_period'], ('Ant', '50')]} " +
             f"[{df_headlines.loc[periods['decade_period'], ('Ant', '5')]}" +
             "\N{EN DASH}" +
             f"{df_headlines.loc[periods['decade_period'], ('Ant', '95')]}] "
             "°C")
    annotation = (r'$\bf{AR6 \ decade \ average}$' +
                  f'\n{end_yr-9}\N{EN DASH}{end_yr} assessment:\n{value}')
    ax1.annotate(
        annotation,
        xy=(df_temp_Obs.index[-1] - 4.5 + 0.05,
            decade_avg - 0.003),
        xytext=(df_temp_Obs.index[-1] + text_offset,
                decade_avg-0.04),
        color=AR6_colour,
        fontweight='regular',
        arrowprops=dict(
            color=AR6_colour,
            arrowstyle='->',
            connectionstyle=f"angle,angleA=0,angleB=-45,rad={rad}"
            ),
        verticalalignment='center'
        )
