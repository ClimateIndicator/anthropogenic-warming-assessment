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


