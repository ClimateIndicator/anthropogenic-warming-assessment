"""Script to generate global warming index."""

import os
import sys

import datetime as dt
import functools

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns

import graphing as gr
import models.AR5_IR as AR5_IR


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
def regress_single_gwi(forc_All, temp_Obs, params, offset=False):
    """Regress GWI timeseries for single given ERF and Observed Temperature."""
    
    temp_All = forc_All.copy()
    vars = list(forc_All)
    for var in vars:
        # Calculate the temperature from ERF
        _forc = forc_All[var]
        _temp = AR5_IR.FTmod(forc_All.shape[0], params) @ _forc
        temp_All[var] = _temp
        # Substract preindustrial baseline
        _ofst = temp_All.loc[(temp_All.index >= start_pi) &
                             (temp_All.index <= end_pi),
                             var
                             ].mean()
        temp_All[var] -= _ofst
        
    temp_All = temp_All.loc[(temp_All.index >= start_yr) &
                            (temp_All.index <= end_yr)]
    
    # print(temp_All.head())
        
    # Calculate regression coefficients
    # The following code below is equivalent to the (more easily readable)
    # three lines of code immediately below
    # a = np.vstack([temp_GHG, temp_Aer, temp_Nat, np.ones(len(temp_GHG))]).T
    # b = np.linalg.lstsq(a, temp_Obs)[0]
    # coef_GHG, coef_Aer, coef_Nat, coef_Cst = b[0], b[1], b[2], b[3]

    # Then we regain (outside this function function for now)
    # temp_TOT = coef_GHG * temp_GHG + coef_Aer * temp_Aer + coef_Nat * temp_Nat + coef_Cst

    if offset:
        temp_All['ones'] = np.ones(temp_All.shape[0])
    temp_Mod_arr = np.array(temp_All)[:]
    temp_Mod_df = temp_All.copy()
    coef_Reg = np.linalg.lstsq(temp_Mod_arr, temp_Obs, rcond=None)[0]
    for i in range(temp_Mod_arr.shape[1]):  # ie number of variables
        temp_Mod_arr[:, i] *= coef_Reg[i]
        temp_Mod_df[list(temp_All)[i]] = temp_Mod_arr[:, i]
    temp_Mod_df['temp_TOT'] = temp_Mod_df.sum(axis=1)
    return temp_Mod_arr, temp_Mod_df


def moving_average(data, w):
    # data_padded = np.pad(data, (w//2, w-1-w//2),
    #                      mode='constant', constant_values=(0, 1.5))
    return np.convolve(data, np.ones(w), 'valid') / w


def temp_signal(data, w, method):
    # Sensibly extend data (to avoid shortening the length of moving average)
 
    # These are the lengths of the pads to add before and after the data.
    start_pad = w//2
    end_pad = w-1-w//2
    
    if method == 'constant':
        # Choices are:
        # - 0 before 1850 (we are defining this as preindustrial)
        # - 1.5 between 2022 and 2050 (the line through the middle)
        data_padded = np.pad(data, (start_pad, end_pad),
                             mode='constant',
                             constant_values=(0, 1.5))
    
    elif method == 'extrapolate':
        # Add zeros to the beginning (corresponding to pre-industrial state)
        extrap_start = np.zeros(start_pad)
        
        # Extrapolate the final w years to the end of the data
        A = np.vstack([np.arange(w), np.ones(w)]).T
        coef = np.linalg.lstsq(A, data[-w:], rcond=None)[0]
        B = np.vstack([np.arange(w + end_pad), np.ones(w + end_pad)]).T
        extrap_end = np.sum(coef*B, axis=1)[-end_pad:]
        data_padded = np.concatenate((extrap_start, data, extrap_end), axis=0)
    
    return moving_average(data_padded, w)
    return np.convolve(data_padded, np.ones(w), 'valid') / w



###############################################################################
# MAIN CODE BODY ##############################################################
###############################################################################

# Request whether to include pre-industrial offset and constant term in
# regression

allowed_options = ['y', 'n']
ao = '/'.join(allowed_options)

# inc_pi_offset = input(f'Subtract 1850-1900 PI baseline? {ao}: ')
# inc_reg_const = input(f'Include a constant term in regression? {ao}: ')

# Following discussion with Myles, we fix the options as the following:
inc_pi_offset = 'y'
inc_reg_const = 'y'

if inc_pi_offset not in allowed_options:
    print(f'{inc_pi_offset} not one of {ao}')
elif inc_reg_const not in allowed_options:
    print(f'{inc_reg_const} not one of {ao}')

inc_pi_offset = True if inc_pi_offset == 'y' else False
inc_reg_const = True if inc_reg_const == 'y' else False

##############################################################################
# READ IN THE DATA ###########################################################
##############################################################################

start_yr, end_yr = 1850, 2022
start_pi, end_pi = 1850, 1900  # As in IPCC AR6 Ch-3 Fig-3.4

sigmas = [[32, 68], [5, 95], [0.3, 99.7]]
sigmas_all = list(np.concatenate((np.sort(np.ravel(sigmas)), [50]), axis=0))


# TEMPERATURE #################################################################
# HadCRUT5 Observations
temp_Path = './data/HadCRUT/HadCRUT.5.0.1.0.analysis.ensemble_series.global.annual.csv'
df_temp_Obs = pd.read_csv(temp_Path,
                          ).rename(columns={'Time': 'Year'}
                          ).set_index('Year')
temp_Obs_ensemble_names = list(df_temp_Obs)[2:]
ofst_Obs = df_temp_Obs.loc[(df_temp_Obs.index >= start_pi) &
                           (df_temp_Obs.index <= end_pi),
                           temp_Obs_ensemble_names
                           ].mean(axis=0)
df_temp_Obs = df_temp_Obs.loc[(df_temp_Obs.index >= start_yr) &
                              (df_temp_Obs.index <= end_yr),
                              temp_Obs_ensemble_names] - ofst_Obs

## CMIP5 PI-CONTROL
obs_yrs = df_temp_Obs.shape[0]
n_yrs = obs_yrs

df_temp_PiC = pd.read_csv('./data/piControl/piControl.csv')

model_names = list(set(['_'.join(ens.split('_')[:1])
                        for ens in list(df_temp_PiC)]))

temp_IV_Group = {}

for ens in list(df_temp_PiC):
    if 'year' not in ens:
        # pi Control data located all over the place in csv; the following
        # lines strip the NaN values, and limits slices to the same length as
        # observed temperatures
        temp = np.array(df_temp_PiC[ens].dropna())[:obs_yrs]
        
        # Remove pre-industrial mean period; this is done because the models
        # use different "zero" temperatures (eg 0, 100, 287, etc).
        # An alternative approach would be to simply subtract the first value
        # to start all models on 0; the removal of the first 50 years
        # is used here in case the models don't start in equilibrium (and jump
        # up by x degrees at the start, for example), and the baseline period
        # is just defined as the same as for the observation PI period.
        temp -= temp[:start_pi-end_pi+1].mean()
    
        # Establish inclusion condition, which is that the smoothed internal
        # variability of a CMIP5 ensemble must operate within certain bounds:
        # 1. there must be a minimum level of variation (to remove those models
        # that are clearly wrong, eg oscillating between 0.01 and 0 warming)
        # 2. they must not exceed a certain min or max temperature bound; the
        # 0.3 value is roughyl similar to a 0.15 drift per century limit. I may
        # need to check this...
        # The final ensemble distribution are plotted against HadCRUT5 median
        # below, to check that the percentiles of this median run are similar
        # to the percentiles on the entire CMIP5 ensemble. ie, if the observed
        # internal variability is essentially a sampling of the climate each
        # year, you would expect the percentiles over history to be similar
        # to the percentiles of the ensemble in any given year. I allow the
        # ensemble to be slightly broader, to allow reasonably allow for a
        # wider range of behaviours than we have seen in the real world.
        temp_ma_3 = moving_average(temp, 3)
        temp_ma_30 = moving_average(temp, 30)
        _cond = (
                 (max(temp_ma_3) < 0.3 and min(temp_ma_3) > -0.3)
                 and ((max(temp_ma_3) - min(temp_ma_3)) > 0.06)
                 and (max(temp_ma_30) < 0.1 and min(temp_ma_30) > -0.1)
                 )

        # Approve actual (ie not smoothed) data if the corresponding smoothed
        # data is approved.
        # The second condition ensures that we aren't including timeseries that
        # are too short (not all pic control runs last the required 173 years).
        if _cond and len(temp) == obs_yrs:
            temp_IV_Group[ens] = temp

# Include the internval variability (IV) from HadCRUT to the overall
# dictionary of internal variabilities.
# for _temp_Ens in df_temp_Obs:
#     # Pick every 10th realisation
#     if int(_temp_Ens.split(' ')[1]) % 10 == 0:
#         _temp_Obs_signal = temp_signal(df_temp_Obs[_temp_Ens], 30)
#         _temp_Obs_IV = df_temp_Obs[_temp_Ens] - _temp_Obs_signal
#         temp_IV_Group[f'HadCRUT5 {_temp_Ens}'] = np.array(_temp_Obs_IV)

temp_Obs_signal = temp_signal(np.array(df_temp_Obs.quantile(q=0.5, axis=1)),
                              30, 'extrapolate')
temp_Obs_IV = df_temp_Obs.quantile(q=0.5, axis=1) - temp_Obs_signal
temp_IV_Group['HadCRUT5 median'] = np.array(temp_Obs_IV)

timeframes = [1, 3, 30]
lims = [0.6, 0.4, 0.15]

fig = plt.figure(figsize=(15, 10))

for t in range(len(timeframes)):
    axA = plt.subplot2grid(shape=(len(timeframes), 4), loc=(t, 0),
                           rowspan=1, colspan=3)
    axB = plt.subplot2grid(shape=(len(timeframes), 4), loc=(t, 3),
                           rowspan=1, colspan=1)
    
    axA.set_ylim(-lims[t], lims[t])
    cut_beg = timeframes[t]//2
    cut_end = timeframes[t]-1-timeframes[t]//2
    _time = np.arange(obs_yrs) + 1850
    
    if timeframes[t] == 1:
        _time_sliced = _time[:]
    else:
        _time_sliced = _time[cut_beg: -cut_end]

    for ens in temp_IV_Group:
        colour = 'xkcd:teal' if ens == 'HadCRUT5 median' else 'gray'
        label = 'HadCRUT5 median' if ens == 'HadCRUT5 median' else 'CMIP5 picontrol'
        alpha = 1 if ens == 'HadCRUT5 median' else 0.3
        _data = moving_average(temp_IV_Group[ens], timeframes[t])
        axA.plot(_time_sliced, _data, color=colour, alpha=alpha, label=label)

        density = ss.gaussian_kde(_data)
        x = np.linspace(axA.get_ylim()[0], axA.get_ylim()[1], 50)
        y = density(x)
        axB.plot(y, x, color=colour, alpha=alpha)

    # axes[i].set_ylim([-0.6, +0.6])
    axA.set_xlim(1845, 2030)
    axB.set_ylim(axA.get_ylim())
    # axB.get_yaxis().set_visible(False)
    axB.get_xaxis().set_visible(False)
    axA.set_ylabel(f'Internal Variability (°C) \n ({timeframes[t]}-year moving mean)')
gr.overall_legend(fig, loc='lower center', ncol=2, nrow=False)
fig.suptitle('Selected Sample of Internal Variability from CMIP5 pi-control')
fig.savefig('plots/0_Selected_CMIP5_Ensembles.png')


print('Number of CMIP5 internal variability samples remaining:' +
      f'{len(temp_IV_Group.keys())}')


#### PLOT THE ENSEMBLE ####
_t_IV = np.array([temp_IV_Group[ens] for ens in temp_IV_Group.keys()])

fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot2grid(shape=(1, 4), loc=(0, 0), rowspan=1, colspan=3)
ax2 = plt.subplot2grid(shape=(1, 4), loc=(0, 3), rowspan=1, colspan=1)

_t_IV_prcntls = np.percentile(_t_IV, sigmas_all, axis=0)
# Internal Variability Sample
for p in range(len(sigmas)):
    ax1.fill_between(df_temp_Obs.index,
                     _t_IV_prcntls[p, :], _t_IV_prcntls[-(p+2), :],
                     color='gray', alpha=0.2)
    ax2.fill_between(x=[3, 4],
                     y1=_t_IV_prcntls[p, :].mean()*np.ones(2),
                     y2=_t_IV_prcntls[-(p+2), :].mean()*np.ones(2),
                     color='gray', alpha=0.2)
               
ax1.plot(df_temp_Obs.index, _t_IV_prcntls[-1, :],
         color='gray', alpha=0.7, label='CMIP5 piControl')
ax2.plot([3, 4], _t_IV_prcntls[-1, :].mean()*np.ones(2),
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


gr.overall_legend(fig, loc='lower center', ncol=3, nrow=False)

ax1.set_ylabel('Internal Variability (°C)')
ax1.set_ylim(-0.6, 0.6)
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(0, 5)
ax2.get_xaxis().set_visible(False)
# Do the HadCRUT5...


fig.suptitle('Selected Sample of Internal Variability from CMIP5 pi-control')
fig.savefig('plots/1_Distribution_Internal_Variability.png')


### ERF
forc_Path = './data/ERF Samples/'
list_ERF = ['_'.join(file.split('_')[1:-1])
            for file in os.listdir(forc_Path)
            if '.csv' in file]
forc_Group = {
            #   'Ant': {'Consists': ['ant'], 'Colour': 'green'},
              'Nat': {'Consists': ['nat'], 'Colour': 'blue'},
              'GHG': {'Consists': ['co2', 'ch4', 'n2o', 'h2o_stratospheric',
                                   'o3_tropospheric', 'o3_stratospheric',
                                   'other_wmghg', 'land_use'],
                      'Colour': 'green'},
              'Aer': {'Consists': ['ari', 'aci', 'bc_on_snow', 'contrails'],
                      'Colour': 'orange'}
              }

for grouping in forc_Group:
    list_df = []
    for element in forc_Group[grouping]['Consists']:
        _df = pd.read_csv(forc_Path + f'rf_{element}_200samples.csv',
                          skiprows=[1]
                          ).rename(columns={'Unnamed: 0': 'Year'}
                          ).set_index('Year')
        list_df.append(_df.loc[_df.index <= end_yr])
    
    forc_Group[grouping]['df'] = functools.reduce(lambda x, y: x.add(y),
                                                  list_df)

forc_Group_names = sorted(list(forc_Group.keys()))

forc_All_ensemble_names = list(forc_Group[forc_Group_names[0]]['df'])

############ Set model parameters #############################################
# We only use a[10], a[11], a[15], a[16]
# Defaults:
# a_ar5[10:12] = [0.631, 0.429]  # AR5 thermal sensitivity coeffs
# a_ar5[15:17] = [8.400, 409.5]  # AR5 thermal time-inc_Constants -- could use Geoffroy et al [4.1,249.]

# a_ar5 = np.zeros(20, 16)

# # Geoffrey 2013 paramters for a_ar5[15:17]
Geoff = np.array([[4.0, 5.0, 4.5, 2.8, 5.2, 3.9, 4.2, 3.6,
                   1.6, 5.3, 4.0, 5.5, 3.5, 3.9, 4.3, 4.0],
                  [126, 267, 193, 132, 289, 200, 317, 197,
                   184, 280, 698, 286, 285, 164, 150, 218]])

###############################################################################
# Calculate GWI ###############################################################
###############################################################################

forc_Yrs = np.array(forc_Group[forc_Group_names[0]]['df'].index)
temp_Yrs = np.array(df_temp_Obs.index)

i = 0
t1 = dt.datetime.now()

# CARRY OUT REGRESSION CALCULATION ############################################
calc_switch = input('Recalculate? y/n: ')
if calc_switch == 'y':
    # Prepare results #############################################################

    samples = int(input('numer of samples (0-200): '))  # for temperature, and ERF
    n = samples * samples * len(temp_IV_Group.keys()) * Geoff.shape[1]
    print('Total number of combinations to sample: ' +
      f'{samples} forcings X {samples} HadCRUT X {len(temp_IV_Group.keys())}' +
      f' Internal Variability X {Geoff.shape[1]} Parameters = {n}')
    # Instead of using a separate array for the residuals and totals for sum total
    # and anthropogenic warming, now just include these in teh attributed results.
    # This is why we have the +1 +1 +1 +1 +1 above:
    #  +1 each for Ant, TOT, Res, Ens, Sig
    # NOTE: the order in dimension is:
    # 'AER, GHG, NAT, CONST, ANT, TOTAL, RESIDUAL, Temp_Ens, Temp_Sig'
    temp_Att_Results = np.zeros(
      (173, len(forc_Group_names) + int(inc_reg_const) + 1 + 1 + 1 + 1 + 1, n))
    Result_Components = forc_Group_names + ['Ant', 'TOT', 'Res', 'T_Ens', 'T_Sig']
    coef_Reg_Results = np.zeros((len(forc_Group_names) + int(inc_reg_const), n))
    IV_names = []
    
    for j in range(Geoff.shape[1]):
        params = np.zeros(20)
        params[10:12] = [0.631, 0.429]
        params[15:17] = Geoff[:, j]
        # params[15:17] = [8.400, 409.5]
        FTmodel = AR5_IR.FTmod(len(forc_Yrs), params)

        for forc_Ens in forc_All_ensemble_names[:samples]:
            forc_All = np.array([forc_Group[group]['df'][forc_Ens]
                                 for group in forc_Group_names]
                                 ).T

            temp_All = FTmodel @ forc_All
            if inc_pi_offset:
                _ofst = temp_All[(forc_Yrs >= start_pi) & (forc_Yrs <= end_pi), :
                                ].mean(axis=0)
            else:
                _ofst = 0
            temp_Mod = temp_All[(forc_Yrs >= start_yr) & (forc_Yrs <= end_yr)] - _ofst

            # Decide whether to include a Constant offset term in regression
            if inc_reg_const:
                temp_Mod = np.append(temp_Mod, np.ones((temp_Mod.shape[0], 1)), axis=1)
            num_vars = temp_Mod.shape[1]

            for temp_sig_Ens in temp_Obs_ensemble_names[:samples]:
                temp_Obs = np.array(df_temp_Obs[temp_sig_Ens])
                temp_Obs_signal = temp_signal(temp_Obs, 30, 'extrapolate')
                temp_Obs_signal -= temp_Obs_signal[:(end_pi - start_pi + 1)].mean()

                for temp_IV_Ens in temp_IV_Group.keys():
                    temp_Ens = temp_Obs_signal + temp_IV_Group[temp_IV_Ens]
                    temp_Ens -= temp_Ens[:(end_pi - start_pi + 1)].mean()
                    
                    # Carry out regression calculation
                    coef_Reg = np.linalg.lstsq(temp_Mod, temp_Ens, rcond=None)[0]
                    temp_Att = temp_Mod * coef_Reg
                    # Save outputs from the calculation:
                    # Regression coefficients
                    coef_Reg_Results[:, i] = coef_Reg
                    # Attributed warming for each component
                    temp_Att_Results[:, :num_vars, i] = temp_Att
                    # Actual temperature "ensemble" that was dependent variable
                    temp_Att_Results[:, -2, i] = temp_Ens
                    # The signal for this ensemble (ie minus internal variability)
                    temp_Att_Results[:, -1, i] = temp_Obs_signal
                    IV_names.append('_'.join(temp_IV_Ens.split('_')[:1]))
                    
                    # Note that it is more efficient to calculate the total and
                    # anthropogenic afterwards for the entire matrix at once.
                    
                    # Visual display of pregress through calculation
                    percentage = int((i+1)/n*100)
                    loading_bar = percentage // 5*'.' + (20 - percentage // 5)*' '
                    print(f'calculating {loading_bar} {percentage}%', end='\r')
                    i += 1


    # CALCULATE OTHER KEY RESULTS #################################################
    # TOTAL
    temp_Att_Results[:, -4, :] = (
        temp_Att_Results[:, :num_vars, :].sum(axis=1))
    # Residual
    temp_Att_Results[:, -3, :] = (
        temp_Att_Results[:, -2, :] - temp_Att_Results[:, -4, :])
    # Anthropogenic
    _temp_Ant_Results = (
        temp_Att_Results[:, -4, :] -
        # Remove the Natural forcing component in next line:
        temp_Att_Results[:, forc_Group_names.index('Nat'), :] -
        # Remove constant term in regression in next line:
        int(inc_reg_const) * temp_Att_Results[:, num_vars-1, :]
                        )
    temp_Att_Results[:, -5, :] = _temp_Ant_Results

    np.save('results/temp_Att_Results.npy', temp_Att_Results)
    np.save('results/coef_Reg_Results.npy', coef_Reg_Results)

elif calc_switch == 'n':
    temp_Att_Results = np.load('results/temp_Att_Results.npy')
    coef_Reg_Results = np.load('results/coef_Reg_Results.npy')
    n = temp_Att_Results.shape[2]

else:
    print(f'{calc_switch} not valid; exiting script.')
    sys.exit()

t2 = dt.datetime.now()
print(f'Total calculation took {t2-t1}')

# For diagnosing: filter out results with particular regression coefficients.

# If you only want to study subsets of the results based on certain constraints
# apply a mask here. The below mask is set to look at results for different
# values of the coefficients.

# Note to self about masking: coef_Reg_Results is the array of all regression
# coefficients, with shape (4, n), where n is total number of samplings.
# We select slice indices (forcing coefficients) we're interested in basing the
# condition on:
# AER is index 0, GHGs index 1, NAT index 2, Const index 3
# Then choose whether you want any or all or the coefficients to meet the   
# condition (in this case being less than zero)
mask_switch = False
if mask_switch:
    coef_mask = np.all(coef_Reg_Results[[0, 2], :] <= 0, axis=0)
    # mask = np.any(coef_Reg_Results[[0], :] <= 0.0, axis=0)

    temp_Att_Results = temp_Att_Results[:, :, coef_mask]
    coef_Reg_Results = coef_Reg_Results[:, coef_mask]
    print(f'Shape of masked attribution results: {temp_Att_Results.shape}')



###############################################################################
# PLOT RESULTS ################################################################
###############################################################################

# GWI PLOT ####################################################################
print('Creating GWI Plot...')

fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), rowspan=2, colspan=3)
ax2 = plt.subplot2grid(shape=(3, 4), loc=(0, 3), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(3, 4), loc=(2, 0), rowspan=1, colspan=3)
ax4 = plt.subplot2grid(shape=(3, 4), loc=(2, 3), rowspan=1, colspan=1)

# Plot the dependent temperature range ########################################
temp_Ens_unique = np.unique(temp_Att_Results[:, -2, :], axis=1)
temp_prcntls = np.percentile(temp_Ens_unique, sigmas_all, axis=1)

for p in range(len(sigmas)):
    ax1.fill_between(temp_Yrs, temp_prcntls[p, :], temp_prcntls[-(p+2), :],
                     color='gray', alpha=0.1)
ax1.plot(temp_Yrs, temp_prcntls[-1, :],
         color='gray', alpha=0.8,
         label='Dependent Temps: HadCRUT5 + CMIP5 piControl')

# Plot the observed temperatures on top as a scatter ##########################
err_pos = (df_temp_Obs.quantile(q=0.95, axis=1) -
           df_temp_Obs.quantile(q=0.5, axis=1))
err_neg = (df_temp_Obs.quantile(q=0.5, axis=1) -
           df_temp_Obs.quantile(q=0.05, axis=1))
ax1.errorbar(temp_Yrs, df_temp_Obs.quantile(q=0.5, axis=1),
             yerr=(err_neg, err_pos),
             fmt='o', color='gray', ms=2.5, lw=1,
             label='Reference Temp: HadCRUT5')
t2a = dt.datetime.now()
print(f'Dependent temperatures took {t2a-t2}')
# Plot the attribution results ################################################

# Select which components we want:
# gwi_component_name = ['Aer', 'GHG', 'Nat', 'Ant']
# gwi_component_list = [0, 1, 2, -5]

# gwi_plot_names = ['TOT', 'Ant', 'Aer', 'GHG', 'Nat']
gwi_plot_names = ['TOT', 'Ant', 'Aer', 'GHG', 'Nat', 'Res']

# gwi_plot_colours = ['purple', 'red', 'orange', 'green', 'blue']
gwi_plot_colours = ['xkcd:magenta', 'xkcd:crimson',
                    'xkcd:goldenrod', 'xkcd:teal', 'xkcd:azure',
                    'gray']

# gwi_plot_components = [-4, -5, 0, 1, 2, -3]
gwi_plot_components = [-4, -5, 0, 1, 2, -3]

gwi_prcntls = np.percentile(temp_Att_Results[:, gwi_plot_components, :],
                            sigmas_all, axis=2)
# WARNING TO SELF: multidimensional np.percentile() changes the order of the
# axes, so that the axis along which you took the percentiles is now the first
# axis, and the other axes are the remaining axes. This doesn't make any sense
# to me why this would be useful, but it is waht it is...
for c in range(len(gwi_plot_names)):
    if gwi_plot_names[c] in {'TOT', 'Aer', 'GHG', 'Nat'}:
        for p in range(len(sigmas)):
            ax1.fill_between(temp_Yrs,
                            gwi_prcntls[p, :, c], gwi_prcntls[-(p+2), :, c],
                            color=gwi_plot_colours[c],
                            alpha=0.1
                            )
        ax1.plot(temp_Yrs, gwi_prcntls[-1, :, c],
                    color=gwi_plot_colours[c],
                    label=gwi_plot_names[c]
                    )
ax1.set_ylabel('Warming Anomaly (°C)')

t2b = dt.datetime.now()
print(f'Independent temperatures took {t2b-t2a}')

# Residuals ###################################################################
for p in range(len(sigmas)):
    ax3.fill_between(temp_Yrs,
                     gwi_prcntls[p, :, -1], gwi_prcntls[-(p+2), :, -1],
                     color='gray', alpha=0.1)

ax3.plot(temp_Yrs, gwi_prcntls[-1, :, -1], color='gray',)
ax3.plot(temp_Yrs, np.zeros(len(temp_Yrs)),
         color='xkcd:magenta', alpha=1.0)
ax3.set_ylabel('Regression Residuals (°C)')

t2c = dt.datetime.now()
print(f'Residuals plot took {t2c-t2b}')


# Distributions ###############################################################
for c in range(len(gwi_plot_names)):
    if gwi_plot_names[c] in {'TOT', 'Aer', 'GHG', 'Nat'}:
        # binwidth = 0.01
        # bins = np.arange(np.min(temp_Att_Results[-1, gwi_plot_components[i], :]),
        #                  np.max(temp_Att_Results[-1, gwi_plot_components[i], :]) + binwidth,
        #                  binwidth)
        # ax2.hist(temp_Att_Results[-1, gwi_plot_components[i], :], bins=bins,
        #          density=True, orientation='horizontal',
        #          color=gwi_plot_colours[i],
        #          alpha=0.3
        #          )
        density = ss.gaussian_kde(temp_Att_Results[-1, gwi_plot_components[c], :])
        x = np.linspace(
            temp_Att_Results[-1, gwi_plot_components[c], :].min(),
            temp_Att_Results[-1, gwi_plot_components[c], :].max(),
            100)
        y = density(x)
        # ax2.plot(y, x, color=gwi_plot_colours[i], alpha=0.7)
        ax2.fill_betweenx(x, np.zeros(len(y)), y,
                          color=gwi_plot_colours[c], alpha=0.3)

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


ax2.set_title(f'PDF in {end_yr}')
ax2.set_ylim(ax1.get_ylim())

t2d = dt.datetime.now()
print(f'Distributions took {t2d-t2c}')

# Plotting anthropogenic vs total warming #####################################
ax4.plot([-0.2, 1.5], [-0.2, 1.5], color='gray', alpha=0.7)

for p in range(len(sigmas)):
    ax4.plot(gwi_prcntls[p, :, 1], gwi_prcntls[p, :, 0],
             color='xkcd:magenta', alpha=0.7)

ax4.set_xlabel('Ant')
ax4.set_ylabel('TOT')
ax4.set_xlim(-0.2, 1.5)
ax4.set_ylim(-0.2, 1.5)

# Make the headline result...
gwi = np.around(np.percentile(temp_Att_Results[-1, -4, :], (50)), decimals=3)
gwi_pls = np.around(np.percentile(temp_Att_Results[-1, -4, :], (95)) -
                    np.percentile(temp_Att_Results[-1, -4, :], (50)),
                    decimals=3)
gwi_min = np.around(np.percentile(temp_Att_Results[-1, -4, :], (50)) -
                    np.percentile(temp_Att_Results[-1, -4, :], (5)),
                    decimals=3)

str_GWI = r'${%s}^{+{%s}}_{-{%s}}$' % (gwi, gwi_pls, gwi_min)
# str_temp_Obs = r'${%s}^{+{%s}}_{-{%s}}$' % (tmp, tmp_pls, tmp_min)
# fig.text(s=(f'Warming in {end_yr}: ' +
#             f'human-induced-warming = {str_GWI} (°C)'),
#             # f'observed warming = {str_temp_Obs}'),
#          y=0.9, x=0.5, horizontalalignment='center')
ax1.set_title(f'Warming in {end_yr}: ' +
              f'human-induced-warming = {str_GWI} (°C)')
fig.suptitle(f'Global Warming Index ({n} samplings)')
gr.overall_legend(fig, 'lower center', 6)
fig.savefig('plots/2_GWI.png')

t3 = dt.datetime.now()
print(f'... all took {t3-t2}')

# # Plot coefficients ###########################################################
print('Creating Coefficient Plot...', end=' ')
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot2grid(
    shape=(1, 1), loc=(0, 0),
    # rowspan=1, colspan=3
    )
# unique_IV_names = sorted(list(set(IV_names)))
# cm = 'Set3'
# cols = np.array(sns.color_palette(cm, len(unique_IV_names)))
# cols_hex = [matplotlib.colors.rgb2hex(cols[i, :])
#             for i in range(cols.shape[0])]

# IV_col_dic = dict(zip(unique_IV_names, cols_hex))
# use_colours = [IV_col_dic[_IV] for _IV in IV_names]
ax.scatter(coef_Reg_Results[0, :], coef_Reg_Results[1, :],
           #    color=use_colours,
           color='xkcd:teal',
           alpha=0.01, edgecolors='none', s=20)
ax.set_xlabel('AER')
ax.set_ylabel('GHG')
# plt.ylim(bottom=0)
fig.suptitle(f'Coefficients from {n} Samplings')
fig.savefig('plots/3_Coefficients.png')
t4 = dt.datetime.now()
print(f'took {t4-t3}')

########### TEST SEABORN
plt.close()

coef_df = pd.DataFrame(coef_Reg_Results.T,
                       columns=['Aer', 'GHG', 'Nat', 'Const'])
# print(coef_df.head())
g = sns.PairGrid(coef_df.sample(5000))
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)
g.fig.suptitle('Regression Coefficient Distributions')
plt.savefig('plots/SNS TEST.png')

###############################################################################
# Recreate IPCC AR6 SPM.2 Plot
# https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf
###############################################################################
print('Creating IPCC Comparison Bar Plot...', end=' ')
SPM2_list = ['TOT', 'Ant', 'Aer', 'GHG', 'Nat']
# Note that the central estimate for aerosols isn't given; only the range is
# specified; a pixel ruler was used on the pdf to get the rough central value.
SPM2_med = [1.09,
            1.07,
            (-250/310)*0.5,
            1.50,
            0.00]
SPM2_neg = [1.09-0.95,
            1.07-0.80,
            ((-250/310)*0.5) - (-0.80),
            1.50-1.00,
            0.00-(-0.10)]
SPM2_pos = [1.20-1.09,
            1.30-1.07,
            0.00-((-250/310)*0.5),
            2.00-1.50,
            0.10-0.00]

recent_years = ((2010 <= temp_Yrs) * (temp_Yrs < 2020))
recent_components = [-4, -5, 0, 1, 2]
# Simultaneously index two dimensions using lists (one indices, one booleans)
idx = np.ix_(recent_years, recent_components)
temp_Att_Results_recent = temp_Att_Results[idx].mean(axis=0)
# Obtain statistics
recent_med = np.percentile(temp_Att_Results_recent[:], (50), axis=1)
recent_neg = recent_med - np.percentile(temp_Att_Results_recent[:], (5), axis=1)
recent_pos = np.percentile(temp_Att_Results_recent[:], (95), axis=1) - recent_med

recent_x_axis = np.arange(len(SPM2_list))
bar_width = 0.3

# Plot SPM2 data
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot2grid(
    shape=(1, 1), loc=(0, 0),
    # rowspan=1, colspan=3
    )
ax.bar(recent_x_axis-bar_width/2, SPM2_med,
       width=bar_width, color='xkcd:teal', label='SPM.2', alpha=0.4)
ax.errorbar(recent_x_axis-bar_width/2, SPM2_med, yerr=(SPM2_neg, SPM2_pos),
             fmt='none', color='black')
# Plot GWI data
ax.bar(recent_x_axis+bar_width/2, recent_med,
        width=bar_width, color='xkcd:azure', label='GWI', alpha=0.4)
ax.errorbar(recent_x_axis+bar_width/2, recent_med, yerr=(recent_neg, recent_pos),
             fmt='none', color='black')
ax.set_xticks(recent_x_axis, SPM2_list)
ax.set_ylabel('Contributions to 2010-2019 warming relative to 1850-1900')
gr.overall_legend(fig, 'lower center', 2)
fig.suptitle('Comparison of GWI to IPCC AR6 SPM.2 Assessment')
fig.savefig('plots/4_SPM2_Comparison.png')

t5 = dt.datetime.now()
print(f'took {t5-t4}')


sys.exit()
############### WHAT IS UP WITH THE NEGATIVE COEFFICIENT FITS...

for example in range(temp_Att_Results.shape[2]):
    plt.plot(temp_Yrs, temp_TEST_Sig_Results[:, example],
        color='black',
        label='Temp Observation Signal')
    plt.plot(temp_Yrs, temp_TEST_Ens_Results[:, example],
        color='black',
        label='Temp Observation Signal + Internal Variability')
    
    for i in range(len(forc_Group_names)):
        plt.plot(temp_Yrs, temp_Att_Results[:, i, example],
            color=forc_Group[forc_Group_names[i]]['Colour'],
            label=str(forc_Group_names[i])
            )
    plt.plot(temp_Yrs, temp_TOT_Results[:, example],
                color='purple',
                label='TOT')
    plt.title(coef_Reg_Results[:, i])

    plt.legend()
    plt.show()

sys.exit()



###############################################################################
# Calculate the historical-only GWI ###########################################
###############################################################################

hist_start = 1900
historical_samples = 50
n = historical_samples ** 2
hist_temp_Att_Results = np.empty((
    end_yr - hist_start + 1,  # num generated years
    len(forc_Group_names) + int(inc_reg_const),  # forcings
    n  # samples
    ))


t1 = dt.datetime.now()
for y in temp_Yrs[temp_Yrs >= hist_start]:
    yi = np.where(temp_Yrs[temp_Yrs >= hist_start] == y)
    forc_Yrs_y = forc_Yrs[forc_Yrs <= y]
    i = 0
    for forc_Ens in forc_All_ensemble_names[:historical_samples]:
        # Calculate forcing -> temperature response. The below steps calculate
        # temperature for all sources simultaneously (eg Nat, GHG, and Aer)
        # forc_All = np.array([df_forc_Nat[forc_Ens], df_forc_Ant[forc_Ens]]).T

        forc_All = np.array([forc_Group[group]['df'][forc_Ens].loc[
                                forc_Group[group]['df'][forc_Ens].index <= y]
                            for group in forc_Group_names]).T
        params = AR5_IR.a_params('Carbon Dioxide')
        temp_All = AR5_IR.FTmod(forc_All.shape[0], params) @ forc_All

        if inc_pi_offset:
            _ofst = temp_All[(forc_Yrs_y >= start_pi) &
                             (forc_Yrs_y <= min(end_pi, y)), :
                             ].mean(axis=0)
        else:
            _ofst = 0
        temp_Mod = temp_All[(forc_Yrs_y >= start_yr) & (forc_Yrs_y <= y)] - _ofst

        # Decide whether to include a inc_Constant offset term in regression
        if inc_reg_const:
            temp_Mod = np.append(temp_Mod, np.ones((temp_Mod.shape[0], 1)), axis=1)

        for temp_Ens in temp_Obs_ensemble_names[:historical_samples]:
            # Select the relevant observational data
            temp_Obs = np.array(df_temp_Obs[temp_Ens])[(temp_Yrs >= start_yr) & (temp_Yrs <= y)]

            # Carry out regression calculation
            coef_Reg = np.linalg.lstsq(temp_Mod, temp_Obs, rcond=None)[0]
            # coef_Reg = sp.optimize.lsq_linear(temp_Mod, temp_Obs, bounds=(1, 1.5))['x']
            
            GWI_y = temp_Mod[-1, :] * coef_Reg
            hist_temp_Att_Results[yi, :, i] = GWI_y
            i += 1
    
    # Visual display of pregress through calculation
    percentage = int((yi[0]+1)/len(list(temp_Yrs[temp_Yrs >= hist_start]))*100)
    loading_bar = percentage // 5*'.' + (20 - percentage // 5)*' '
    print(f'calculating {loading_bar} {percentage}%', end='\r')

t2 = dt.datetime.now()
print(f'Total calculation took {t2-t1}')


plt.errorbar(temp_Yrs, df_temp_Obs.quantile(q=0.5, axis=1),
             yerr=(err_neg, err_pos),
             fmt='o', color='black', ms=2.5, lw=1,
             label='HadCRUT5')

for p in sigmas:
    for i in range(len(forc_Group_names)):
        plt.fill_between(temp_Yrs[temp_Yrs >= hist_start],  
            np.percentile(hist_temp_Att_Results[:, i, :], (p[0]), axis=1),
            np.percentile(hist_temp_Att_Results[:, i, :], (p[1]), axis=1),
            color=forc_Group[forc_Group_names[i]]['Colour'],
            alpha=0.1)

        if p == sigmas[-1]:
            plt.plot(temp_Yrs[temp_Yrs >= hist_start],
                np.percentile(hist_temp_Att_Results[:, i, :], (50), axis=1),
                color=forc_Group[forc_Group_names[i]]['Colour'],
                label=forc_Group_names[i]
                )
hist_temp_Ant_Results = (temp_Att_Results.sum(axis=1) -
                    temp_Att_Results[:, forc_Group_names.index('Nat'), :] - 
                    int(inc_reg_const) * temp_Att_Results[:, len(forc_Group_names), :]  # Remove constant term in regression
                    )
hist_temp_TOT_Results = hist_temp_Att_Results.sum(axis=1)
for p in sigmas:
    plt.fill_between(temp_Yrs[temp_Yrs >= hist_start],
                    np.percentile(hist_temp_TOT_Results, (p[0]), axis=1),
                    np.percentile(hist_temp_TOT_Results, (p[1]), axis=1),
                    color='gray',
                    alpha=0.1)
    if p == sigmas[-1]:
        plt.plot(temp_Yrs[temp_Yrs >= hist_start],
                np.percentile(hist_temp_TOT_Results, (50), axis=1),
                color='gray',
                label='TOT')

plt.legend()
plt.title('Historical GWI')
plt.savefig(f'Historical GWI PI_offset-{inc_pi_offset} Reg_Const-{inc_reg_const}.png')