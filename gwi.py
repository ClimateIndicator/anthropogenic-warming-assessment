"""Script to generate global warming index."""

import os
import sys
import glob

import datetime as dt
import functools
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr

# import matplotlib
import matplotlib.pyplot as plt
# import scipy.stats as ss
import seaborn as sns

import graphing as gr

import models.AR5_IR as AR5_IR
import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair
import pymagicc


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
def load_ERF_Stuart():
    """Load the ERFs from Stuart's ERF datasets."""
    forc_Path = './data/ERF Samples/Stuart/'
    # list_ERF = ['_'.join(file.split('_')[1:-1])
    #             for file in os.listdir(forc_Path)
    #             if '.csv' in file]
    forc_Group = {
                #   'Ant': {'Consists': ['ant'], 'Colour': 'green'},
                'Nat': {'Consists': ['nat'],
                        'Colour': 'green'},
                'GHG': {'Consists': ['co2', 'ch4', 'n2o', 'other_wmghg'],
                        'Colour': 'orange'},
                'OHF': {'Consists': ['ari', 'aci', 'bc_on_snow', 'contrails',
                                     'o3_tropospheric', 'o3_stratospheric',
                                     'h2o_stratospheric', 'land_use'],
                        'Colour': 'blue'}
                }

    for grouping in forc_Group:
        list_df = []
        for element in forc_Group[grouping]['Consists']:
            _df = pd.read_csv(
                forc_Path + f'rf_{element}_200samples.csv', skiprows=[1]
                   ).rename(columns={'Unnamed: 0': 'Year'}
                   ).set_index('Year')
            list_df.append(_df.loc[_df.index <= end_yr])

        forc_Group[grouping]['df'] = functools.reduce(lambda x, y: x.add(y),
                                                      list_df)
    return forc_Group


def load_ERF_CMIP6():
    """Load the ERFs from CMIP6."""
    # ERF location
    file_ERF = 'data/ERF Samples/Chris/ERF_DAMIP_1000.nc'
    # import ERF_file to xarray dataset and convert to pandas dataframe
    df_ERF = xr.open_dataset(file_ERF).to_dataframe()
    # assign the columns the name 'variable'
    df_ERF.columns.names = ['variable']
    # remove the column called 'total' from df_ERF
    df_ERF = df_ERF.drop(columns='total')
    # rename the variable columns
    df_ERF = df_ERF.rename(columns={'wmghg': 'GHG',
                                    'other_ant': 'OHF',
                                    'natural':'Nat'})
    # move the multi-index 'ensemble' level to a column,
    # and then set the 'ensemble' column to second column level
    df_ERF = df_ERF.reset_index(level='ensemble')
    df_ERF['ensemble'] = 'ens' + df_ERF['ensemble'].astype(str)
    df_ERF = df_ERF.pivot(columns='ensemble')

    return df_ERF


def load_HadCRUT(start_pi, end_pi):
    """Load HadCRUT5 observations and remove PI baseline."""
    temp_ens_Path = (
        './data/Temp/HadCRUT/' +
        'HadCRUT.5.0.1.0.analysis.ensemble_series.global.annual.csv')
    # read temp_Path into pandas dataframe, rename column 'Time' to 'Year'
    # and set the index to 'Year', keeping only columns with 'Realization' in
    # the column name, since these are the ensembles
    df_temp_Obs = pd.read_csv(temp_ens_Path,
                              ).rename(columns={'Time': 'Year'}
                                       ).set_index('Year'
                                                   ).filter(regex='Realization')

    # Find PI offset that is the PI-mean of the median (HadCRUT best estimate)
    # of the ensemble and substract this from entire ensemble. Importantly,
    # the same offset is applied to the entire ensemble to maintain accurate
    # spread of HadCRUT (ie it is wrong to subtract the PI-mean for each
    # ensemble member from itself).
    ofst_Obs = df_temp_Obs.median(axis=1).loc[
        (df_temp_Obs.index >= start_pi) &
        (df_temp_Obs.index <= end_pi),
        ].mean(axis=0)
    df_temp_Obs -= ofst_Obs

    return df_temp_Obs


def load_PiC_Stuart(n_yrs):
    """Load piControl data from Stuart's ERF datasets."""
    df_temp_PiC = pd.read_csv('./data/piControl/piControl.csv'
                          ).rename(columns={'year': 'Year'}
                                   ).set_index('Year')
    # model_names = list(set(['_'.join(ens.split('_')[:1])
    #                         for ens in list(df_temp_PiC)]))

    temp_IV_Group = {}

    for ens in list(df_temp_PiC):
        # pi Control data located all over the place in csv; the following
        # lines strip the NaN values, and limits slices to the same length as
        # observed temperatures
        temp = df_temp_PiC[ens].dropna().to_numpy()[:n_yrs]

        # Remove pre-industrial mean period; this is done because the models
        # use different "zero" temperatures (eg 0, 100, 287, etc).
        # An alternative approach would be to simply subtract the first value
        # to start all models on 0; the removal of the first 50 years
        # is used here in case the models don't start in equilibrium (and jump
        # up by x degrees at the start, for example), and the baseline period
        # is just defined as the same as for the observation PI period.
        temp -= temp[:start_pi-end_pi+1].mean()

        if len(temp) == n_yrs:
            temp_IV_Group[ens] = temp

    return pd.DataFrame(temp_IV_Group)


def load_PiC_CMIP6(n_yrs, start_pi, end_pi):
    """Create DataFrame of piControl data from .MAG files."""
    # Create list of all .MAG files recursively inside the directory
    # data/piControl/CMIP6. These files are simply as extracted from zip
    # downloaded from https://cmip6.science.unimelb.edu.au/results?experiment_id=piControl&normalised=&mip_era=CMIP6&timeseriestype=average-year-mid-year&variable_id=tas&region=World#download
    # (ie a CMIP6 archive for pre-meaned data, saving data/time.)
    mag_files = sorted(glob.glob('data/piControl/CMIP6/**/*.MAG',
                                 recursive=True))
    dict_temp = {}
    for file in mag_files:
        # Adopt nomenclature format that matches earlier csv from Stuart
        group = file.split('/')[6]
        model = file.split('/')[-1].split('_')[3]
        member = file.split('/')[-1].split('_')[5]
        var = file.split('/')[-1].split('_')[1]
        experiment = file.split('/')[-1].split('_')[4]
        model_name = '_'.join([group, model, member, var, experiment])

        # use pymagicc to read the .MAG file
        df_PiC = pymagicc.io.MAGICCData(file).to_xarray().to_dataframe()
        # select only the data with keyword 'world' in the level 1 index
        df_PiC = df_PiC.xs('World', level=1)
        # replace the cftime index with an integer for the cftime year
        df_PiC.index = df_PiC.index.year

        temp = df_PiC.dropna().to_numpy().ravel()

        # Create multiple segments with 50% overlap from each other.
        # ie 0:173, 86:259, 172:345, etc
        segments = (temp.shape[0] - (n_yrs - n_yrs//2)) // (n_yrs//2)
        for s in range(segments):
            # print(s*(n_yrs//2), s*(n_yrs//2)+n_yrs)
            temp_s = temp[s*(n_yrs//2):s*(n_yrs//2)+n_yrs]
            temp_s = temp_s - temp_s[:(end_pi-start_pi)].mean()
            dict_temp[
                f'{model_name}_slice-{s*(n_yrs//2)}:{s*(n_yrs//2)+n_yrs}'
                ] = temp_s

    return pd.DataFrame(dict_temp)


def filter_PiControl(df, timeframes):
    """Remove simulations that correspond poorly with observations."""
    dict_temp_PiC = {}
    for ens in list(df):
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
        temp = df[ens].to_numpy()
        temp_ma_3 = moving_average(temp, 3)
        temp_ma_30 = moving_average(temp, 30)
        _cond = (
                    (max(temp_ma_3) < 0.3 and min(temp_ma_3) > -0.3)
                    and ((max(temp_ma_3) - min(temp_ma_3)) > 0.06)
                    and (max(temp_ma_30) < 0.1 and min(temp_ma_30) > -0.1)
                    )

        # Approve actual (ie not smoothed) data if the corresponding smoothed
        # data is approved.
        if _cond:
            dict_temp_PiC[ens] = temp

    return pd.DataFrame(dict_temp_PiC)


def GWI_faster(
        model_choice, inc_reg_const, inc_pi_offset,
        df_forc, df_params, df_temp_PiC, df_temp_Obs,
        start_yr, end_yr, start_pi, end_pi):
    """Calculate the global warming index (GWI)."""
    """Parallelise over FaIR parameterisations, exploit vectorisation of
    FaIR model by running all forcings at once through it, and separate
    regression against observations and piControl to add rather than multiply
    linear regressions' computational time."""

    variables = df_forc.columns.get_level_values('variable').unique().to_list()
    ensembles = df_forc.columns.get_level_values("ensemble").unique().to_list()

    # # Prepare results #########################################################
    n = (df_temp_Obs.shape[1] * df_temp_PiC.shape[1] *
         len(df_forc.columns.get_level_values("ensemble").unique()) *
         #  len(df_params.columns.levels[0])
         1  # only use 1 FaIR parameter set per function call in parallel method
         )
    # # Include residuals and totals for sum total and anthropogenic warming in
    # # the same array as attributed results. +1 each for Ant, TOT, Res,
    # # InternalVariability, ObservedTemperatures
    # # NOTE: the order in dimension is:
    # # 'GHG, NAT, OHF, CONST, ANT, TOTAL, RESIDUAL, Temp_PiC, Temp_Obs'
    # # vars = ['GHG', 'Nat', 'OHF', 'Const', 'Ant', 'Tot', 'Res', 'PiC', 'Obs']
    vars = ['GHG', 'Nat', 'OHF', 'Ant', 'Tot', 'Res']
    temp_Att_Results = np.empty(
      (end_yr - start_yr + 1,  # years
       len(variables) + 3,  # variables
       n),  # samples
      dtype=np.float32  # make the array smaller in memory
      )
    # coef_Reg_Results = np.zeros((len(variables) + int(inc_reg_const), n))

    # slice df_temp_obs dataframe to include years between start_yr and end_yr
    df_temp_Obs = df_temp_Obs.loc[start_yr:end_yr]
    # slice df_temp_PiC dataframe to include years between start_yr and end_yr
    df_temp_PiC = df_temp_PiC.loc[start_yr:end_yr]
    temp_Yrs = df_temp_Obs.index.to_numpy()
    forc_Yrs = df_forc.index.to_numpy()

    # Prepare FaIR parameters for this particular model.
    # print(model_choice)
    params_FaIR = df_params[model_choice]
    params_FaIR.columns = pd.MultiIndex.from_product(
        [[model_choice], params_FaIR.columns])

    # Prepare results array for temperatures.
    temp_Mod_array = np.empty(shape=(forc_Yrs.shape[0],
                              len(variables),
                              len(ensembles)))

    # Calculate temperatures from forcings for all ensembles at once,
    # leveraging FaIR's vectorisation
    for var in variables:
        # Select forcings for the specific ensemble member
        forc_var_All = df_forc.loc[:end_yr, (var, slice(None))]

        # FaIR won't run without emissions or concentrations, so specify
        # no zero emissions for input.
        emis_FAIR = fair.return_empty_emissions(
            df_to_copy=False,
            start_year=min(forc_Yrs), end_year=end_yr, timestep=1,
            scen_names=ensembles)
        # Prepare a FaIR-compatible forcing dataframe
        forc_FaIR = fair.return_empty_forcing(
            df_to_copy=False,
            start_year=min(forc_Yrs), end_year=end_yr, timestep=1,
            scen_names=ensembles)
        for ens in ensembles:
            forc_FaIR[ens] = forc_var_All[(var, ens)].to_numpy()
        # Run FaIR. Convert output to numpy array for later regression.
        temp_All = fair.run_FaIR(emissions_in=emis_FAIR,
                                 forcing_in=forc_FaIR,
                                 thermal_parameters=params_FaIR,
                                 show_run_info=False)['T'].to_numpy()
        temp_Mod_array[:, variables.index(var), :] = temp_All

    i = 0
    for ens in range(len(ensembles)):
        # Cut the full-forcing-length temperatures down to the same length as
        # the other temperature data
        yr_mask = ((forc_Yrs >= start_yr) & (forc_Yrs <= end_yr))
        temp_Mod = temp_Mod_array[yr_mask, :, ens]

        # Remove pre-industrial offset before regression
        if inc_pi_offset:
            _ofst = temp_Mod[(temp_Yrs >= start_pi) &
                             (temp_Yrs <= end_pi), :
                             ].mean(axis=0)
            temp_Mod = temp_Mod - _ofst

        # Decide whether to include a Constant offset term in regression
        if inc_reg_const:
            temp_Mod = np.append(temp_Mod,
                                 np.ones((temp_Mod.shape[0], 1)),
                                 axis=1)
        n_reg_vars = temp_Mod.shape[1]

        coef_Obs_Results = np.empty((temp_Mod.shape[1],
                                     df_temp_Obs.shape[1]))
        coef_PiC_Results = np.empty((temp_Mod.shape[1],
                                     df_temp_PiC.shape[1]))

        c_i = 0
        for temp_Obs_Ens in df_temp_Obs.columns:
            temp_Obs_i = df_temp_Obs[temp_Obs_Ens].to_numpy()
            coef_Obs_i = np.linalg.lstsq(temp_Mod, temp_Obs_i, rcond=None)[0]
            coef_Obs_Results[:, c_i] = coef_Obs_i
            c_i += 1

        c_j = 0
        for temp_PiC_Ens in df_temp_PiC.columns:
            temp_PiC_j = df_temp_PiC[temp_PiC_Ens].to_numpy()
            coef_PiC_j = np.linalg.lstsq(temp_Mod, temp_PiC_j, rcond=None)[0]
            coef_PiC_Results[:, c_j] = coef_PiC_j
            c_j += 1

        for c_k in range(coef_Obs_Results.shape[1]):
            for c_l in range(coef_PiC_Results.shape[1]):
                # Regression coefficients
                coef_Reg = (coef_Obs_Results[:, c_k] +
                            coef_PiC_Results[:, c_l])
                # Attributed warming for each component
                temp_Att = temp_Mod * coef_Reg

                # Extract T_Obs and T_PiC data for this c_i, c_j combo.
                temp_Obs_kl = df_temp_Obs[df_temp_Obs.columns[c_k]
                                          ].to_numpy()
                # temp_PiC_kl = df_temp_PiC[df_temp_PiC.columns[c_l]
                #                           ].to_numpy()

                # Save outputs from the calculation:
                # Regression coefficients
                # coef_Reg_Results[:, i] = coef_Reg

                # Attributed warming for each component
                temp_Att_Results[:, :(n_reg_vars-(1*inc_reg_const)), i] = \
                    temp_Att[:, :-1]

                # # Actual piControl IV sample that used for this c_k, c_l
                # temp_Att_Results[:, -2, i] = temp_PiC_kl
                # # The temp_Obs (dependent var) for this c_k, c_l
                # temp_Att_Results[:, -1, i] = temp_Obs_kl
                # TOTAL
                temp_Tot = temp_Att.sum(axis=1)
                temp_Att_Results[:, -2, i] = temp_Tot
                # RESIDUAL
                temp_Att_Results[:, -1, i] = (temp_Obs_kl - temp_Tot)
                # ANTROPOGENIC
                temp_Ant = (temp_Att[:, variables.index('GHG')] +
                            temp_Att[:, variables.index('OHF')])
                temp_Att_Results[:, -3, i] = temp_Ant

                # Visual display of pregress through calculation
                if i % 1000 == 0:
                    percentage = int((i+1)/n*100)
                    loading_bar = (percentage // 5*'.' +
                                (20 - percentage // 5)*' ')
                    print(f'calculating {loading_bar} {percentage}%', end='\r')
                i += 1

    # print(f"calculating {20*'.'} {100}%", end='\r')
    return temp_Att_Results


def GWI(
        variables, inc_reg_const,
        df_forc, df_params, df_temp_PiC, df_temp_Obs,
        start_yr, end_yr):
    """Calculate the global warming index (GWI)."""
    # - BRING start_pi AND end_pi INSIDE THE FUNCTION


    # Prepare results #########################################################
    n = (df_temp_Obs.shape[1] * df_temp_PiC.shape[1] *
         len(forc_subset.columns.get_level_values("ensemble").unique()) *
         len(df_params.columns.levels[0]))
    # Include residuals and totals for sum total and anthropogenic warming in
    # the same array as attributed results. +1 each for Ant, TOT, Res,
    # InternalVariability, ObservedTemperatures
    # NOTE: the order in dimension is:
    # 'GHG, NAT, OHF, CONST, ANT, TOTAL, RESIDUAL, Temp_PiC, Temp_Obs'
    # vars = ['GHG', 'Nat', 'OHF', 'Const', 'Ant', 'Tot', 'Res', 'PiC', 'Obs']
    vars = ['GHG', 'Nat', 'OHF', 'Ant', 'Tot', 'Res']
    temp_Att_Results = np.zeros(
      (end_yr - start_yr + 1,  # years
       len(variables) + 3,  # variables
       n),  # samples
      dtype=np.float32  # make the array smaller in memory
      )
    coef_Reg_Results = np.zeros((len(variables) + int(inc_reg_const), n))

    forc_Yrs = df_forc.index.to_numpy()
    # slice df_temp_obs dataframe to include years between start_yr and end_yr
    df_temp_Obs = df_temp_Obs.loc[start_yr:end_yr]
    # slice df_temp_PiC dataframe to include years between start_yr and end_yr
    df_temp_PiC = df_temp_PiC.loc[start_yr:end_yr]

    # Loop over all sampling combinations #####################################
    i = 0
    for CMIP6_model in df_params.columns.levels[0].unique():
        # Select the specific model's parameters
        params_FaIR = df_params[CMIP6_model]
        # Since the above line seems to get rid of the top colum level (the
        # model name), and therefore reduce the level to 1, we need to re-add
        # the level=0 column name (the model name) in order for this to be
        # compatible with the required FaIR format...
        params_FaIR.columns = pd.MultiIndex.from_product(
            [[CMIP6_model], params_FaIR.columns])

        for forc_Ens in df_forc.columns.get_level_values("ensemble").unique():
            # Select forcings for the specific ensemble member
            forc_Ens_All = df_forc.loc[:end_yr, (slice(None), forc_Ens)]

            # FaIR won't run without emissions or concentrations, so specify
            # no zero emissions for input.
            emis_FAIR = fair.return_empty_emissions(
                df_to_copy=False,
                start_year=min(forc_Yrs), end_year=end_yr, timestep=1,
                scen_names=variables)
            # Prepare a FaIR-compatible forcing dataframe
            forc_FaIR = fair.return_empty_forcing(
                df_to_copy=False,
                start_year=min(forc_Yrs), end_year=end_yr, timestep=1,
                scen_names=variables)
            for var in variables:
                forc_FaIR[var] = forc_Ens_All[var].to_numpy()

            # Run FaIR
            # Convert back into numpy array for comapbililty with the pre-FaIR
            # code below.
            temp_All = fair.run_FaIR(emissions_in=emis_FAIR,
                                     forcing_in=forc_FaIR,
                                     thermal_parameters=params_FaIR,
                                     show_run_info=False)['T'].to_numpy()

            # Remove pre-industrial offset before regression
            if inc_pi_offset:
                _ofst = temp_All[(forc_Yrs >= start_pi) &
                                 (forc_Yrs <= end_pi), :
                                 ].mean(axis=0)
            else:
                _ofst = 0
            temp_Mod = temp_All[(forc_Yrs >= start_yr) &
                                (forc_Yrs <= end_yr)] - _ofst

            # Decide whether to include a Constant offset term in regression
            if inc_reg_const:
                temp_Mod = np.append(temp_Mod,
                                     np.ones((temp_Mod.shape[0], 1)),
                                     axis=1)
            n_reg_vars = temp_Mod.shape[1]

            coef_Obs_Results = np.empty((temp_Mod.shape[1],
                                         df_temp_Obs.shape[1]))
            coef_PiC_Results = np.empty((temp_Mod.shape[1],
                                         df_temp_PiC.shape[1]))

            c_i = 0
            for temp_Obs_Ens in df_temp_Obs.columns:
                temp_Obs_i = df_temp_Obs[temp_Obs_Ens].to_numpy()
                coef_Obs_i = np.linalg.lstsq(temp_Mod, temp_Obs_i,
                                             rcond=None)[0]
                coef_Obs_Results[:, c_i] = coef_Obs_i
                c_i += 1

            c_j = 0
            for temp_PiC_Ens in df_temp_PiC.columns:
                temp_PiC_j = df_temp_PiC[temp_PiC_Ens].to_numpy()
                coef_PiC_j = np.linalg.lstsq(temp_Mod, temp_PiC_j,
                                             rcond=None)[0]
                coef_PiC_Results[:, c_j] = coef_PiC_j
                c_j += 1

            for c_k in range(coef_Obs_Results.shape[1]):
                for c_l in range(coef_PiC_Results.shape[1]):
                    # Regression coefficients
                    coef_Reg = (coef_Obs_Results[:, c_k] +
                                coef_PiC_Results[:, c_l])
                    # Attributed warming for each component
                    temp_Att = temp_Mod * coef_Reg

                    # Extract T_Obs and T_PiC data for this c_i, c_j combo.
                    temp_Obs_kl = df_temp_Obs[df_temp_Obs.columns[c_k]
                                              ].to_numpy()
                    # temp_PiC_kl = df_temp_PiC[df_temp_PiC.columns[c_l]
                    #                           ].to_numpy()

                    # Save outputs from the calculation:
                    # Regression coefficients
                    coef_Reg_Results[:, i] = coef_Reg

                    # Attributed warming for each component
                    temp_Att_Results[:, :(n_reg_vars-(1*inc_reg_const)), i] = \
                        temp_Att[:, :-1]

                    # # Actual piControl IV sample that used for this c_k, c_l
                    # temp_Att_Results[:, -2, i] = temp_PiC_kl
                    # # The temp_Obs (dependent var) for this c_k, c_l
                    # temp_Att_Results[:, -1, i] = temp_Obs_kl
                    # TOTAL
                    temp_Tot = temp_Att.sum(axis=1)
                    temp_Att_Results[:, -2, i] = temp_Tot
                    # RESIDUAL
                    temp_Att_Results[:, -1, i] = (temp_Obs_kl - temp_Tot)
                    # ANTROPOGENIC
                    temp_Ant = (temp_Att[:, variables.index('GHG')] + 
                                temp_Att[:, variables.index('OHF')])
                    temp_Att_Results[:, -3, i] = temp_Ant

                    # Visual display of pregress through calculation
                    if i % 1000 == 0:
                        percentage = int((i+1)/n*100)
                        loading_bar = (percentage // 5*'.' +
                                    (20 - percentage // 5)*' ')
                        print(f'calculating {loading_bar} {percentage}%', end='\r')
                    i += 1

    print(f"calculating {20*'.'} {100}%", end='\r')
    return temp_Att_Results, coef_Reg_Results, vars


def moving_average(data, w):
    """Calculate a moving average of data with window size w."""
    # data_padded = np.pad(data, (w//2, w-1-w//2),
    #                      mode='constant', constant_values=(0, 1.5))
    return np.convolve(data, np.ones(w), 'valid') / w


def temp_signal(data, w, method):
    """Calculate the temperature signal as moving average of window w."""
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


def final_value_of_trend(temp):
    """Used for calculating the SR15 definition of present-day warming."""
    
    """Pass a 15-year long timeseries to this function and it will compute
    a linear trend through it, and return the final value of it. This
    corresponds to the SR15 definition of warming, if the 'present-day' in
    consideration is the final observable year; the SR15 definition would
    extrapolate this linear trend for 15 more years and take the mid-value,
    which is simply the end value of the first 15 years."""

    """SR15 definition: 'warming at a given point in time is defined as the global average
    temperatures for a 30-year period centred on that time, extrapolating into
    the future if necessary'. For these calculations, therefore, we take the 
    final 15 years of the timeseries, take the trend through it, and then
    warming is given by the value of the trend in the final (present-day)
    year."""

    time = np.arange(temp.shape[0])
    fit = np.poly1d(np.polyfit(time, temp, 1))
    return fit(time)[-1]


###############################################################################
# MAIN CODE BODY ##############################################################
###############################################################################

if __name__ == "__main__":

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

    # model_choice = 'AR5_IR'
    model_choice = 'FaIR_V2'

    start_yr, end_yr = 1850, 2022
    start_pi, end_pi = 1850, 1900  # As in IPCC AR6 Ch-3 Fig-3.4

    # sigmas = [[32, 68], [5, 95], [0.3, 99.7]]
    sigmas = [[17, 83], [5, 95]]
    sigmas_all = list(np.concatenate((np.sort(np.ravel(sigmas)), [50]),
                                     axis=0))

    # plot_folder = 'plots/'
    plot_folder = 'plots/development/'


    ###########################################################################
    # READ IN THE DATA ########################################################
    ###########################################################################

    # ERF
    # forc_Group = load_ERF_Stuart()
    # forc_Group_names = sorted(list(forc_Group.keys()))
    # print(forc_Group_names)

    df_forc = load_ERF_CMIP6()
    forc_Group_names = sorted(
        df_forc.columns.get_level_values('variable').unique())

    # TEMPERATURE
    df_temp_Obs = load_HadCRUT(start_pi, end_pi)
    n_yrs = df_temp_Obs.shape[0]

    # CMIP6 PI-CONTROL
    timeframes = [1, 3, 30]
    # df_temp_PiC = load_PiC_Stuart(n_yrs)
    df_temp_PiC = load_PiC_CMIP6(n_yrs, start_pi, end_pi)
    df_temp_PiC = filter_PiControl(df_temp_PiC, timeframes)
    df_temp_PiC.set_index(np.arange(end_yr-start_yr+1)+1850, inplace=True)

    # Create a very rough estimate of the internal variability for the HadCRUT5
    # best estimate.
    # TODO: Regress natural forcings out of this as well...
    temp_Obs_signal = temp_signal(
        df_temp_Obs.quantile(q=0.5, axis=1).to_numpy(), 30, 'extrapolate')
    temp_Obs_IV = df_temp_Obs.quantile(q=0.5, axis=1) - temp_Obs_signal

    # PLOT THE INTERNAL VARIABILITY ###########################################
    fig = plt.figure(figsize=(15, 10))
    gr.running_mean_internal_variability(
        timeframes, df_temp_PiC, temp_Obs_IV)
    gr.overall_legend(fig, loc='lower center', ncol=2, nrow=False)
    fig.suptitle(
        'Selected Sample of Internal Variability from CMIP6 PiControl')
    fig.savefig(f'{plot_folder}0_Selected_CMIP6_Ensembles.png')

    #### PLOT THE ENSEMBLE ####
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid(shape=(1, 4), loc=(0, 0), rowspan=1, colspan=3)
    ax2 = plt.subplot2grid(shape=(1, 4), loc=(0, 3), rowspan=1, colspan=1)
    gr.plot_internal_variability_sample(
        ax1, ax2, df_temp_PiC, df_temp_Obs, temp_Obs_IV, sigmas, sigmas_all)
    gr.overall_legend(fig, loc='lower center', ncol=3, nrow=False)
    fig.suptitle(
        'Selected Sample of Internal Variability from CMIP6 pi-control')
    fig.savefig(f'{plot_folder}1_Distribution_Internal_Variability.png')

    ###########################################################################
    # CARRY OUT GWI CALCULATION ###############################################
    ############ Set model parameters #########################################
    if model_choice == 'AR5_IR':
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

    elif model_choice == 'FaIR_V2':
        # The original location of the FaIR tunings is here
        # CMIP6_param_csv = ('models/FaIR_V2/FaIRv2_0_0_alpha1/fair/util/' +
        #                    'parameter-sets/CMIP6_climresp.csv')
        # Which is simply copied to the following location for transparency
        # and convenience.
        CMIP6_param_csv = ('models/FaIR_CMIP6_climresp.csv')
        CMIP6_param_df = pd.read_csv(
            CMIP6_param_csv, index_col=[0], header=[0, 1])

    # Calculate GWI ###########################################################
    forc_Yrs = np.array(df_forc.index)
    temp_Yrs = np.array(df_temp_Obs.index)

    t1 = dt.datetime.now()
    calc_switch = input('Recalculate? y/n: ')

    if calc_switch == 'y':
        samples = int(input('Number of samples for each source (0-200): '))
        # Select random sub-set sampling of all ensemble members:

        # 1. Select random samples of the forcing data

        # print(f'Forcing ensembles all: {forc_Group["GHG"]["df"].shape[1]}')
        # forc_Group_subset_columns = forc_Group['GHG']['df'].sample(
        #     n=min(samples, forc_Group['GHG']['df'].shape[1]), axis=1).columns
        # forc_Group_subset = {
        #     var: {'df': forc_Group[var]['df'][forc_Group_subset_columns]}
        #     for var in forc_Group_names}
        # print(f'Forcing ensembles pruned: {forc_Group_subset["GHG"]["df"].shape[1]}')

        print(f'Forcing ensemble all: {len(df_forc.columns.levels[1])}')
        forc_sample = np.random.choice(
            df_forc.columns.levels[1],
            min(samples, len(df_forc.columns.levels[1])),
            replace=False)
        # select all variables for first column level,
        # and forc_sample for second column level
        forc_subset = df_forc.loc[:, (slice(None), forc_sample)]
        # forc_subset = df_forc.xs(tuple(forc_sample), axis=1, level=1)
        _nf = len(forc_subset.columns.get_level_values("ensemble").unique())
        print(f'Forcing ensemble pruned: {_nf}')

        # 2. Select all samples of the model parameters
        print('FaIR Parameters: '
              f'{len(CMIP6_param_df.columns.levels[0].unique())}')
        if model_choice == 'AR5_IR':
            params_subset = Geoff[:, :min(samples, Geoff.shape[1])]
        elif model_choice == 'FaIR_V2':
            params_subset = CMIP6_param_df
            models = CMIP6_param_df.columns.levels[0].unique().to_list()

        # 3. Select random samples of the temperature data
        print(f'Temperature ensembles all: {df_temp_Obs.shape[1]}')
        df_temp_Obs_subset = df_temp_Obs.sample(
            n=min(samples, df_temp_Obs.shape[1]), axis=1)
        print(f'Temperature ensembles pruned: {df_temp_Obs_subset.shape[1]}')

        # 4. Select random samples of the internal variability
        print(f'Internal variability ensembles all: {df_temp_PiC.shape[1]}')
        df_temp_PiC_subset = df_temp_PiC.sample(
            n=min(samples, df_temp_PiC.shape[1]), axis=1)
        print('Internal variability ensembles pruned: '
              f'{df_temp_PiC_subset.shape[1]}')
        
        T0 = dt.datetime.now()
        # temp_Att_Results, coef_Reg_Results, vars = GWI(
        #     forc_Group_names,
        #     inc_reg_const,
        #     forc_subset,
        #     params_subset,
        #     df_temp_PiC_subset,
        #     df_temp_Obs_subset,
        #     start_yr,
        #     end_yr)
        T1 = dt.datetime.now()

        with mp.Pool(os.cpu_count()) as p:
            print('Partialising Function')
            partial_GWI = functools.partial(
                GWI_faster,
                inc_reg_const=inc_reg_const,
                inc_pi_offset=inc_pi_offset,
                df_forc=forc_subset,
                df_params=params_subset,
                df_temp_PiC=df_temp_PiC_subset,
                df_temp_Obs=df_temp_Obs_subset,
                start_yr=start_yr,
                end_yr=end_yr,
                start_pi=start_pi,
                end_pi=end_pi,
            )
            print('Calculating GWI (parallelised)', end=' ')
            results = p.map(partial_GWI, models)
        vars = df_forc.columns.get_level_values(
            'variable').unique().to_list()
        vars.extend(['Ant', 'Tot', 'Res'])

        T1_1 = dt.datetime.now()
        print(f'... took {T1_1 - T1} seconds')  
        print('Concatenating Results', end=' ')
        temp_Att_Results = np.concatenate(results, axis=2)
        T2 = dt.datetime.now()
        print(f'... took {T2 - T1_1} seconds')

        # print(f'GWI original took {T1 - T0} seconds')                    
        n = temp_Att_Results.shape[2]
        n_parallel = temp_Att_Results.shape[2]

        # FILTER RESULTS ######################################################
        # For diagnosing: filter out results with particular regression
        # coefficients.

        # If you only want to study subsets of the results based on certain
        # constraints apply a mask here. The below mask is set to look at
        # results for different values of the coefficients.

        # Note to self about masking: coef_Reg_Results is the array of all
        # regression coefficients, with shape (4, n), where n is total number
        # of samplings. We select slice indices (forcing coefficients) we're
        # interested in basing the condition on:
        # AER is index 0, GHGs index 1, NAT index 2, Const index 3
        # Then choose whether you want any or all or the coefficients to meet
        # the condition (in this case being less than zero)
        mask_switch = False
        if mask_switch:
            coef_mask = np.all(coef_Reg_Results[[0, 2], :] <= 0, axis=0)
            # mask = np.any(coef_Reg_Results[[0], :] <= 0.0, axis=0)

            temp_Att_Results = temp_Att_Results[:, :, coef_mask]
            coef_Reg_Results = coef_Reg_Results[:, coef_mask]
            print('Shape of masked attribution results:',
                  temp_Att_Results.shape)

        # np.save('results/temp_Att_Results.npy', temp_Att_Results)
        # np.save('results/coef_Reg_Results.npy', coef_Reg_Results)

        # PRODUCE FINAL RESULTS DATASETS ######################################
        # Remove old results first
        files = os.listdir('results')
        csvs = [f for f in files if f.endswith('.csv')]
        for csv in csvs:
            os.remove('results/' + csv)

        # WARNING TO SELF: multidimensional np.percentile() changes the order
        # of the axes, so that the axis along which you took the percentiles is
        # now the first axis, and the other axes are the remaining axes. This
        # doesn't make any sense to me why this would be useful, but it is what
        # it is...

        # TIMESERIES RESULTS
        print('Calculating percentiles', end=' ')
        gwi_timeseries_array = np.percentile(
            temp_Att_Results, sigmas_all, axis=2)
        dict_Results = {
            (var, sigma):
            gwi_timeseries_array[sigmas_all.index(sigma), :, vars.index(var)]
            for var in vars for sigma in sigmas_all
        }
        df_Results = pd.DataFrame(dict_Results, index=temp_Yrs)
        df_Results.columns.names = ['variable', 'percentile']
        df_Results.index.name = 'Year'
        df_Results.to_csv(f'results/GWI_results_timeseries_{n}.csv')
        T3 = dt.datetime.now()
        print(f'... took {T3 - T2} seconds')

        # HEADLINE RESULTS
        print('Calculating headlines')

        # GWI-ANNUAL DEFINITION (SIMPLE VALUE IN A GIVEN YEAR)
        dfs = [df_Results.loc[[2017]], df_Results.loc[[2022]]]

        # SR15 DEFINITION (CENTRE OF 30-YEAR TREND)
        # Calculate the linear trend of the final 15 years of the timeseries
        # and use this to calculate the present-day warming
        print('Calculating SR15-definition temps', end=' ')

        for year in [2017, 2022]:
            years_SR15 = ((year-15 <= temp_Yrs) * (temp_Yrs <= year))
            temp_Att_Results_SR15_recent = temp_Att_Results[years_SR15, :, :]

            # Calculate SR15-definition warming for each var-ens combination
            # See SR15 Ch1 1.2.1
            # temp_Att_Results_SR15 = np.apply_along_axis(
            #     final_value_of_trend, 0, temp_Att_Results_SR15_recent)

            temp_Att_Results_SR15 = np.empty(
                temp_Att_Results_SR15_recent.shape[1:])
            for vv in range(temp_Att_Results_SR15_recent.shape[1]):
                print(vv)
                with mp.Pool(os.cpu_count()) as p:
                    times = [temp_Att_Results_SR15_recent[:, vv, ii]
                             for ii
                             in range(temp_Att_Results_SR15_recent.shape[2])]
                    results = p.map(final_value_of_trend, times)
                temp_Att_Results_SR15[vv, :] = np.array(results)

            # Obtain statistics
            gwi_headline_array = np.percentile(
                temp_Att_Results_SR15, sigmas_all, axis=1)
            dict_Results = {
                (var, sigma):
                gwi_headline_array[sigmas_all.index(sigma), vars.index(var)]
                for var in vars for sigma in sigmas_all
            }
            df_headlines_i = pd.DataFrame(
                dict_Results, index=[f'{year} (SR15 definition)'])
            df_headlines_i.columns.names = ['variable', 'percentile']
            df_headlines_i.index.name = 'Year'
            dfs.append(df_headlines_i)

        T4 = dt.datetime.now()
        print('... took', T4 - T3, 'seconds')


        # AR6 DEFINITION (DECADE MEAN)
        print('Calculating AR6-definition temps', end=' ')
        for years in [[2010, 2019], [2013, 2022]]:
            recent_years = ((years[0] <= temp_Yrs) * (temp_Yrs <= years[1]))
            temp_Att_Results_AR6 = \
                temp_Att_Results[recent_years, :, :].mean(axis=0)
            # Obtain statistics
            gwi_headline_array = np.percentile(
                temp_Att_Results_AR6, sigmas_all, axis=1)
            dict_Results = {
                (var, sigma):
                gwi_headline_array[sigmas_all.index(sigma), vars.index(var)]
                for var in vars for sigma in sigmas_all
            }
            df_headlines_i = pd.DataFrame(
                dict_Results, index=['-'.join([str(y) for y in years])])
            df_headlines_i.columns.names = ['variable', 'percentile']
            df_headlines_i.index.name = 'Year'
            dfs.append(df_headlines_i)

        df_headlines = pd.concat(dfs, axis=0)
        df_headlines.to_csv(f'results/GWI_results_headlines_{n}.csv')
        T5 = dt.datetime.now()
        print(f'... took {T5 - T4} seconds')

    
    # GWI MULTI PLOT ##########################################################
    # print('Creating GWI Multi Plot...')
    # fig = plt.figure(figsize=(15, 10))
    # ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), rowspan=2, colspan=3)
    # ax2 = plt.subplot2grid(shape=(3, 4), loc=(0, 3), rowspan=2, colspan=1)
    # ax3 = plt.subplot2grid(shape=(3, 4), loc=(2, 0), rowspan=1, colspan=3)
    # ax4 = plt.subplot2grid(shape=(3, 4), loc=(2, 3), rowspan=1, colspan=1)
    # ax2.set_ylim(ax1.get_ylim())
    # gr.gwi_timeseries(ax1,
    #                   df_temp_Obs, df_temp_PiC, df_Walsh_ts,
    #                   plot_vars, var_colours)
    # gr.gwi_residuals(ax3, df_Walsh_ts)
    # gr.gwi_tot_vs_ant(ax4, df_Walsh_ts)
    # gr.overall_legend(fig, 'lower center', 6)
    # fig.suptitle(f'GWI Timeseries Plot for {n} runs')
    # fig.savefig(f'{plot_folder}2_GWI_timeseries_multiplot.png')


    ###########################################################################
    # Recreate IPCC AR6 SPM.2 Plot
    # https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf
    ###########################################################################


    # # Create AR5 SPM2-esque comparison for the same data
    # fig = plt.figure(figsize=(12, 8))
    # ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=3)
    # gr.Fig_SPM2_validation_plot(
    #     ax, '2010-2019', bar_plot_vars, dict_updates_hl, source_colours)
    # gr.overall_legend(fig, 'lower center', len(dict_updates_hl.keys()))
    # fig.suptitle('Comparison of GWI to IPCC AR6 SPM.2 Assessment')
    # fig.savefig(f'{plot_folder}4-0_SPM2_Comparison_2010-2019.png')

    # # Create updated AR6 SPM2-esque plot containing results for both
    # # AR6 and SR15 definitions for present-day warming
    # fig = plt.figure(figsize=(12, 8))
    # ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
    # # ax2 = plt.subplot2grid(shape=(1, 4), loc=(0, 1), rowspan=1, colspan=3)
    # gr.Fig_SPM2_results_plot(
    #     ax=ax,
    #     periods=['2013-2022', '2022'],
    #     vars=bar_plot_vars,
    #     dict_dfs={'Walsh': df_Walsh_hl, 'Ribes': df_Ribes_hl, },
    #     period_cols=period_colours
    #     )
    # gr.overall_legend(fig, 'lower center', 3)
    # fig.suptitle('Assessed contributions to warming relative to 1850–1900')
    # fig.savefig(f'{plot_folder}4-1_SPM2_Update_2022.png')


    sys.exit()

    # # # Plot coefficients #########################################################
    # print('Creating Coefficient Plot...', end=' ')
    # fig = plt.figure(figsize=(15, 10))
    # ax = plt.subplot2grid(
    #     shape=(1, 1), loc=(0, 0),
    #     # rowspan=1, colspan=3
    #     )

    # ax.scatter(coef_Reg_Results[0, :], coef_Reg_Results[1, :],
    #         #    color=use_colours,
    #         color='xkcd:teal',
    #         alpha=0.01, edgecolors='none', s=20)
    # ax.set_xlabel('OHF')
    # ax.set_ylabel('GHG')
    # # plt.ylim(bottom=0)
    # fig.suptitle(f'Coefficients from {n} Samplings')
    # fig.savefig(f'{plot_folder}3_Coefficients.png')
    # t5 = dt.datetime.now()
    # print(f'took {t5-t4}')

    # ########### TEST SEABORN
    # plt.close()

    # coef_df = pd.DataFrame(coef_Reg_Results.T,
    #                     columns=['GHG', 'Nat', 'OHF', 'Const'])
    # # print(coef_df.head())
    # g = sns.PairGrid(coef_df.sample(5000))
    # g.map_upper(sns.scatterplot)
    # g.map_lower(sns.kdeplot)
    # g.map_diag(sns.kdeplot, lw=3, legend=False)
    # g.fig.suptitle('Regression Coefficient Distributions')
    # plt.savefig(f'{plot_folder}SNS_TEST.png')



    # ############### WHAT IS UP WITH THE NEGATIVE COEFFICIENT FITS...

    # for example in range(temp_Att_Results.shape[2]):
    #     plt.plot(temp_Yrs, temp_TEST_Sig_Results[:, example],
    #         color='black',
    #         label='Temp Observation Signal')
    #     plt.plot(temp_Yrs, temp_TEST_Ens_Results[:, example],
    #         color='black',
    #         label='Temp Observation Signal + Internal Variability')
        
    #     for i in range(len(forc_Group_names)):
    #         plt.plot(temp_Yrs, temp_Att_Results[:, i, example],
    #             color=forc_Group[forc_Group_names[i]]['Colour'],
    #             label=str(forc_Group_names[i])
    #             )
    #     plt.plot(temp_Yrs, temp_TOT_Results[:, example],
    #                 color='purple',
    #                 label='TOT')
    #     plt.title(coef_Reg_Results[:, i])

    #     plt.legend()
    #     plt.show()

    # sys.exit()