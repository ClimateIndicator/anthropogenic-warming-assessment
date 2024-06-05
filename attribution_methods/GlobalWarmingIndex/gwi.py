"""Script to generate global warming index."""

import os
import sys

import datetime as dt
import functools
import multiprocessing as mp

import numpy as np
import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt
# import scipy.stats as ss

import src.graphing as gr
import src.definitions as defs

import models.AR5_IR as AR5_IR
import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair

###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
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

    # Prepare results #########################################################
    n = (df_temp_Obs.shape[1] * df_temp_PiC.shape[1] *
         len(df_forc.columns.get_level_values("ensemble").unique()) *
         # only use 1 FaIR parameter set per function call in parallel method:
         1
         )
    # Include residuals and totals for sum total and anthropogenic warming in
    # the same array as attributed results. +1 each for Ant, TOT, Res,
    # InternalVariability, ObservedTemperatures
    # NOTE: the order in dimension is:
    # vars = ['GHG', 'Nat', 'OHF', 'Ant', 'Tot', 'Res']
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
        # Cut the full-forcing-length temperatures (that were calculated from
        # ERF) down to the same length as the other temperature data
        yr_mask = ((forc_Yrs >= start_yr) & (forc_Yrs <= end_yr))
        temp_Mod = temp_Mod_array[yr_mask, :, ens]

        # Remove pre-industrial offset before regression
        if inc_pi_offset:
            _ofst = temp_Mod[(temp_Yrs >= start_pi) &
                             (temp_Yrs <= end_pi), :
                             ].mean(axis=0)
            temp_Mod = temp_Mod - _ofst

        # Toggle whether to include a Constant offset term in regression
        if inc_reg_const:
            temp_Mod = np.append(temp_Mod,
                                 np.ones((temp_Mod.shape[0], 1)),
                                 axis=1)
        n_reg_vars = temp_Mod.shape[1]

        coef_Obs_Results = np.empty((temp_Mod.shape[1],
                                     df_temp_Obs.shape[1]))
        coef_PiC_Results = np.empty((temp_Mod.shape[1],
                                     df_temp_PiC.shape[1]))

        # Regress against observations
        c_i = 0
        for temp_Obs_Ens in df_temp_Obs.columns:
            temp_Obs_i = df_temp_Obs[temp_Obs_Ens].to_numpy()
            coef_Obs_i = np.linalg.lstsq(temp_Mod, temp_Obs_i, rcond=None)[0]
            coef_Obs_Results[:, c_i] = coef_Obs_i
            c_i += 1

        # Regress against piControl
        c_j = 0
        for temp_PiC_Ens in df_temp_PiC.columns:
            temp_PiC_j = df_temp_PiC[temp_PiC_Ens].to_numpy()
            coef_PiC_j = np.linalg.lstsq(temp_Mod, temp_PiC_j, rcond=None)[0]
            coef_PiC_Results[:, c_j] = coef_PiC_j
            c_j += 1

        # Combine regression coefficients from observations and piControl
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
    # the same array as attributed results. +1 each for Ant, Tot, Res
    # NOTE: the order in dimension is:
    # vars = ['GHG', 'Nat', 'OHF', 'Ant', 'Tot', 'Res']
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
        # Since the above line seems to get rid of the top column level (the
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

            # Regree against observations
            c_i = 0
            for temp_Obs_Ens in df_temp_Obs.columns:
                temp_Obs_i = df_temp_Obs[temp_Obs_Ens].to_numpy()
                coef_Obs_i = np.linalg.lstsq(temp_Mod, temp_Obs_i,
                                             rcond=None)[0]
                coef_Obs_Results[:, c_i] = coef_Obs_i
                c_i += 1
            
            # Regress against piControl
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
                        print(f'calculating {loading_bar} {percentage}%',
                              end='\r')
                    i += 1

    print(f"calculating {20*'.'} {100}%", end='\r')
    return temp_Att_Results, coef_Reg_Results


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
    # Following discussion with Myles, fix the regression options as the
    # following:
    inc_pi_offset = 'y'
    inc_reg_const = 'y'

    if inc_pi_offset not in allowed_options:
        print(f'{inc_pi_offset} not one of {ao}')
        sys.exit()
    elif inc_reg_const not in allowed_options:
        print(f'{inc_reg_const} not one of {ao}')
        sys.exit()

    inc_pi_offset = True if inc_pi_offset == 'y' else False
    inc_reg_const = True if inc_reg_const == 'y' else False

    # model_choice = 'AR5_IR'
    model_choice = 'FaIR_V2'

    start_yr, end_yr = 1850, 2023
    start_pi, end_pi = 1850, 1900  # As in IPCC AR6 Ch.3

    # sigmas = [[32, 68], [5, 95], [0.3, 99.7]]
    sigmas = [[17, 83], [5, 95]]
    sigmas_all = list(
        np.concatenate((np.sort(np.ravel(sigmas)), [50]), axis=0)
        )

    plot_folder = 'plots/'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    ###########################################################################
    # READ IN THE DATA ########################################################
    ###########################################################################

    # Effective Radiative Forcing
    df_forc = defs.load_ERF_CMIP6()
    forc_Group_names = sorted(
        df_forc.columns.get_level_values('variable').unique())

    # TEMPERATURE
    df_temp_Obs = defs.load_HadCRUT(start_pi, end_pi, start_yr, end_yr)
    n_yrs = df_temp_Obs.shape[0]

    # CMIP6 PI-CONTROL
    timeframes = [1, 3, 30]
    # df_temp_PiC = load_PiC_Stuart(n_yrs)
    df_temp_PiC = defs.load_PiC_CMIP6(n_yrs, start_pi, end_pi)
    df_temp_PiC = defs.filter_PiControl(df_temp_PiC, timeframes)
    df_temp_PiC.set_index(np.arange(n_yrs)+start_yr, inplace=True)
    # NOTE: For end_yr=2022, we get 183 realisations for piControl
    # NOTE: For end_yr=2023, we get 181 realisations for piControl

    # Create a very rough estimate of the internal variability for the HadCRUT5
    # best estimate.
    # TODO: Regress natural forcings out of this as well...
    temp_Obs_signal = defs.temp_signal(
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

    # PLOT THE ENSEMBLE #######################################################
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
        # AR5 thermal sensitivity coeffs
        # a_ar5[10:12] = [0.631, 0.429]
        # AR5 thermal time-inc_Constants -- could use Geoffroy et al [4.1,249.]
        # a_ar5[15:17] = [8.400, 409.5]
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
        # Which is simply copied to the following location and committed
        # to repo explicitly for transparency and convenience.
        CMIP6_param_csv = ('models/FaIR_CMIP6_climresp.csv')
        CMIP6_param_df = pd.read_csv(
            CMIP6_param_csv, index_col=[0], header=[0, 1])

    # Calculate GWI ###########################################################
    forc_Yrs = np.array(df_forc.index)
    temp_Yrs = np.array(df_temp_Obs.index)

    t1 = dt.datetime.now()
    # calc_switch = input('Recalculate? y/n: ')
    # if calc_switch == 'y':

    if len(sys.argv) > 1:
        # Use command line arguments instead of interactivity, so that we can
        # run the script using nohup.
        samples = int(sys.argv[1])
    else:
        # If no command line argument passed, use interactivity.
        samples = int(input('Max number of samples for each source (0-200): '))

    # Select random sub-set sampling of all ensemble members:

    # 1. Select random samples of the forcing data
    print(f'Forcing ensemble all: {len(df_forc.columns.levels[1])}')
    forc_sample = np.random.choice(
        df_forc.columns.levels[1],
        min(samples, len(df_forc.columns.levels[1])),
        replace=False)
    # select all variables for first column level, and forc_sample for
    # second column level
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
    
    # Print the total available ensemble size
    _n_all = (
        len(df_forc.columns.levels[1].unique()) *
        len(CMIP6_param_df.columns.levels[0].unique()) *
        df_temp_Obs.shape[1] *
        df_temp_PiC.shape[1]
    )
    print(f'Max available ensemble: {_n_all}')
    # Print the randomly subsampled ensemble size
    _n_sub = (
        _nf *  # number of forcing ensembles
        len(CMIP6_param_df.columns.levels[0].unique()) *
        df_temp_Obs_subset.shape[1] *
        df_temp_PiC_subset.shape[1]
    )
    print(f'Sub-sampled ensemble size: {_n_sub}')

    # Parallelise GWI calculation, with each thread corresponding to a
    # single (model) parameterisation for FaIR.
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

    vars = df_forc.columns.get_level_values('variable').unique().to_list()
    vars.extend(['Ant', 'Tot', 'Res'])

    T1_1 = dt.datetime.now()
    print(f'... took {T1_1 - T1}')
    print('Concatenating Results', end=' ')
    temp_Att_Results = np.concatenate(results, axis=2)
    T2 = dt.datetime.now()
    print(f'... took {T2 - T1_1}')

    n = temp_Att_Results.shape[2]

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
        print('Shape of masked attribution results:', temp_Att_Results.shape)

    # PRODUCE FINAL RESULTS DATASETS ######################################
    # Remove old results first
    if not os.path.exists('results'):
        os.makedirs('results')
    files = os.listdir('results')
    # csvs = [f for f in files if f.endswith('.csv')]
    # for csv in csvs:
    #     os.remove('results/' + csv)

    current_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    iteration = current_time

    # NOTE TO SELF: multidimensional np.percentile() changes the order of
    # the axes, so that the axis along which you took the percentiles is
    # now the first axis, and the other axes are the remaining axes...

    # TIMESERIES RESULTS
    print('Calculating percentiles', end=' ')
    gwi_timeseries_array = np.percentile(temp_Att_Results, sigmas_all, axis=2)
    dict_Results = {
        (var, sigma):
        gwi_timeseries_array[sigmas_all.index(sigma), :, vars.index(var)]
        for var in vars for sigma in sigmas_all
    }
    df_Results = pd.DataFrame(dict_Results, index=temp_Yrs)
    df_Results.columns.names = ['variable', 'percentile']
    df_Results.index.name = 'Year'
    df_Results.to_csv(f'results/GWI_results_timeseries_{n}_{iteration}.csv')
    T3 = dt.datetime.now()
    print(f'... took {T3 - T2}')

    # HEADLINE RESULTS
    print('Calculating headlines')

    # GWI-ANNUAL DEFINITION (SIMPLE VALUE IN A GIVEN YEAR)
    dfs = [df_Results.loc[[2017]], df_Results.loc[[end_yr]]]

    # SR15 DEFINITION (CENTRE OF 30-YEAR TREND)
    # Calculate the linear trend of the final 15 years of the timeseries
    # and use this to calculate the present-day warming
    print('Calculating SR15-definition temps', end=' ')

    for year in [2017, end_yr]:
        years_SR15 = ((year-15 <= temp_Yrs) * (temp_Yrs <= year))
        temp_Att_Results_SR15_recent = temp_Att_Results[years_SR15, :, :]

        # Calculate SR15-definition warming for each var-ens combination
        # See SR15 Ch1 1.2.1
        # temp_Att_Results_SR15 = np.apply_along_axis(
        #     final_value_of_trend, 0, temp_Att_Results_SR15_recent)
        temp_Att_Results_SR15 = np.empty(
            temp_Att_Results_SR15_recent.shape[1:])
        for vv in range(temp_Att_Results_SR15_recent.shape[1]):
            # print(vv)
            with mp.Pool(os.cpu_count()) as p:
                times = [temp_Att_Results_SR15_recent[:, vv, ii]
                            for ii
                            in range(temp_Att_Results_SR15_recent.shape[2])]
                # final_value_of_trend is from src/definitions.py
                results = p.map(defs.final_value_of_trend, times)
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
    print(f'... took {T4 - T3}')

    # AR6 DEFINITION (DECADE MEAN)
    print('Calculating AR6-definition temps', end=' ')
    for years in [[2010, 2019], [end_yr-9, end_yr]]:
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
    df_headlines.to_csv(f'results/GWI_results_headlines_{n}_{iteration}.csv')
    T5 = dt.datetime.now()
    print(f'... took {T5 - T4}')

    # RATE: AR6 DEFINITION
    if len(sys.argv) == 3 and sys.argv[-1] == 'include-rate':
        # Use command line arguments to trigger expensive rate calculation, so
        # that we can run the script using nohup.
        rate_toggle = True
    else:
        # If no command line argument passed, use interactivity as a backup.
        rate_toggle = input('Calculate rates? (y/n): ')
        rate_toggle = True if rate_toggle == 'y' else False

    if rate_toggle:
        T6 = dt.datetime.now()
        dfs_rates = []
        for year in np.arange(1950, end_yr+1):
            print(f'Calculating AR6-definition warming rate: {year}', end='\r')
            recent_years = ((year-9 <= temp_Yrs) * (temp_Yrs <= year))
            ten_slice = temp_Att_Results[recent_years, :, :]

            # Calculate AR6-definition warming rate for each var-ens combination
            # See AR6 WGI Chapter 3 Table 3.1
            temp_Rate_Results = np.empty(
                ten_slice.shape[1:])
            # Only include 'Ant'
            for vv in range(ten_slice.shape[1]):
            # Parallelise over ensemble members
                with mp.Pool(os.cpu_count()) as p:
                    single_series = [ten_slice[:, vv, ii]
                                    for ii in range(ten_slice.shape[2])]
                    # final_value_of_trend is from src/definitions.py
                    results = p.map(defs.rate_func, single_series)
                temp_Rate_Results[vv, :] = np.array(results)

            # Obtain statistics
            gwi_rate_array = np.percentile(
                temp_Rate_Results, sigmas_all, axis=1)
            dict_Results = {
                (var, sigma):
                gwi_rate_array[sigmas_all.index(sigma), vars.index(var)]
                for var in vars for sigma in sigmas_all
            }
            df_rates_i = pd.DataFrame(
                dict_Results, index=[f'{year-9}-{year} (AR6 rate definition)'])
            df_rates_i.columns.names = ['variable', 'percentile']
            df_rates_i.index.name = 'Year'
            dfs_rates.append(df_rates_i)

        df_rates = pd.concat(dfs_rates, axis=0)
        df_rates.to_csv(f'results/GWI_results_rates_{n}_{iteration}.csv')
        T7 = dt.datetime.now()
        print('')
        print(f'... took {T7 - T6}')