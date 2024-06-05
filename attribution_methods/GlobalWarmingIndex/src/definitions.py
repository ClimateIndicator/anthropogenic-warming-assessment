import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import functools
import xarray as xr
import glob
from pathlib import Path
import pymagicc


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
# def load_ERF_Old(end_yr):
#     """Load the ERFs from Stuart's ERF datasets."""
#     here = Path(__file__).parent
#     forc_Path = here / '../data/ERF Samples/Stuart/'
#     # list_ERF = ['_'.join(file.split('_')[1:-1])
#     #             for file in os.listdir(forc_Path)
#     #             if '.csv' in file]
#     forc_Group = {
#                 #   'Ant': {'Consists': ['ant'], 'Colour': 'green'},
#                 'Nat': {'Consists': ['nat'],
#                         'Colour': 'green'},
#                 'GHG': {'Consists': ['co2', 'ch4', 'n2o', 'other_wmghg'],
#                         'Colour': 'orange'},
#                 'OHF': {'Consists': ['ari', 'aci', 'bc_on_snow', 'contrails',
#                                      'o3_tropospheric', 'o3_stratospheric',
#                                      'h2o_stratospheric', 'land_use'],
#                         'Colour': 'blue'}
#                 }

#     for grouping in forc_Group:
#         list_df = []
#         for element in forc_Group[grouping]['Consists']:
#             _df = pd.read_csv(
#                 forc_Path + f'rf_{element}_200samples.csv',
#                 skiprows=[1]
#                             ).rename(columns={'Unnamed: 0': 'Year'}
#                             ).set_index('Year')
#             list_df.append(_df.loc[_df.index <= end_yr])

#         forc_Group[grouping]['df'] = functools.reduce(lambda x, y: x.add(y),
#                                                       list_df)
#     return forc_Group


def load_ERF_CMIP6():
    """Load the ERFs from Chris."""
    # ERF location
    here = Path(__file__).parent
    file_ERF = here / '../data/ERF Samples/Chris/ERF_DAMIP_1000_1750-2023.nc'
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


def load_HadCRUT(start_pi, end_pi, start_yr, end_yr):
    """Load HadCRUT5 observations and remove PI baseline."""
    here = Path(__file__).parent
    temp_ens_Path = (
        '../data/Temp/HadCRUT/' +
        'HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.csv')
    temp_ens_Path = here / temp_ens_Path
    # read temp_Path into pandas dataframe, rename column 'Time' to 'Year'
    # and set the index to 'Year', keeping only columns with 'Realization' in
    # the column name, since these are the ensembles
    df_temp_Obs = pd.read_csv(temp_ens_Path,
                              ).rename(columns={'Time': 'Year'}
                                       ).set_index('Year'
                                                   ).filter(regex='Realization'
                                                            )

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

    # Filter only years between start_yr and end_yr
    df_temp_Obs = df_temp_Obs.loc[
        (df_temp_Obs.index >= start_yr) &
        (df_temp_Obs.index <= end_yr),
        ]

    return df_temp_Obs


# def load_PiC_Old(n_yrs):
#     """Load piControl data from Stuart's ERF datasets."""
#     here = Path(__file__).parent
#     file_PiC = here / '../data/piControl/piControl.csv'

#     df_temp_PiC = pd.read_csv(file_PiC
#                           ).rename(columns={'year': 'Year'}
#                                    ).set_index('Year')
#     # model_names = list(set(['_'.join(ens.split('_')[:1])
#     #                         for ens in list(df_temp_PiC)]))

#     temp_IV_Group = {}

#     for ens in list(df_temp_PiC):
#         # pi Control data located all over the place in csv; the following
#         # lines strip the NaN values, and limits slices to the same length as
#         # observed temperatures
#         temp = df_temp_PiC[ens].dropna().to_numpy()[:n_yrs]

#         # Remove pre-industrial mean period; this is done because the models
#         # use different "zero" temperatures (eg 0, 100, 287, etc).
#         # An alternative approach would be to simply subtract the first value
#         # to start all models on 0; the removal of the first 50 years
#         # is used here in case the models don't start in equilibrium (and
#         # jump up by x degrees at the start, for example), and the baseline
#         # period is just defined as the same as for the observation PI
#         # period.
#         temp -= temp[:start_pi-end_pi+1].mean()

#         if len(temp) == n_yrs:
#             temp_IV_Group[ens] = temp

#     return pd.DataFrame(temp_IV_Group)


def load_PiC_CMIP6(n_yrs, start_pi, end_pi):
    """Create DataFrame of piControl data from .MAG files."""
    # Create list of all .MAG files recursively inside the directory
    # data/piControl/CMIP6. These files are simply as extracted from zip
    # downloaded from https://cmip6.science.unimelb.edu.au/results?experiment_id=piControl&normalised=&mip_era=CMIP6&timeseriestype=average-year-mid-year&variable_id=tas&region=World#download
    # (ie a CMIP6 archive for pre-meaned data, saving data/time.)
    here = Path(__file__).parent
    path_PiC = here / '../data/piControl/CMIP6/**/*.MAG'
    path_PiC = str(path_PiC)
    mag_files = sorted(glob.glob(path_PiC, recursive=True))
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
        # variability of a CMIP6 ensemble must operate within certain bounds:
        # 1. there must be a minimum level of variation (to remove those models
        # that are clearly wrong, eg oscillating between 0.01 and 0 warming)
        # 2. they must not exceed a certain min or max temperature bound; the
        # 0.3 value is roughly similar to a 0.15 drift per century limit as
        # used in Haustein et al 2017, and Leach et al 2021.
        #
        # The final ensemble distribution are plotted against HadCRUT5 median
        # in gwi.py, to check that the percentiles of this median run are
        # similar to the percentiles on the entire CMIP5 ensemble. ie, if the
        # observed internal variability is essentially a sampling of the
        # climate each year, you would expect the percentiles over the observed
        # history to be similar to the percentiles across the ensemble (ie
        # multiple parallel realisations of reality) in any given year. We
        # allow the ensemble to be slightly broader, to reasonably allow for a
        # wider range of behaviours than we have so far seen in the real world.
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
    """Used for calculating the SR1.5 definition of present-day warming."""

    """Pass a 15-year long timeseries to this function and it will compute
    a linear trend through it, and return the final value of the trend. This
    corresponds to the SR15 definition of warming, if the 'present-day' in
    consideration is the final observable year; the SR15 definition would
    extrapolate this linear trend for 15 more years and take the mid-value,
    which is simply the end value of the first 15 years."""

    """SR1.5 definition: 'warming at a given point in time is defined as the
    global average temperatures for a 30-year period centred on that time,
    extrapolating into the future if necessary'. For these calculations,
    therefore, we take the final 15 years of the timeseries, take the trend
    through it, and then warming is given by the value of the trend in the
    final (present-day) year."""

    time = np.arange(temp.shape[0])
    fit = np.poly1d(np.polyfit(time, temp, 1))
    return fit(time)[-1]


def rate_func(array):
    # Instead of passing years array, just set the start year for the slice
    # to zero
    times = np.arange(array.shape[0])
    fit = np.polyfit(x=times, y=array, deg=1)
    return fit[0]


def rate_HadCRUT5(start_pi, end_pi, start_yr, end_yr, sigmas_all):
    # Load the HadCRUT5 dataset
    df_temp_Obs = load_HadCRUT(start_pi, end_pi, start_yr, end_yr)
    temp_Yrs = df_temp_Obs.index.values
    arr_temp_Obs = df_temp_Obs.values
    # Apply the function defs.rate_calc to each column of this dataframe

    dfs_rates = []
    for year in np.arange(1950, end_yr+1):
        print(year, end='\r')
        recent_years = ((year-9 <= temp_Yrs) * (temp_Yrs <= year))
        ten_slice = arr_temp_Obs[recent_years, :]

        with mp.Pool(os.cpu_count()) as p:
            single_series = [ten_slice[:, ii]
                             for ii in range(ten_slice.shape[-1])]
            results = p.map(rate_func, single_series)
        forc_Rate_results = np.array(results)

        # Obtain statistics
        obs_rate_array = np.percentile(
            forc_Rate_results, sigmas_all, axis=0)
        dict_Results = {
            ('Obs', str(sigma)): obs_rate_array[sigmas_all.index(sigma)]
            for sigma in sigmas_all}
        df_rates_i = pd.DataFrame(
            dict_Results, index=[f'{year-9}-{year} (AR6 rate definition)'])
        df_rates_i.columns.names = ['variable', 'percentile']
        df_rates_i.index.name = 'Year'
        dfs_rates.append(df_rates_i)
    df_rates = pd.concat(dfs_rates, axis=0)
    return df_rates


def rate_ERF(end_yr, sigmas_all):
    rate_vars = ['Nat', 'GHG', 'OHF', 'Ant', 'Tot']
    df_forc = load_ERF_CMIP6()
    forc_Group_names = sorted(
        df_forc.columns.get_level_values('variable').unique())
    forc_Ens_names = sorted(
        df_forc.columns.get_level_values('ensemble').unique())
    forc_Yrs = df_forc.index.values

    # Apply the function defs.rate_calc to each column of this dataframe
    dfs_rates = []
    arr_forc = np.empty(
        (len(forc_Yrs), len(forc_Group_names)+2, len(forc_Ens_names)))
    # Move the data for each forcing group into a separate array dimension
    for vv in forc_Group_names:
        arr_forc[:, rate_vars.index(vv), :] = df_forc[vv].values
    arr_forc[:, rate_vars.index('Ant'), :] = (
        arr_forc[:, rate_vars.index('GHG'), :] +
        arr_forc[:, rate_vars.index('OHF'), :])
    arr_forc[:, rate_vars.index('Tot'), :] = (
        arr_forc[:, rate_vars.index('Ant'), :] +
        arr_forc[:, rate_vars.index('Nat'), :]
    )

    for year in np.arange(1950, end_yr+1):
        print(f'Calculating AR6-definition ERF rate: {year}', end='\r')
        recent_years = ((year-9 <= forc_Yrs) * (forc_Yrs <= year))
        ten_slice = arr_forc[recent_years, :, :]

        # Calculate AR6-definition ERF rate for each var-ens combination
        forc_Rate_results = np.empty(
            ten_slice.shape[1:])
        # Only include 'Ant'
        for vv in range(ten_slice.shape[1]):
            # Parallelise over ensemble members
            with mp.Pool(os.cpu_count()) as p:
                single_series = [ten_slice[:, vv, ii]
                                 for ii in range(ten_slice.shape[2])]
                # final_value_of_trend is from src/definitions.py
                results = p.map(rate_func, single_series)
            forc_Rate_results[vv, :] = np.array(results)

        # Obtain statistics
        forc_rate_array = np.percentile(
            forc_Rate_results, sigmas_all, axis=1)
        dict_Results = {
            (var, str(sigma)):
            forc_rate_array[sigmas_all.index(sigma), rate_vars.index(var)]
            for var in rate_vars for sigma in sigmas_all
        }
        df_rates_i = pd.DataFrame(
            dict_Results, index=[f'{year-9}-{year} (AR6 rate definition)'])
        df_rates_i.columns.names = ['variable', 'percentile']
        df_rates_i.index.name = 'Year'
        dfs_rates.append(df_rates_i)
    print('')

    df_forc_rates = pd.concat(dfs_rates, axis=0)
    return df_forc_rates


def en_dash_ify(df):
    r"""Replace - with \N{EN DASH} in date danges in dataframes."""
    """This is required by ESSD formatting"""
    # List the rows with a - character in them
    rows_to_rename = [r for r in df.index if '-' in r]
    # Rename those rows, replacing the - with a \N{EN DASH}
    df.rename(
        index={r: r.replace('-', '\N{EN DASH}') for r in rows_to_rename},
        inplace=True)
    return df


def un_en_dash_ify(df):
    r"""Replace \N{EN DASH} with - in date danges in dataframes."""
    """For the purposes of saving to csv, where a normal '-' is likely safest
    for people to use, and most consistent with files from collaborators."""
    # List the rows with a - character in them
    rows_to_rename = [r for r in df.index if '\N{EN DASH}' in r]
    # Rename those rows, replacing the - with a \N{EN DASH}
    df.rename(
        index={r: r.replace('\N{EN DASH}', '-') for r in rows_to_rename},
        inplace=True)
    return df
