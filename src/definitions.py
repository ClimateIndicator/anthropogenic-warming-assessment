import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import functools
import xarray as xr
import glob
from pathlib import Path
import pymagicc
import time
from urllib.error import HTTPError, URLError


SUB_VAR_MAPPING = {
    'GHG': ['co2', 'ch4', 'n2o', 'halogen'],
    'OHF': ['aerosol-radiation_interactions', 'aerosol-cloud_interactions',
            'contrails', 'land_use', 'bc_snow', 'h2o_strat', 'o3'],
    'Nat': ['solar', 'volcanic'],
    'Ant': ['GHG', 'OHF'],
    'Tot': ['Ant', 'Nat']
}


VAR_NAMES = {
    'Obs': 'Observed warming',
    'Tot': 'Total forced warming',
    'Ant': 'Human-induced warming',
    'GHG': 'Well-mixed greenhouse gases',
    'OHF': 'Other human forcings',
    'Nat': 'All natural drivers',
    'Res': 'Residual (Internal variability)',
    'co2': 'Carbon dioxide',
    'ch4': 'Methane',
    'n2o': 'Nitrous oxide',
    'halogen': 'Halogenated gases',
    'aerosol-radiation_interactions': 'Aerosol-radiation interactions',
    'aerosol-cloud_interactions': 'Aerosol-cloud interactions',
    'land_use': 'Land-use reflectance',
    'bc_snow': 'Black carbon on snow',
    'h2o_strat': 'Stratospheric water vapour',
    'o3': 'Ozone',
    'solar': 'Solar',
    'volcanic': 'Volcanic',
    'contrails': 'Aviation contrails'
}


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################


def load_ERF_CMIP6(indicator_year, full_version=False):
    """Load the ERFs from Chris.

    Args:
        indicator_year (int): Final year of ERF file.
        full_version (bool): If True, require the *_full.nc dataset.
    """
    # ERF location
    here = Path(__file__).parent
    if full_version:
        file_ERF = here / (
            f'../data/ERF Samples/Chris/ERF_DAMIP_1000_1750-{indicator_year}_full.nc'
        )
    else:
        file_ERF = here / (
            f'../data/ERF Samples/Chris/ERF_DAMIP_1000_1750-{indicator_year}.nc'
        )

    if not file_ERF.exists():
        raise FileNotFoundError(f'ERF file not found: {file_ERF}')

    # import ERF_file to xarray dataset and convert to pandas dataframe
    df_ERF = xr.open_dataset(file_ERF).to_dataframe()
    # assign the columns the name 'variable'
    df_ERF.columns.names = ['variable']
    # remove the column called 'total' from df_ERF
    df_ERF = df_ERF.drop(columns='total', errors='ignore')
    # rename the variable columns
    df_ERF = df_ERF.rename(columns={
        'wmghg': 'GHG',
        'other_ant': 'OHF',
        'natural': 'Nat'
    })
    # move the multi-index 'ensemble' level to a column,
    # and then set the 'ensemble' column to second column level
    df_ERF = df_ERF.reset_index(level='ensemble')
    df_ERF['ensemble'] = 'ens' + df_ERF['ensemble'].astype(str)
    df_ERF = df_ERF.pivot(columns='ensemble')

    return df_ERF


def load_HadCRUT(start_pi, end_pi, start_yr, end_yr):
    """Load HadCRUT5 observations and remove PI baseline."""
    here = Path(__file__).parent
    temp_dir = here / f'../data/Temp/HadCRUT/'
    matches = sorted(
        temp_dir.glob('HadCRUT.*.analysis.ensemble_series.global.annual.csv')
    )
    if not matches:
        raise FileNotFoundError(
            f'No HadCRUT file found in {temp_dir} matching pattern '
            "'HadCRUT.*.analysis.ensemble_series.global.annual.csv'."
        )
    if len(matches) > 1:
        raise ValueError(
            f'Multiple HadCRUT files found in {temp_dir}; expected one match: '
            f'{[m.name for m in matches]}'
        )
    temp_ens_Path = matches[0]
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


def load_Temp_IGCC(start_pi, end_pi, end_yr):
    """Load IGCC observations and remove PI baseline."""
    here = Path(__file__).parent
    temp_Path = (
        '../data/Temp/IGCC/' +
        f'IGCC_data_series_{end_yr}.csv'
    )
    temp_Path = here / temp_Path
    # Read CSV and normalize headers to handle incidental whitespace.
    df_temp_Obs = pd.read_csv(temp_Path)
    df_temp_Obs.columns = df_temp_Obs.columns.str.strip()
    df_temp_Obs = df_temp_Obs.set_index('Year')
    # Select the column named 'GMST'
    df_temp_Obs = df_temp_Obs[['GMST']]
    # Calculate the mean of the years 1850-1900:
    df_temp_Obs_mean = df_temp_Obs.loc[
        (df_temp_Obs.index >= start_pi) &
        (df_temp_Obs.index <= end_pi),
        'GMST'].mean(axis=0)
    return df_temp_Obs


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


def rate_IGCC(start_pi, end_pi, start_yr, end_yr):
    df_temp_Obs = load_Temp_IGCC(start_pi, end_pi, end_yr)
    temp_Yrs = df_temp_Obs.index.values
    arr_temp_Obs = df_temp_Obs.values
    # Apply the function defs.rate_calc to each column of this dataframe
    dfs_rates = []
    for year in np.arange(1950, end_yr+1):
        print(year, end='\r')
        recent_years = ((year-9 <= temp_Yrs) * (temp_Yrs <= year))
        ten_slice = arr_temp_Obs[recent_years, :]
        results = np.array(rate_func(ten_slice))

        dict_Results = {('Obs', str(50)): results}
        df_rates_i = pd.DataFrame(
            dict_Results, index=[f'{year-9}-{year} (AR6 rate definition)'])
        df_rates_i.columns.names = ['variable', 'percentile']
        df_rates_i.index.name = 'Year'
        dfs_rates.append(df_rates_i)
        df_rates = pd.concat(dfs_rates, axis=0)
    return df_rates


def extra_vars(regress_vars):
    """Return aggregate variables that sit above regress_vars in hierarchy."""
    regress_vars = set(regress_vars)
    extra = set()

    changed = True
    while changed:
        changed = False
        for parent, children in SUB_VAR_MAPPING.items():
            if parent in regress_vars or parent in extra:
                continue
            if any((child in regress_vars) or (child in extra)
                   for child in children):
                extra.add(parent)
                changed = True

    return list(extra)


def map_var_to_regression_aggregate(var_list_ERF, regress_vars):
    """Map each ERF variable to the highest relevant regression aggregate."""
    extra_vars_list = extra_vars(regress_vars)
    reduced_mapping = {k: v for k, v in SUB_VAR_MAPPING.items()
                       if k not in extra_vars_list}

    def get_highest_parent(target, mapping):
        for parent, children in mapping.items():
            if target in children:
                return get_highest_parent(parent, mapping)
        return target

    mapped = {}
    for var in var_list_ERF:
        mapped[var] = get_highest_parent(var, reduced_mapping)

    for reg_var in regress_vars:
        mapped[reg_var] = reg_var

    return mapped


def rate_ERF(end_yr, sigmas_all, variable_mode='aggregate',
             regress_vars=None):
    if regress_vars is None:
        regress_vars = ['GHG', 'OHF', 'Nat']

    df_forc = load_ERF_CMIP6(
        end_yr, full_version=(variable_mode == 'all')
    )
    forc_Group_names = sorted(
        df_forc.columns.get_level_values('variable').unique())
    forc_Ens_names = sorted(
        df_forc.columns.get_level_values('ensemble').unique())
    forc_Yrs = df_forc.index.values

    forc_arrays = {var: df_forc[var].values for var in forc_Group_names}
    var_to_reg = map_var_to_regression_aggregate(forc_Group_names, regress_vars)

    # Build missing regression aggregates from children if needed.
    for reg_var in regress_vars:
        if reg_var not in forc_arrays:
            children = [var for var, parent in var_to_reg.items()
                        if parent == reg_var and var in forc_arrays]
            if children:
                forc_arrays[reg_var] = np.sum(
                    [forc_arrays[var] for var in children], axis=0
                )

    if ('GHG' in forc_arrays) and ('OHF' in forc_arrays):
        forc_arrays['Ant'] = forc_arrays['GHG'] + forc_arrays['OHF']

    if ('Ant' in forc_arrays) and ('Nat' in forc_arrays):
        forc_arrays['Tot'] = forc_arrays['Ant'] + forc_arrays['Nat']

    if variable_mode == 'aggregate':
        rate_vars = [var for var in ['Nat', 'GHG', 'OHF', 'Ant', 'Tot']
                     if var in forc_arrays]
    elif variable_mode == 'all':
        rate_vars = list(forc_Group_names)
        for reg_var in regress_vars:
            if (reg_var in forc_arrays) and (reg_var not in rate_vars):
                rate_vars.append(reg_var)
        for agg_var in ['Ant', 'Tot']:
            if (agg_var in forc_arrays) and (agg_var not in rate_vars):
                rate_vars.append(agg_var)
    else:
        raise ValueError(
            f"Unknown variable_mode='{variable_mode}'. "
            "Use 'aggregate' or 'all'."
        )

    # Apply the rate function to each selected variable/ensemble combination.
    dfs_rates = []
    arr_forc = np.stack([forc_arrays[var] for var in rate_vars], axis=1)

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


def read_csv_with_retries(
    file_location, retries=5, base_wait_seconds=1, **kwargs
):
    """Read a CSV with retry/backoff for transient remote fetch failures."""
    for attempt in range(retries):
        try:
            return pd.read_csv(file_location, **kwargs)
        except HTTPError as err:
            is_last_attempt = attempt == retries - 1
            if err.code != 429 or is_last_attempt:
                raise
            wait_seconds = base_wait_seconds * (2 ** attempt)
            print(
                f'HTTP 429 while fetching {file_location}; '
                f'retrying in {wait_seconds}s...'
            )
            time.sleep(wait_seconds)
        except URLError:
            if attempt == retries - 1:
                raise
            wait_seconds = base_wait_seconds * (2 ** attempt)
            print(
                f'Network error while fetching {file_location}; '
                f'retrying in {wait_seconds}s...'
            )
            time.sleep(wait_seconds)

    # Defensive fallback; the loop should have returned or raised.
    raise RuntimeError(
        f'Unable to fetch CSV after {retries} attempts: {file_location}'
    )
