"""Script to generate global warming index."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import os
import sys
import functools
import scipy as sp
import seaborn as sns

matplotlib.rcParams.update(
    {'font.size': 11,
    #  'font.family': 'Helvetica',
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


# DEFINE AR5-IR MODEL
# Code copied from warming contributions

def EFmod(nyr, a):
    """Create linear operator to convert emissions to forcing."""
    Fcal = np.zeros((nyr, nyr))

    # extend time array to compute derivatives
    time = np.arange(nyr + 1)

    # compute inc_Constant term (if there is one, otherwise a[0]=0)
    F_0 = a[4] * a[13] * a[0] * time

    # loop over gas decay terms to calculate AGWP using AR5 formula
    for j in [1, 2, 3]:
        F_0 = F_0 + a[j] * a[4] * a[13] * a[j+5] * (1 - np.exp(-time / a[j+5]))

    # first-difference AGWP to obtain AGFP
    for i in range(0, nyr):
        Fcal[i, 0] = F_0[i+1]-F_0[i]

    # build up the rest of the Toeplitz matrix
    for j in range(1, nyr):
        Fcal[j:nyr, j] = Fcal[0:nyr-j, 0]

    return Fcal


def FTmod(nyr, a):
    """Create linear operator to convert forcing to warming."""
    Tcal = np.zeros((nyr, nyr))

    # shift time array to compute derivatives
    time = np.arange(nyr) + 0.5

    # loop over thermal response times using AR5 formula
    for j in [0, 1]:
        Tcal[:, 0] = Tcal[:, 0] + (a[j+10] / a[j+15]) * np.exp(-time / a[j+15])

    # build up the rest of the Toeplitz matrix
    for j in range(1, nyr):
        Tcal[j:nyr, j] = Tcal[0:nyr-j, 0]

    return Tcal


def ETmod(nyr, a):
    """Create linear operator to convert emissions to warming."""
    Tcal = np.zeros((nyr, nyr))

    # add one to the time array for consistency with AR5 formulae
    time = np.arange(nyr) + 1

    # loop over thermal response times using AR5 formula for AGTP
    for j in [0, 1]:
        Tcal[:, 0] = Tcal[:, 0] + a[4] * a[13] * \
            a[0] * a[j+10] * (1 - np.exp(-time / a[j+15]))

        # loop over gas decay terms using AR5 formula for AGTP
        for i in [1, 2, 3]:
            Tcal[:, 0] = Tcal[:, 0]+a[4]*a[13]*a[i]*a[i+5]*a[j+10] * \
                (np.exp(-time/a[i+5])-np.exp(-time/a[j+15]))/(a[i+5]-a[j+15])

    # build up the rest of the Toeplitz matrix
    for j in range(1, nyr):
        Tcal[j:nyr, j] = Tcal[0:nyr-j, 0]

    return Tcal


def a_params(gas):
    """Return the AR5 model parameter sets, in units GtCO2."""
    # First set up AR5 model parameters,
    # using syntax of FaIRv1.3 but units of GtCO2, not GtC

    m_atm = 5.1352 * 10**18  # AR5 official mass of atmosphere in kg
    m_air = 28.97 * 10**-3   # AR5 official molar mass of air
    # m_car = 12.01 * 10**-3   # AR5 official molar mass of carbon
    m_co2 = 44.01 * 10**-3   # AR5 official molar mass of CO2
    m_ch4 = 16.043 * 10**-3  # AR5 official molar mass of methane
    m_n2o = 44.013 * 10**-3  # AR5 official molar mass of nitrous oxide

    # scl = 1 * 10**3
    a_ar5 = np.zeros(20)

    # Set to AR5 Values for CO2
    a_ar5[0:4] = [0.21787, 0.22896, 0.28454, 0.26863]  # AR5 carbon cycle coefficients
    a_ar5[4] = 1.e12 * 1.e6 / m_co2 / (m_atm / m_air)  # old value = 0.471 ppm/GtC # convert GtCO2 to ppm
    a_ar5[5:9] = [1.e8, 381.330, 34.7850, 4.12370]     # AR5 carbon cycle timescales
    a_ar5[10:12] = [0.631, 0.429]                      # AR5 thermal sensitivity coeffs
    a_ar5[13] = 1.37e-2                                # AR5 rad efficiency in W/m2/ppm
    a_ar5[14] = 0
    a_ar5[15:17] = [8.400, 409.5]                      # AR5 thermal time-inc_Constants -- could use Geoffroy et al [4.1,249.]
    a_ar5[18:21] = 0

    ECS = 3.0
    a_ar5[10:12] *= ECS / np.sum(a_ar5[10:12]) / 3.7  # Rescale thermal sensitivity coeffs to prescribed ECS

    # Set to AR5 Values for CH4
    a_ch4 = a_ar5.copy()
    a_ch4[0:4] = [0, 1.0, 0, 0]
    a_ch4[4] = 1.e12 * 1.e9 / m_ch4 / (m_atm / m_air)  # convert GtCH4 to ppb
    a_ch4[5:9] = [1, 12.4, 1, 1]                       # methane lifetime
    a_ch4[13] = 1.65 * 3.6324360e-4                    # Adjusted radiative efficiency in W/m2/ppb

    # Set to AR5 Values for N2O
    a_n2o = a_ar5.copy()
    a_n2o[0:4] = [0, 1.0, 0, 0]
    a_n2o[4] = 1.e12 * 1.e9 / m_n2o / (m_atm / m_air)         # convert GtN2O to ppb
    a_n2o[5:9] = [1, 121., 1, 1]                              # N2O lifetime
    a_n2o[13] = (1.-0.36 * 1.65 * 3.63e-4 / 3.0e-3) * 3.0e-3  # Adjusted radiative efficiency in W/m2/ppb

    if gas == 'Carbon Dioxide':
        return a_ar5
    elif gas == 'Methane':
        return a_ch4
    elif gas == 'Nitrous Oxide':
        return a_n2o


def regress_single_gwi(forc_All, temp_Obs, params, offset=False):
    """Regress GWI timeseries for single given ERF and Observed Temperature."""
    
    temp_All = forc_All.copy()
    vars = list(forc_All)
    for var in vars:
        # Calculate the temperature from ERF
        _forc = forc_All[var]
        _temp = FTmod(forc_All.shape[0], params) @ _forc
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


# READ IN THE DATA
# df = pd.read_excel('.\data\otto_2016_gwi_excel.xlsx', sheet_name='Main',
#                    header=5, skiprows=[6])

start_yr, end_yr = 1850, 2022
start_pi, end_pi = 1850, 1900  # As in IPCC AR6 Ch-3 Fig-3.4

sigmas = [[32, 68], [5, 95], [0.3, 99.7]]


# Read Observed Temperatures
# year_Ind = df.loc[(df['Year'] >= start_yr) & (df['Year'] <= end_yr), 'Year']

# ofst_Obs = df.loc[(df['Year'] >= start_pi) & (df['Year'] <= end_pi),
#                   'Obs warm'].mean()
# temp_Obs = df.loc[(df['Year'] >= start_yr) & (df['Year'] <= end_yr),
#                   'Obs warm'] - ofst_Obs#

##############################################################################
# LOAD DATA
##############################################################################

### TEMPERATURE
## OBS
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
# n_yrs = obs_yrs + ma - 1
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
for ens in temp_IV_Group:
    plt.plot(moving_average(temp_IV_Group[ens], 30))
plt.savefig('0_Selected_CMIP5_Ensembles.png')
plt.close()


print('Number of CMIP5 internal variability samples remaining:' +
      f'{len(temp_IV_Group.keys())}')

# Include the internval variability (IV) from HadCRUT to the overall
# dictionary of internal variabilities.
# for _temp_Ens in df_temp_Obs:
#     # Pick every 10th realisation
#     if int(_temp_Ens.split(' ')[1]) % 10 == 0:
#         _temp_Obs_signal = temp_signal(df_temp_Obs[_temp_Ens], 30)
#         _temp_Obs_IV = df_temp_Obs[_temp_Ens] - _temp_Obs_signal
#         temp_IV_Group[f'HadCRUT5 {_temp_Ens}'] = np.array(_temp_Obs_IV)

temp_Obs_signal = temp_signal(df_temp_Obs.quantile(q=0.5, axis=1),
                              30, 'extrapolate')
temp_Obs_IV = df_temp_Obs.quantile(q=0.5, axis=1) - temp_Obs_signal
temp_IV_Group['HadCRUT5 median'] = np.array(temp_Obs_IV)


#### PLOT THE ENSEMBLE ####
_t = np.array([temp_IV_Group[ens] for ens in temp_IV_Group.keys()])

for p in sigmas:
    plt.fill_between(df_temp_Obs.index,
                     np.percentile(_t, p[0], axis=0),
                     np.percentile(_t, p[1], axis=0),
                     alpha=0.2, color='gray')
    if p == sigmas[-1]:
        plt.plot(df_temp_Obs.index,
                 np.percentile(_t, 50, axis=0),
                 color='gray',
                 label='CMIP5 piControl')


plt.plot(df_temp_Obs.index, temp_Obs_IV,
         label='HadCRUT_5_median')

for p in sigmas:
    plt.plot(df_temp_Obs.index,
             np.percentile(temp_Obs_IV, p[0]) * np.ones(len(df_temp_Obs.index)),
             color='pink')
    plt.plot(df_temp_Obs.index,
             np.percentile(temp_Obs_IV, p[1]) * np.ones(len(df_temp_Obs.index)),
             color='pink')
plt.plot(df_temp_Obs.index,
         np.percentile(temp_Obs_IV, 50) * np.ones(len(df_temp_Obs.index)),
         color='pink', label='HadCRUT5 percentiles')
plt.title('Ensemble of pruned CMIP5 piControl runs')
plt.legend()
plt.savefig('1_Distribution_Internal_Variability.png')
plt.close()


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
# for group in forc_Group:
#     plt.plot(forc_Group[group]['df'].quantile(q=0.5, axis=1), label=group)
# plt.legend()
# plt.show()
# sys.exit()

# df_forc_Ant = pd.read_csv(forc_Path + 'rf_ant_200samples.csv', skiprows=[1]
#                           ).rename(columns={'Unnamed: 0': 'Year'}
#                           ).set_index('Year')
# df_forc_Nat = pd.read_csv(forc_Path + 'rf_nat_200samples.csv', skiprows=[1]
#                           ).rename(columns={'Unnamed: 0': 'Year'}
#                           ).set_index('Year')
# df_forc_Ant = df_forc_Ant.loc[df_forc_Ant.index <= end_yr]
# df_forc_Nat = df_forc_Nat.loc[df_forc_Nat.index <= end_yr]


forc_All_ensemble_names = list(forc_Group[forc_Group_names[0]]['df'])



# # Read Anthropogenic and Natural Forcing
# df_forc = pd.read_excel('.\data\otto_2016_gwi_excel.xlsx', sheet_name='RF',
#                    header=59).rename(columns={"v YEARS/GAS >": 'Year'},
#                                      errors="raise")
# # print(df_forc.head())
# forc_GHG = df_forc.loc[(df_forc['Year'] >= 1765) & (df_forc['Year'] <= end_yr),
#                   'GHG_RF']
# forc_Ant = df_forc.loc[(df_forc['Year'] >= 1765) & (df_forc['Year'] <= end_yr),
#                   'TOTAL_ANTHRO_RF']

# df_forc['Ant'] = df_forc['TOTAL_ANTHRO_RF']
# df_forc['GHG'] = df_forc['GHG_RF']
# df_forc['Aer'] = df_forc['TOTAL_ANTHRO_RF'] - df_forc['GHG_RF']
# df_forc['Nat'] = df_forc['SOLAR_RF'] + df_forc['VOLCANIC_ANNUAL_RF']

# forc_All = df_forc.loc[(df_forc['Year'] >= start_yr) &
#                        (df_forc['Year'] <= end_yr),
#                        ['Year', 'Nat', 'GHG', 'Aer']]

# forc_All = df_forc.loc[(df_forc['Year'] >= start_yr) &
#                        (df_forc['Year'] <= end_yr),
#                        ['Year', 'Nat', 'Ant']]


############ Set model parameters #############################################
# We only use a[10], a[11], a[15], a[16]
# Defaults:
# a_ar5[10:12] = [0.631, 0.429]  # AR5 thermal sensitivity coeffs
# a_ar5[15:17] = [8.400, 409.5]  # AR5 thermal time-inc_Constants -- could use Geoffroy et al [4.1,249.]

# a_ar5 = np.zeros(20, 16)
# a_ar5[10:12] = [0.631, 0.429]
# a_ar5[15:17] = [8.400, 409.5]
# # Geoffrey 2013 paramters for a_ar5[15:17]
Geoff = np.array([[4.0, 5.0, 4.5, 2.8, 5.2, 3.9, 4.2, 3.6,
                   1.6, 5.3, 4.0, 5.5, 3.5, 3.9, 4.3, 4.0],
                  [126, 267, 193, 132, 289, 200, 317, 197,
                   184, 280, 698, 286, 285, 164, 150, 218]])


# Regress
samples = int(input('numer of samples (0-200): '))  # for temperature, and ERF
n = samples * samples * len(temp_IV_Group.keys()) * Geoff.shape[1]
print('Total number of combinations to sample: ' +
      f'{samples} forcings X {samples} HadCRUT X {len(temp_IV_Group.keys())}' +
      f' Internal Variability X {Geoff.shape[1]} Parameters = {n}')

###############################
# t1 = dt.datetime.now()
# i = 1

# for forc_Ens in forc_All_ensemble_names[:samples]:
#     for temp_Ens in temp_Obs_ensemble_names[:samples]:
#         # forc_All = pd.concat([df_forc_Ant[forc_Ens], df_forc_Nat[forc_Ens]], axis=1)
#         forc_All = pd.DataFrame({'Nat': df_forc_Nat[forc_Ens],
#                                  'Ant': df_forc_Ant[forc_Ens]})
#         temp_Obs = df_temp_Obs[temp_Ens]
#         temp_Att = regress_single_gwi(forc_All, temp_Obs,
#                                       a_params('Carbon Dioxide'),
#                                       offset=False)
#         # plt.plot(temp_Att_df)
#         # Visually show how far through the calculation we are
#         percentage = int(i/n*100)
#         loading_bar = percentage // 5*'.' + (20 - percentage // 5)*' '
#         print(f'calculating {loading_bar} {percentage}%', end='\r')
#         i += 1
# print('')
# t2 = dt.datetime.now()
# print(f'total caltulation took {t2-t1}')
# # plt.show()


################
forc_Yrs = np.array(forc_Group[forc_Group_names[0]]['df'].index)
temp_Yrs = np.array(df_temp_Obs.index)

temp_Att_Results = np.zeros(
    (173, len(forc_Group_names) + int(inc_reg_const), n))
# temp_Ens_unique = np.zeros((173, samples * len(temp_IV_Group.keys())))
temp_TOT_Residuals = np.zeros((173, n))

temp_TEST_Ens_Results = np.zeros((173, n))
temp_TEST_Sig_Results = np.zeros((173, n))

coef_Reg_Results = np.zeros((len(forc_Group_names) + int(inc_reg_const), n))
IV_names = []

i = 0
t1 = dt.datetime.now()

###############################################################################

for j in range(Geoff.shape[1]):
    params = np.zeros(20)
    params[10:12] = [0.631, 0.429]
    params[15:17] = Geoff[:, j]
    params[15:17] = [8.400, 409.5]
    
# for j in range(1):
#     params = np.zeros(20)
#     params[10:12] = [0.631, 0.429]
#     params[15:17] = Geoff[:, j]
#     params[15:17] = [8.400, 409.5]

    for forc_Ens in forc_All_ensemble_names[:samples]:
        forc_All = np.array([forc_Group[group]['df'][forc_Ens]
                        for group in forc_Group_names]).T
        # params = a_params('Carbon Dioxide')

        temp_All = FTmod(forc_All.shape[0], params) @ forc_All
        if inc_pi_offset:
            _ofst = temp_All[(forc_Yrs >= start_pi) & (forc_Yrs <= end_pi), :
                            ].mean(axis=0)
        else:
            _ofst = 0
        temp_Mod = temp_All[(forc_Yrs >= start_yr) & (forc_Yrs <= end_yr)] - _ofst

        # Decide whether to include a Constant offset term in regression
        if inc_reg_const:
            temp_Mod = np.append(temp_Mod, np.ones((temp_Mod.shape[0], 1)), axis=1)

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

                # Save the outputs from the calculation
                temp_TEST_Ens_Results[:, i] = temp_Ens
                temp_TEST_Sig_Results[:, i] = temp_Obs_signal
                coef_Reg_Results[:, i] = coef_Reg
                temp_Att_Results[:, :, i] = temp_Att
                temp_TOT_Residuals[:, i] = temp_Ens - temp_Att.sum(axis=1)
                IV_names.append('_'.join(temp_IV_Ens.split('_')[:1]))


                # Visual display of pregress through calculation
                percentage = int((i+1)/n*100)
                loading_bar = percentage // 5*'.' + (20 - percentage // 5)*' '
                print(f'calculating {loading_bar} {percentage}%', end='\r')
                i += 1

t2 = dt.datetime.now()
print(f'Total calculation took {t2-t1}')
print(len(set(IV_names)))
###############################################################################

# for forc_Ens in forc_All_ensemble_names[:samples]:
#     # Calculate forcing -> temperature response. The below steps calculate
#     # temperature for all sources simultaneously (eg Nat, GHG, and Aer)
#     # forc_All = np.array([df_forc_Nat[forc_Ens], df_forc_Ant[forc_Ens]]).T
#     forc_All = np.array([forc_Group[group]['df'][forc_Ens]
#                          for group in forc_Group_names]).T
#     params = a_params('Carbon Dioxide')
#     temp_All = FTmod(forc_All.shape[0], params) @ forc_All
#     if inc_pi_offset:
#         _ofst = temp_All[(forc_Yrs >= start_pi) & (forc_Yrs <= end_pi), :
#                          ].mean(axis=0)
#     else:
#         _ofst = 0
#     temp_Mod = temp_All[(forc_Yrs >= start_yr) & (forc_Yrs <= end_yr)] - _ofst
#     # temp_Mod = temp_All[(forc_Yrs >= start_yr) & (forc_Yrs <= end_yr)]

#     # Decide whether to include a Constant offset term in regression
#     if inc_reg_const:
#         temp_Mod = np.append(temp_Mod, np.ones((temp_Mod.shape[0], 1)), axis=1)

#     for temp_Ens in temp_Obs_ensemble_names[:samples]:
#         # Select the relevant observational data
#         temp_Obs = np.array(df_temp_Obs[temp_Ens])

#         # Carry out regression calculation
#         coef_Reg = np.linalg.lstsq(temp_Mod, temp_Obs, rcond=None)[0]
#         temp_Att = temp_Mod * coef_Reg
#         temp_Att_Results[:, :, i] = temp_Att
#         temp_TOT_Residuals[:, i] = temp_Att.sum(axis=1) - temp_Obs

#         # Visual display of pregress through calculation
#         percentage = int((i+1)/n*100)
#         loading_bar = percentage // 5*'.' + (20 - percentage // 5)*' '
#         print(f'calculating {loading_bar} {percentage}%', end='\r')
#         i += 1

# t2 = dt.datetime.now()
# print(f'Total calculation took {t2-t1}')


# Plot the data

fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), rowspan=2, colspan=3)
ax2 = plt.subplot2grid(shape=(3, 4), loc=(0, 3), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(3, 4), loc=(2, 0), rowspan=1, colspan=3)

# Plot the internal variability range:

temp_Ens_unique = np.unique(temp_TEST_Ens_Results, axis=1)
for p in sigmas:
    ax1.fill_between(df_temp_Obs.index,
                     np.percentile(temp_Ens_unique, p[0], axis=1),
                     np.percentile(temp_Ens_unique, p[1], axis=1),
                     alpha=0.05, color='black')
    if p == sigmas[-1]:
        ax1.plot(df_temp_Obs.index,
                 np.percentile(temp_Ens_unique, 50, axis=1),
                 color='black',
                 label='CMIP5 piControl')

# Plot the observed temperatures on top as a scatter
err_pos = (df_temp_Obs.quantile(q=0.95, axis=1) -
           df_temp_Obs.quantile(q=0.5, axis=1))
err_neg = (df_temp_Obs.quantile(q=0.5, axis=1) -
           df_temp_Obs.quantile(q=0.05, axis=1))
ax1.errorbar(temp_Yrs, df_temp_Obs.quantile(q=0.5, axis=1),
             yerr=(err_neg, err_pos),
             fmt='o', color='black', ms=2.5, lw=1,
             label='HadCRUT5')



# For diagnosing: filter out results with particular regression coefficients.

# Note to self about masking: coef_Reg_Results is the array of all regression
# coefficients, with shape (4, n), where n is total number of samplings.
# We select slice indices (forcing coefficients) we're interested in basing the
# condition on:
# AER is index 0, GHGs index 1, NAT index 2, Const index 3
# Then choose whether you want any or all or the coefficients to meet the   
# condition (in this case being less than zero)


print(f'shape of coef_Reg_Results: {coef_Reg_Results.shape}')
mask_switch = False
if mask_switch:
    mask = np.all(coef_Reg_Results[[0, 2], :] <= 0, axis=0)
    mask = np.any(coef_Reg_Results[[0], :] <= 0.0, axis=0)

    temp_Att_Results = temp_Att_Results[:, :, mask]
    temp_TOT_Residuals = temp_TOT_Residuals[:, mask]
    temp_TEST_Ens_Results = temp_TEST_Ens_Results[:, mask]
    temp_TEST_Sig_Results = temp_TEST_Sig_Results[:, mask]
    coef_Reg_Results = coef_Reg_Results[:, mask]
    print(f'Shape of masked attribution results: {temp_Att_Results.shape}')


temp_Ant_Results = (temp_Att_Results.sum(axis=1) -
                    temp_Att_Results[:, forc_Group_names.index('Nat'), :] - 
                    int(inc_reg_const) * temp_Att_Results[:, len(forc_Group_names), :]  # Remove constant term in regression
                    )
temp_TOT_Results = temp_Att_Results.sum(axis=1)


for p in sigmas:
    for i in range(len(forc_Group_names)):
        ax1.fill_between(temp_Yrs,
            np.percentile(temp_Att_Results[:, i, :], (p[0]), axis=1),
            np.percentile(temp_Att_Results[:, i, :], (p[1]), axis=1),
            color=forc_Group[forc_Group_names[i]]['Colour'],
            alpha=0.1
            )
        if p == sigmas[-1]:
            ax1.plot(temp_Yrs,
                np.percentile(temp_Att_Results[:, i, :], (50), axis=1),
                color=forc_Group[forc_Group_names[i]]['Colour'],
                label=(forc_Group_names[i]),
                )

    ax1.fill_between(temp_Yrs,
                     np.percentile(temp_TOT_Results[:, :], (p[0]), axis=1),
                     np.percentile(temp_TOT_Results[:, :], (p[1]), axis=1),
                     color='purple',
                     alpha=0.1)
    ax1.fill_between(temp_Yrs,
                     np.percentile(temp_Ant_Results[:, :], (p[0]), axis=1),
                     np.percentile(temp_Ant_Results[:, :], (p[1]), axis=1),
                     color='red',
                     alpha=0.1)

    if p == sigmas[-1]:
        ax1.plot(temp_Yrs, np.percentile(temp_TOT_Results[:, :], (50), axis=1),
                 color='purple', label='TOTAL')
        ax1.plot(temp_Yrs, np.percentile(temp_Ant_Results[:, :], (50), axis=1),
                 color='red', label='Ant')

ax1.set_ylabel('Warming Anomaly (⁰C)')

for p in sigmas:
    ax3.fill_between(temp_Yrs,
        np.percentile(temp_TOT_Residuals[:, :], (p[0]), axis=1),
        np.percentile(temp_TOT_Residuals[:, :], (p[1]), axis=1),
        color='gray',
        alpha=0.1
        )
    if p == sigmas[-1]:
        ax3.plot(temp_Yrs,
            np.percentile(temp_TOT_Residuals[:, :], (50), axis=1),
            color='gray',
            label=(forc_Group_names[i]),
            )
ax3.plot(temp_Yrs, np.zeros(len(temp_Yrs)),
         color='purple', alpha=0.5,
         label='Attributed')
ax3.set_ylabel('Regression Residuals (⁰C)')


for i in range(len(forc_Group_names)):
    binwidth = 0.01
    bins = np.arange(np.min(temp_Att_Results[-1, i, :]),
                     np.max(temp_Att_Results[-1, i, :]) + binwidth,
                     binwidth)
    ax2.hist(temp_Att_Results[-1, i, :], bins=bins,
             density=True, orientation='horizontal',
             color=forc_Group[forc_Group_names[i]]['Colour'],
             alpha=0.3
             )

bins = np.arange(np.min(temp_Ant_Results[-1, :]),
                 np.max(temp_Ant_Results[-1, :]) + binwidth,
                 binwidth)
ax2.hist(temp_Ant_Results[-1, :], bins=bins,
         density=True, orientation='horizontal',
         color='pink', alpha=0.3
         )

bins = np.arange(np.min(temp_TOT_Results[-1, :]),
                 np.max(temp_TOT_Results[-1, :]) + binwidth,
                 binwidth)
ax2.hist(temp_TOT_Results[-1, :], bins=bins,
         density=True, orientation='horizontal',
         color='gray', alpha=0.3
         )


ax2.set_xlabel(f'PDF in {end_yr}')
ax2.set_ylim(ax1.get_ylim())
# ax2.set_xticklabels([])
# ax2.set_yticklabels([])
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax2.text()

gwi = np.around(np.percentile(temp_TOT_Results[-1, :], (50)), decimals=3)
gwi_pls = np.around(np.percentile(temp_TOT_Results[-1, :], (95)) -
                    np.percentile(temp_TOT_Results[-1, :], (50)),
                    decimals=3)
gwi_min = np.around(np.percentile(temp_TOT_Results[-1, :], (50)) -
                    np.percentile(temp_TOT_Results[-1, :], (5)),
                    decimals=3)
# tmp = np.around(np.percentile(df_temp_Obs[-1, :], (50)), decimals=2)
# tmp_pls = np.around(np.percentile(np.array(df_temp_Obs)[-1, :], (95)) -
#                     np.percentile(np.array(df_temp_Obs)[-1, :], (50)),
#                     decimals=2)
# tmp_min = np.around(np.percentile(np.array(df_temp_Obs)[-1, :], (50)) -
#                     np.percentile(np.array(df_temp_Obs)[-1, :], (5)),
#                     decimals=2)
str_GWI = r'${%s}^{+{%s}}_{-{%s}}$' % (gwi, gwi_pls, gwi_min)
# str_temp_Obs = r'${%s}^{+{%s}}_{-{%s}}$' % (tmp, tmp_pls, tmp_min)
fig.text(s=(f'Warming in {end_yr}: ' +
            f'human-induced-warming = {str_GWI} (⁰C)'),
            # f'observed warming = {str_temp_Obs}'),
         y=0.9, x=0.5, horizontalalignment='center')

fig.suptitle(f'Global Warming Index ({n} samplings)')
ax1.legend()
plt.savefig('2_GWI.png')
# plt.savefig(f'GWI PI_Offset-{inc_pi_offset} Reg_Const-{inc_reg_const}.png')
plt.close()



unique_IV_names = sorted(list(set(IV_names)))
cm = 'Set3'
cols = np.array(sns.color_palette(cm, len(unique_IV_names)))
cols_hex = [matplotlib.colors.rgb2hex(cols[i, :])
            for i in range(cols.shape[0])]

IV_col_dic = dict(zip(unique_IV_names, cols_hex))
use_colours = [IV_col_dic[_IV] for _IV in IV_names]
plt.scatter(coef_Reg_Results[0, :], coef_Reg_Results[1, :], color=use_colours,
            alpha=0.02, edgecolors='none', s=20)
plt.xlabel('AER')
plt.ylabel('GHG')
# plt.ylim(bottom=0)
plt.title(f'Coefficients from {n} Samplings')
plt.savefig('3_Coefficients.png')
plt.close()
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
# Now calculate the historical-only GWI #######################################
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
        params = a_params('Carbon Dioxide')
        temp_All = FTmod(forc_All.shape[0], params) @ forc_All

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