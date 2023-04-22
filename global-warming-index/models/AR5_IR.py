"""Script defining IPCC AR5 Impulse Response Model."""

import numpy as np

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
