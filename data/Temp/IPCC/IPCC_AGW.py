import pandas as pd
import src.definitions as defs


def load_AR6_assessment():
    # QUOTED HEADLINE RESULTS FROM IPCC 6TH ASSESSMENT CYCLE ##################
    # Create dataframe of results from AR6 WG1 Ch.3
    df_AR6_assessment = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.07,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
        ('Ant',  '5'): 0.80,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
        ('Ant', '95'): 1.30,  # AR6 3.3.1.1.2 p442, and SPM A.1.3
        ('GHG', '50'): 1.40,  # AR6 We introduce multi-method assessment here;
        # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of
        # likely range to give 1.5
        ('GHG',  '5'): 1.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('GHG', '95'): 2.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('Nat', '50'): 0.03,  # We introduce multi-method assessment here;
        # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of
        # likely range to give 0.0
        ('Nat',  '5'): -0.10,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('Nat', '95'): 0.10,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('OHF', '50'): -0.32,  # We introduce multi-method assessment here;
        # 3.3.1.1.2 had no value for this, and SPM2 just plotted midpoint of
        # likely range to give -0.4
        ('OHF',  '5'): -0.80,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('OHF', '95'): 0.00,  # AR6 3.3.1.1.2 p442, SPM A.1.3
        ('Int', '50'): 0.00,  # AR6 3.3.1.1.2 had no value for this, and SPM2
        # just plotted midpoint of likely range to give 0.0, which is still
        # used here.
        ('Int',  '5'): -0.20,  # AR6 3.3.1.1.2 p443, SPM A.1.3
        ('Int', '95'): 0.20,  # AR6 3.3.1.1.2 p443, SPM A.1.3
    }, index=['2010\N{EN DASH}2019'])
    df_AR6_assessment.columns.names = ['variable', 'percentile']
    df_AR6_assessment.index.name = 'Year'

    return df_AR6_assessment


def load_SR15_assessment():
    # Create dataframe of results from SR15 Ch.1
    df_SR15_assessment = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.0,  # SR15 1.2.1.3
        ('Ant',  '5'): 0.8,  # SR15 1.2.1.3
        ('Ant', '95'): 1.2,  # SR15 1.2.1.3
    }, index=['2017'])
    df_SR15_assessment.columns.names = ['variable', 'percentile']
    df_SR15_assessment.index.name = 'Year'

    return df_SR15_assessment


def load_IPCC_Haustein():

    # Haustein 2017 (GWI)
    df_AR6_Haustein = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.064,
        ('Ant',  '5'): 0.941,
        ('Ant', '95'): 1.222,
        ('GHG', '50'): 1.259,
        ('GHG',  '5'): 1.259,
        ('GHG', '95'): 1.259,
        ('Nat', '50'): 0.026,
        ('Nat',  '5'): 0.001,
        ('Nat', '95'): 0.069,
        ('OHF', '50'): -0.195,
        ('OHF',  '5'): -0.195,
        ('OHF', '95'): -0.195,
    }, index=['2010\N{EN DASH}2019'])
    df_AR6_Haustein.columns.names = ['variable', 'percentile']
    df_AR6_Haustein.index.name = 'Year'

    df_SR15_Haustein = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.02,  # SR15 1.2.1.3
        ('Ant',  '5'): 0.87,  # SR15 1.2.1.3
        ('Ant', '95'): 1.22,  # SR15 1.2.1.3
    }, index=['2017'])
    df_SR15_Haustein.columns.names = ['variable', 'percentile']
    df_SR15_Haustein.index.name = 'Year'

    df_IPCC_Haustein = pd.concat([df_AR6_Haustein, df_SR15_Haustein])

    return df_IPCC_Haustein


def load_IPCC_Ribes():
    # Ribes (KCC)
    df_AR6_Ribes = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.03,
        ('Ant',  '5'): 0.89,
        ('Ant', '95'): 1.17,
        ('GHG', '50'): 1.44,
        ('GHG',  '5'): 1.12,
        ('GHG', '95'): 1.76,
        ('Nat', '50'): 0.06,
        ('Nat',  '5'): 0.04,
        ('Nat', '95'): 0.08,
        ('OHF', '50'): -0.40,
        ('OHF',  '5'): -0.69,
        ('OHF', '95'): -0.12,
        ('Int', '50'): -0.02,
        ('Int',  '5'): -0.18,
        ('Int', '95'): 0.14,
    }, index=['2010\N{EN DASH}2019'])
    df_AR6_Ribes.columns.names = ['variable', 'percentile']
    df_AR6_Ribes.index.name = 'Year'
    df_IPCC_Ribes = df_AR6_Ribes

    return df_IPCC_Ribes


def load_IPCC_Gillett():

    # Gillet (ROF)
    df_AR6_Gillett = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.11,
        ('Ant',  '5'): 0.92,
        ('Ant', '95'): 1.30,
        ('GHG', '50'): 1.50,
        ('GHG',  '5'): 1.06,
        ('GHG', '95'): 1.94,
        ('Nat', '50'): 0.01,
        ('Nat',  '5'): -0.02,
        ('Nat', '95'): 0.05,
        ('OHF', '50'): -0.37,
        ('OHF',  '5'): -0.71,
        ('OHF', '95'): -0.03,
    }, index=['2010\N{EN DASH}2019'])
    df_AR6_Gillett.columns.names = ['variable', 'percentile']
    df_AR6_Gillett.index.name = 'Year'
    df_IPCC_Gillett = df_AR6_Gillett

    return df_IPCC_Gillett


def load_IPCC_Smith():
    # Smith (AR6 WGI Chapter 7)
    df_AR6_Smith = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Ant', '50'): 1.066304612,
        ('Ant',  '5'): 0.823021383,
        ('Ant', '95'): 1.353390492,
        ('GHG', '50'): 1.341781251,
        ('GHG',  '5'): 0.993864648,
        ('GHG', '95'): 1.836139027,
        ('Nat', '50'): 0.073580353,
        ('Nat',  '5'): 0.04283156,
        ('Nat', '95'): 0.119195551,
        ('OHF', '50'): -0.269287921,
        ('OHF',  '5'): -0.628487091,
        ('OHF', '95'): -0.026618862,
    }, index=['2010\N{EN DASH}2019'])
    df_AR6_Smith.columns.names = ['variable', 'percentile']
    df_AR6_Smith.index.name = 'Year'
    df_IPCC_Smith = df_AR6_Smith

    return df_IPCC_Smith


def load_IPCC_df():
    # Combine 6th assessment cycle results from both SR1.5 and AR6
    df_IPCC_assessment = pd.concat(
        [load_AR6_assessment(), load_SR15_assessment()]
        )

    unendashed_assessment = defs.un_en_dash_ify(df_IPCC_assessment.copy())
    unendashed_assessment.to_csv('results/Assessment-6thIPCC_headlines.csv')

    # QUOTED RESULTS FROM INDIVIDUAL AR6 ATTRIBUTION METHODS ##################
    # These results for each method are quoted here from the AR6 assessment.
    # Data available from https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3_nathan/esmvaltool/diag_scripts/ipcc_ar6/fig3_8.py
    df_IPCC_Haustein = load_IPCC_Haustein()
    df_IPCC_Ribes = load_IPCC_Ribes()
    df_IPCC_Gillett = load_IPCC_Gillett()
    df_IPCC_Smith = load_IPCC_Smith()

    # Combine all IPCC-quoted results (not the updated results) into one
    # dictionary
    dict_IPCC_hl = {
        'Assessment': df_IPCC_assessment,
        'Haustein': df_IPCC_Haustein,
        'Ribes': df_IPCC_Ribes,
        'Gillett': df_IPCC_Gillett,
        'Smith': df_IPCC_Smith,
    }

    return dict_IPCC_hl
