import pandas as pd


def load_observation_assessment():
    # Add observations quoted from AR6
    df_AR6_Obs = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        ('Obs', '50'): 1.06,  # AR6 3.3.1.1.2 p442 from observations
        ('Obs',  '5'): 0.88,  # AR6 3.3.1.1.2 p442 from observations
        ('Obs', '95'): 1.21,  # AR6 3.3.1.1.2 p442 from observations
    }, index=['2010\N{EN DASH}2019'])

    df_AR6_Obs.columns.names = ['variable', 'percentile']
    df_AR6_Obs.index.name = 'Year'

    return df_AR6_Obs
