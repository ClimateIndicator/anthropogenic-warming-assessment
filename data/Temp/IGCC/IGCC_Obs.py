import pandas as pd


def load_observation_assessment(assess_yr):
    # Add updated observation results from the annual updates paper section 4
    df_update_Obs_repeat = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        # 2010-2019 (2025 analysis): 1.055 [0.89-1.22] From Blair, paper Sect. 7
        # 2010-2019 (2024 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 7
        # 2010-2019 (2023 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 6
        # 2010-2019 (2022 analysis): 1.07 [0.89-1.22] From Blair, paper Sect. 4
        ('Obs', '50'): 1.055,
        ('Obs',  '5'): 0.89,  # TODO Blair checking whether this needs updating
        ('Obs', '95'): 1.22   # TODO Blair checking whether this needs updating
    }, index=['2010\N{EN DASH}2019'])
    df_update_Obs_repeat.columns.names = ['variable', 'percentile']
    df_update_Obs_repeat.index.name = 'Year'

    assess_range = '2016\N{EN DASH}2025'

    df_update_Obs_update = pd.DataFrame({
        # (VARIABLE, PERCENTILE): VALUE
        # 2016-2025 (2025 analysis): 1.26 [1.13-1.36] From Blair, paper Sect. 7
        # 2015-2024 (2024 analysis): 1.24 [1.11-1.35] From Blair, paper Sect. 7
        # 2014-2023 (2023 analysis): 1.19 [1.06-1.30] From Blair, paper Sect. 6
        # 2013-2022 (2022 analysis): 1.14 [1.00-1.25] From Blair, paper Sect. 4
        ('Obs', '50'): 1.26,
        ('Obs',  '5'): 1.13,
        ('Obs', '95'): 1.36,
    }, index=[assess_range])

    if assess_range != f'{assess_yr-9}\N{EN DASH}{assess_yr}':
        raise ValueError(
            f'Observation assessment period is {assess_range} '
            f'but expected {assess_yr-9}-{assess_yr}')

    df_update_Obs_update.columns.names = ['variable', 'percentile']
    df_update_Obs_update.index.name = 'Year'

    df_IGCC_Obs = pd.concat([
                        df_update_Obs_repeat,
                        df_update_Obs_update
                        ])

    return df_IGCC_Obs
