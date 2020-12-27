import pandas as pd
import numpy as np


def smooth_pm_columns(dataSet):
    dataSet['pm10_norm'] = 1
    for i in range(0, len(dataSet)):
        if i > 1:
            dataSet.loc[i, 'pm10_norm'] = dataSet.loc[i, 'pm10'] * 0.4 + dataSet.loc[i - 1, 'pm10'] * 0.4 + dataSet.loc[i - 1, 'pm10_norm'] * 0.2
        else:
            dataSet.loc[i, 'pm10_norm'] = dataSet.loc[i, 'pm10']
    return dataSet

def prepare_data():
    pd.set_option('display.max_columns', None)
    train_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\2018 Dane\\2018_3.csv')
    train_df = smooth_pm_columns(train_df)
    train_df_part_2 = pd.read_csv('C:\\Users\\sliwk\\Downloads\\2017 Dane\\2017_3.csv')
    train_df_part_2 = smooth_pm_columns(train_df_part_2)
    train_df = pd.concat([train_df, train_df_part_2], ignore_index=1)
    train_df = train_df.reindex(np.random.permutation(train_df.index))


    test_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\2019 Dane\\2019_3.csv')
    test_df = smooth_pm_columns(test_df)
    test_df = test_df.reindex(np.random.permutation(test_df.index))

    print('Loaded csv')

    # Calculate the Z-scores of each column in the training set:
    train_df_mean = train_df.mean()
    train_df_std = train_df.std()
    train_df_norm = (train_df - train_df_mean) / train_df_std

    # Calculate the Z-scores of each column in the test set.
    test_df_mean = test_df.mean()
    test_df_std = test_df.std()
    test_df_norm = (test_df - test_df_mean) / test_df_std


    print("Normalized the values.")

    return train_df, train_df_norm, test_df, test_df_norm
