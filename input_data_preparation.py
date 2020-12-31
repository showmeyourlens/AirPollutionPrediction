import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import feature_column


def add_past_pollution(dataSet, hour):
    pm10_last_column = 'pm10_last'
    pm25_last_column = 'pm25_last'
    pm10_last2_column = 'pm10_last2'
    pm25_last2_column = 'pm25_last2'
    pm10_column = 'pm10'
    pm25_column = 'pm25'
    index_column = 'index'
    current_index = 0
    iterator = 0
    dataSet[pm10_last_column] = np.nan
    dataSet[pm25_last_column] = np.nan
    dataSet[pm10_last2_column] = np.nan
    dataSet[pm25_last2_column] = np.nan
    try:
        for i in range(0, len(dataSet)):
            iterator = i
            if i > 1:
                current_index = dataSet.loc[i, index_column]
                last_index = dataSet.loc[i-1, index_column]
                last2_index = dataSet.loc[i-2, index_column]

                if current_index - int(hour) == last_index:
                    dataSet.loc[i, pm10_last_column] = dataSet.loc[i-1, pm10_column]
                    dataSet.loc[i, pm25_last_column] = dataSet.loc[i-1, pm25_column]

                if current_index - (int(hour) * 2) == last2_index:
                    dataSet.loc[i, pm10_last2_column] = dataSet.loc[i - 2, pm10_column]
                    dataSet.loc[i, pm25_last2_column] = dataSet.loc[i - 2, pm25_column]

    except KeyError:
        print("iteration {} current index {}".format(iterator, current_index))

    result = dataSet[dataSet[pm10_last_column].notna()]
    result = result[dataSet[pm10_last2_column].notna()]

    return result


def prepare_data(training_years, testing_years, hour):
    pd.set_option('display.max_columns', None)

    train_dfs = []

    for year in training_years:
        temp_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\' + hour + 'h.csv')
        print("Adding past pollution to {} {}h dataset".format(year, hour))
        temp_df = add_past_pollution(temp_df, hour)
        train_dfs.append(temp_df)
        print("Added dataset to training DF")

    train_df = pd.concat(train_dfs, ignore_index=1)
    train_df = train_df.reindex(np.random.permutation(train_df.index))

    test_dfs = []

    for year in testing_years:
        temp_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\' + hour + 'h.csv')
        print("Adding past pollution to {} {}h dataset".format(year, hour))
        temp_df = add_past_pollution(temp_df, hour)
        test_dfs.append(temp_df)
        print("Added dataset to testing DF")

    test_df = pd.concat(test_dfs, ignore_index=1)
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


def add_feature_columns(train_df):
    feature_columns = []

    dayOfYear = tf.feature_column.numeric_column("day_of_year")
    feature_columns.append(dayOfYear)

    # boundaries_day_of_year = list(np.arange(int(min(train_df['day_of_year'])), int(max(train_df['day_of_year'])), 1))
    # bucketized_day_of_year = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("day_of_year"), boundaries_day_of_year)
    # boundaries_hour = list(np.arange(int(min(train_df['hour'])), int(max(train_df['hour'])), 1))
    # bucketized_hour = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("hour"), boundaries_hour)
    # latitude_x_longitude = tf.feature_column.crossed_column([bucketized_day_of_year, bucketized_hour], hash_bucket_size=2200)
    # crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
    # feature_columns.append(crossed_feature)

    hour = tf.feature_column.numeric_column("hour")
    feature_columns.append(hour)

    wind = tf.feature_column.numeric_column("wind_speed")
    feature_columns.append(wind)

    humidity = tf.feature_column.numeric_column("humidity")
    feature_columns.append(humidity)

    weather_current = tf.feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("weather_code", train_df["weather_code"].unique()))
    feature_columns.append(weather_current)

    weather_past = tf.feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("past_weather_code", train_df["past_weather_code"].unique()))
    feature_columns.append(weather_past)

    temperature = tf.feature_column.numeric_column("temperature")
    feature_columns.append(temperature)

    pm10_last = tf.feature_column.numeric_column("pm10_last")
    feature_columns.append(pm10_last)

    pm10_last = tf.feature_column.numeric_column("pm25_last")
    feature_columns.append(pm10_last)

    pm10_last2 = tf.feature_column.numeric_column("pm10_last2")
    feature_columns.append(pm10_last2)

    pm25_last2 = tf.feature_column.numeric_column("pm25_last2")
    feature_columns.append(pm25_last2)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    return feature_columns
