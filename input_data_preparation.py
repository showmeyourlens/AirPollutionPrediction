import pandas as pd
import numpy as np
import tensorflow as tf
import os
import copy
import math
from sklearn.preprocessing import OneHotEncoder
from tensorflow import feature_column
import datetime


pm10_last_column = 'pm10_last'
pm25_last_column = 'pm25_last'
pm10_last2_column = 'pm10_last2'
pm25_last2_column = 'pm25_last2'
pm10_column = 'pm10'
pm25_column = 'pm25'
index_column = 'index'
pm10_copy = 'pm10_copy'
pm25_copy = 'pm25_copy'
data_start_row = 10
data_periods = 12


def add_past_pollution(dataSet, hour):
    current_index = 0
    iterator = 0
    dataSet[pm10_last_column] = np.nan
    dataSet[pm25_last_column] = np.nan
    dataSet[pm10_last2_column] = np.nan
    dataSet[pm25_last2_column] = np.nan
    # try:
    #     for i in range(0, len(dataSet)):
    #         iterator = i
    #         if i > 1:
    #             add_past_one_record(i, dataSet, hour)
    #
    # except KeyError:
    #     print("iteration {} current index {}".format(iterator, current_index))

    result = dataSet[dataSet[pm10_last_column].notna()]
    result = result[dataSet[pm10_last2_column].notna()]

    return result


def add_past_one_record(record_pos, dataSet, hour):
    current_index = dataSet.loc[record_pos, index_column]
    last_index = dataSet.loc[record_pos - 1, index_column]
    last2_index = dataSet.loc[record_pos - 2, index_column]

    # if current_index - int(hour) == last_index:
    #     dataSet.loc[record_pos, pm10_last_column] = dataSet.loc[record_pos - 1, pm10_column]
    #     dataSet.loc[record_pos, pm25_last_column] = dataSet.loc[record_pos - 1, pm25_column]
    #
    # if current_index - (int(hour) * 2) == last2_index:
    #     dataSet.loc[record_pos, pm10_last2_column] = dataSet.loc[record_pos - 1, pm10_column] - dataSet.loc[record_pos - 2, pm10_column]
    #     dataSet.loc[record_pos, pm25_last2_column] = dataSet.loc[record_pos - 1, pm25_column] - dataSet.loc[record_pos - 2, pm25_column]

    pass


def prepare_research_data(test_df):
    research_df = copy.deepcopy(test_df)
    research_df.sort_values("index", inplace=True)
    counter = 0
    current_row = data_start_row
    research_df[pm10_copy] = research_df[pm10_column]
    research_df[pm25_copy] = research_df[pm25_column]
    # while counter < data_periods:
    #     if counter < 2:
    #         counter = counter + 1
    #         current_row = current_row + 1
    #         continue
    #     research_df.loc[current_row, pm10_column] = 0.0
    #     research_df.loc[current_row, pm25_column] = 0.0
    #     research_df.loc[current_row, pm10_last_column] = 0.0
    #     research_df.loc[current_row, pm25_last_column] = 0.0
    #     research_df.loc[current_row, pm10_last2_column] = 0.0
    #     research_df.loc[current_row, pm25_last2_column] = 0.0
    #
    #     counter = counter + 1
    #     current_row = current_row + 1

    research_df.drop(research_df.index[:data_start_row], inplace=True)
    research_df.drop(research_df.index[data_periods:], inplace=True)
    research_df.reset_index(inplace=True, drop=True)
    return research_df


def create_prepared_files(years, hour):
    for year in years:
        output_file_path = 'C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\' + hour + 'h proc.csv'
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        temp_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\' + hour + 'h.csv')
        print("Adding new columns to {} {}h dataset".format(year, hour))
        temp_df = add_past_pollution(temp_df, hour)
        print("Adding columns finished")
        temp_df.to_csv(output_file_path, sep=",")


def is_increased_traffic(x):
    week_day = x.weekday()
    hour = x.hour
    if 0 <= week_day <= 3:
        if (7 <= hour <= 9) | (15 <= hour <= 17):
            return 1
    if week_day == 4:
        if (7 <= hour <= 9) | (15 <= hour <= 18):
            return 2
    return 0


def prepare_linear_timeseries_data(run_params):
    output = []
    hour = int(run_params.hour_resolution)
    for year in run_params.training_years:
        raw_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\timeseries.csv')
        raw_df['pm10'] = raw_df['pm10'].replace(0.0, np.nan)
        raw_df['pm25'] = raw_df['pm25'].replace(0.0, np.nan)
        raw_df['pm10'] = np.clip(raw_df['pm10'].values, 0, 100)
        raw_df['pm25'] = np.clip(raw_df['pm10'].values, 0, 80)

        reshaped_raw_data = raw_df.values.reshape(-1, hour, raw_df.shape[1])
        result = pd.DataFrame(reshaped_raw_data.mean(1))
        result.columns = raw_df.columns
        data_every_x_hour = raw_df[::hour].reset_index(drop=True)
        result['year'] = data_every_x_hour['year']
        result['month'] = data_every_x_hour['month']
        result['day'] = data_every_x_hour['day']
        result['hour'] = data_every_x_hour['hour']
        result['index'] = data_every_x_hour['index']

        result = result[result['pm10'].notna()]
        result = result[result['pm25'].notna()]
        result['datetime'] = pd.to_datetime(result[['year', 'month', 'day', 'hour']])
        result['increased_traffic'] = result['datetime'].apply(lambda x: is_increased_traffic(x)).values
        output.append(result)
        print("Added dataset to training DF")

    train_df = pd.concat(output, ignore_index=1)

    dt_pop = train_df.pop('datetime')

    train_df['day_of_year'] = np.abs(train_df['day'].values - 197) % 183
    train_df['hour_of_day'] = np.abs(train_df['hour'].values - 12)
    train_df['wind_direction'] = np.sin(train_df['wind_direction'].values * np.pi / 180)

    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std

    # output_file_path = 'C:\\Users\\sliwk\\Downloads\\Dane\\Output\\correlation.csv'
    # train_df.to_csv(output_file_path, sep=';')

    train_df['datetime'] = dt_pop
    # weather_code = train_df['weather_code']
    # past_weather_code = train_df['past_weather_code']
    #
    # x = OneHotEncoder(dtype=int).fit_transform(weather_code.values.reshape(-1, 1)).toarray()
    # y = OneHotEncoder(dtype=int).fit_transform(past_weather_code.values.reshape(-1, 1)).toarray()
    #
    train_df['weather_code'] = np.ndarray.astype(np.asarray(train_df['weather_code'] * train_std['weather_code'] + train_mean['weather_code']), dtype=int)
    train_df['past_weather_code'] = np.ndarray.astype(np.asarray(train_df['past_weather_code'] * train_std['past_weather_code'] + train_mean['past_weather_code']), dtype=int)

    train_df['weather_code'] = np.ndarray.astype(np.asarray(train_df['weather_code']), dtype=int)
    train_df['past_weather_code'] = np.ndarray.astype(np.asarray(train_df['past_weather_code']), dtype=int)

    train_df = train_df.reindex(np.random.permutation(train_df.index))

    return train_df, train_mean[run_params.predicted_label], train_std[run_params.predicted_label]


def prepare_timeseries_data(years, hour):
    output = []
    hour = int(hour)
    for year in years:
        raw_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\timeseries.csv')
        raw_df = raw_df.drop('index', axis=1)
        raw_df['pm10'] = raw_df['pm10'].replace(0.0, np.nan)
        raw_df['pm25'] = raw_df['pm25'].replace(0.0, np.nan)
        raw_df['pm10'] = np.clip(raw_df['pm10'].values, 0, 100)
        raw_df['pm25'] = np.clip(raw_df['pm10'].values, 0, 80)

        reshaped_raw_data = raw_df.values.reshape(-1, hour, raw_df.shape[1])
        result = pd.DataFrame(reshaped_raw_data.mean(1))
        result.columns = raw_df.columns
        data_every_x_hour = raw_df[::hour].reset_index(drop=True)
        result['year'] = data_every_x_hour['year']
        result['month'] = data_every_x_hour['month']
        result['day'] = data_every_x_hour['day']
        result['hour'] = data_every_x_hour['hour']

        result = result[result['pm10'].notna()]
        result = result[result['pm25'].notna()]
        result['datetime'] = pd.to_datetime(result[['year', 'month', 'day', 'hour']])
        result['increased_traffic'] = result['datetime'].apply(lambda x: is_increased_traffic(x))
        output.append(result)
        print("Added dataset to training DF")

    train_df = pd.concat(output, ignore_index=1)

    day = 24 * 60 * 60
    year = (365.2425) * day

    date_time = pd.to_datetime(train_df.pop('datetime'))
    timestamp_s = date_time.map(datetime.datetime.timestamp)

    train_df['Day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    train_df['Day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    train_df['Year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    train_df['Year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return train_df


def prepare_data(training_years, testing_years, hour, reindex):
    pd.set_option('display.max_columns', None)

    train_dfs = []

    for year in training_years:
        temp_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\' + hour + 'h proc.csv')
        train_dfs.append(temp_df)
        print("Added dataset to training DF")

    train_df = pd.concat(train_dfs, ignore_index=1)

    if reindex:
        train_df = train_df.reindex(np.random.permutation(train_df.index))

    test_dfs = []

    for year in testing_years:
        temp_df = pd.read_csv('C:\\Users\\sliwk\\Downloads\\Dane\\' + year + ' Dane\\' + hour + 'h proc.csv')
        test_dfs.append(temp_df)
        print("Added dataset to testing DF")

    test_df = pd.concat(test_dfs, ignore_index=1)
    if reindex:
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

    feature_columns.append(tf.feature_column.numeric_column('day_of_year'))
    feature_columns.append(tf.feature_column.numeric_column('hour_of_day'))

    # boundaries_day_of_year = list(np.arange(int(min(train_df['day_of_year'])), int(max(train_df['day_of_year'])), 1))
    # bucketized_day_of_year = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("day_of_year"), boundaries_day_of_year)
    # boundaries_hour = list(np.arange(int(min(train_df['hour'])), int(max(train_df['hour'])), 1))
    # bucketized_hour = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("hour"), boundaries_hour)
    # latitude_x_longitude = tf.feature_column.crossed_column([bucketized_day_of_year, bucketized_hour], hash_bucket_size=2200)
    # crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
    # feature_columns.append(crossed_feature)y

    wind = tf.feature_column.numeric_column("wind_speed")
    feature_columns.append(wind)

    wind = tf.feature_column.numeric_column("wind_direction")
    feature_columns.append(wind)

    humidity = tf.feature_column.numeric_column("humidity")
    feature_columns.append(humidity)

    temperature = tf.feature_column.numeric_column("temperature")
    feature_columns.append(temperature)

    temperature = tf.feature_column.numeric_column("increased_traffic")
    feature_columns.append(temperature)

    print(train_df['weather_code'].unique())
    weather_current = tf.feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("weather_code", train_df['weather_code'].unique()))
    feature_columns.append(weather_current)

    print(train_df['past_weather_code'].unique())
    weather_past = tf.feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("past_weather_code", train_df['past_weather_code'].unique()))
    feature_columns.append(weather_past)

    # pm10_last = tf.feature_column.numeric_column("pm10_last")
    # feature_columns.append(pm10_last)
    #
    # pm10_last = tf.feature_column.numeric_column("pm25_last")
    # feature_columns.append(pm10_last)
    #
    # pm10_last2 = tf.feature_column.numeric_column("pm10_last2")
    # feature_columns.append(pm10_last2)
    #
    # pm25_last2 = tf.feature_column.numeric_column("pm25_last2")
    # feature_columns.append(pm25_last2)

    return feature_columns
