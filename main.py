import numpy as np
import pandas as pd
import tensorflow as tf
import input_data_preparation as data_prep
import binary_linear_regression as blr
import linear_regression as lr
import plotting as plot
from tensorflow import feature_column
from tensorflow.keras import layers

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_df, train_df_norm, test_df, test_df_norm = data_prep.prepare_data()

    feature_columns = []

    dayOfYear = tf.feature_column.numeric_column("day_of_year")
    feature_columns.append(dayOfYear)

    boundaries_day_of_year = list(np.arange(int(min(train_df['day_of_year'])), int(max(train_df['day_of_year'])), 1))
    bucketized_day_of_year = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("day_of_year"), boundaries_day_of_year)
    boundaries_hour = list(np.arange(int(min(train_df['hour'])), int(max(train_df['hour'])), 1))
    bucketized_hour = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("hour"), boundaries_hour)
    latitude_x_longitude = tf.feature_column.crossed_column([bucketized_day_of_year, bucketized_hour], hash_bucket_size=2200)
    crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
    feature_columns.append(crossed_feature)

    wind = tf.feature_column.numeric_column("wind_speed")
    feature_columns.append(wind)

    humidity = tf.feature_column.numeric_column("humidity")
    feature_columns.append(humidity)

    weather = tf.feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("weather_code", train_df["weather_code"].unique()))
    feature_columns.append(weather)

    temperature = tf.feature_column.numeric_column("temperature")
    feature_columns.append(temperature)

    feature_layer = layers.DenseFeatures(feature_columns)

    # The following variables are the hyperparameters.
    learning_rate = 0.01
    epochs = 100
    batch_size = 80
    label_name = 'pm10_norm'
    # label_name = 'IsPM10Exceeded'
    classification_threshold = 0.4

    metrics = [tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision()]

    # Establish the model's topography.
    my_model = lr.create_model(learning_rate, feature_layer)

    # Train the model on the training set.
    epochs, hist = lr.train_model(my_model, train_df, epochs, label_name, batch_size)

    test_features = {name: np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop(label_name))  # isolate the label
    my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
    predictions = my_model.predict(test_features)

    for i in range(0, 10):
        print('{} {}'.format(predictions[i], test_label[i]))

    out = test_df
    out['prediction'] = predictions

    out.to_csv('C:\\Users\\sliwk\\Downloads\\2019 Dane\\2019_3_out.csv')
    # Plot a graph of the metric(s) vs. epochs.
    # list_of_metrics_to_plot = ['loss', 'root_mean_squared_error']
    # plot.plot_curve(epochs, hist, list_of_metrics_to_plot)
    # list_of_metrics_to_plot = ['precision', 'recall']
    # plot.plot_curve(epochs, hist, list_of_metrics_to_plot)

    print('finished')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
