import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder

import input_data_preparation as data_prep
from tensorflow.keras import layers


# @title Define functions to create and train a model, and a plotting function
def create_model(my_learning_rate, feature_layer):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.

    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, dataset, epochs, label_name, batch_size):
    """Feed a dataset into the model in order to train it."""

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True, verbose=1)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse


def get_trained_model(train_df, test_df, feature_layer, label_name, run_params):
    # The following variables are the hyperparameters.
    # label_name = 'IsPM10Exceeded'
    classification_threshold = 0.4

    metrics = [tf.keras.metrics.RootMeanSquaredError(),
               tf.keras.metrics.Recall(),
               tf.keras.metrics.Precision()]

    # Establish the model's topography.
    my_model = create_model(run_params.learning_rate, feature_layer)

    # Train the model on the training set.
    epochs, hist = train_model(my_model, train_df, run_params.epochs, label_name, run_params.batch_size)

    print("Model trained")
    test_features = {name: np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop(label_name))  # isolate the label

    print("Evaluating:")
    my_model.evaluate(x=test_features, y=test_label, batch_size=run_params.batch_size, verbose=1)

    return my_model


def run_algorithm(run_params):

    # train_df, train_df_norm, test_df, test_df_norm = data_prep.prepare_data(run_params.training_years, run_params.testing_years, run_params.hour_resolution, True)

    data_df, train_mean, train_std = data_prep.prepare_linear_timeseries_data(run_params)

    n = len(data_df)
    train_df = data_df[int(n * 0.05):int(n * 0.8)]
    test_df = data_df[int(n * 0.8):]

    feature_columns = data_prep.add_feature_columns(train_df)

    feature_layer = layers.DenseFeatures(feature_columns)

    out = DataFrame()

    train_df.pop('datetime')
    out['datetime'] = test_df.pop('datetime')
    out[run_params.predicted_label] = test_df[run_params.predicted_label] * train_std + train_mean
    # out[run_params.predicted_label] = test_df[run_params.predicted_label]
    my_model = get_trained_model(train_df, test_df, feature_layer, run_params.predicted_label, run_params)
    print(my_model.trainable_variables)
    print("Train mean: {}".format(train_mean))
    print("Train std: {}".format(train_std))
    test_df.sort_values("index", inplace=True)
    test_features = {name: np.array(value) for name, value in test_df.items()}
    out[run_params.predicted_label + '_prediction'] = my_model.predict(test_features) * train_std + train_mean
    # out[run_params.predicted_label + '_prediction'] = my_model.predict(test_features)

    return out


def do_research(run_params):

    output_file_path = 'C:\\Users\\sliwk\\Downloads\\Dane\\Output\\' + run_params.hour_resolution + 'h out.csv'
    file = open(output_file_path, "a")
    file.close()

    train_df, train_df_norm, test_df, test_df_norm = data_prep.prepare_data(run_params.training_years, run_params.testing_years, run_params.hour_resolution, True)

    feature_columns = data_prep.add_feature_columns(train_df)

    feature_layer = layers.DenseFeatures(feature_columns)

    my_model = get_trained_model(train_df, test_df, feature_layer, run_params.predicted_label, run_params)

    research_df = data_prep.prepare_research_data(test_df)

    counter = 0

    for record in research_df.index:
        if counter < 2:
            counter = counter + 1
            continue
        data_prep.add_past_one_record(counter, research_df, run_params.hour_resolution)
        lol = research_df.loc[counter:counter]
        test_features = {name: np.array(value) for name, value in lol.items()}
        prediction = my_model.predict(test_features)
        value = prediction.item(0)
        research_df.loc[counter, run_params.predicted_label] = value
        counter = counter + 1

    research_df.sort_values("index", inplace=True)

    return research_df

