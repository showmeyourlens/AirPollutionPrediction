import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame

import input_data_preparation as data_prep
from tensorflow.keras import layers


# @title Define functions to create and train a model, and a plotting function
def create_model(my_learning_rate, feature_layer):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Add one linear layer to the model to yield a simple linear regressor.
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
                        epochs=epochs, shuffle=True, verbose=0)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse


def get_predictions(train_df, test_df, feature_layer, label_name, run_params):
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

    result = my_model.predict(test_features)

    return result


def run_algorithm(run_params):
    output_file_path = 'C:\\Users\\sliwk\\Downloads\\Dane\\Output\\' + run_params.hour_resolution + 'h out.csv'
    file = open(output_file_path, "a")
    file.close()

    train_df, train_df_norm, test_df, test_df_norm = data_prep.prepare_data(run_params.training_years, run_params.testing_years, run_params.hour_resolution)

    feature_columns = data_prep.add_feature_columns(train_df)

    feature_layer = layers.DenseFeatures(feature_columns)

    out = DataFrame()

    for predicted_label in run_params.predicted_labels:
        out[predicted_label] = test_df[predicted_label]
        linear_regression_pm10_predictions = get_predictions(train_df, test_df, feature_layer, predicted_label, run_params)
        out[predicted_label + '_prediction'] = linear_regression_pm10_predictions

    out.sort_index(inplace=True)
    return out

