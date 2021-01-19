import numpy as np
import tensorflow as tf
import input_data_preparation as data_prep
import matplotlib.pyplot as plt
from pandas import DataFrame
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(32),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(128),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def run_nn(run_params):
    train_df, train_df_norm, test_df, test_df_norm = data_prep.prepare_data(run_params.training_years,
                                                                            run_params.testing_years,
                                                                            run_params.hour_resolution, True)

    normalizer = preprocessing.Normalization()
    train_labels = train_df.pop('pm10')
    train_df.pop('pm25')
    test_labels = test_df.pop('pm10')
    test_df.pop('pm25')

    train_df = np.asarray(train_df)
    test_df = np.asarray(test_df)

    normalizer.adapt(np.array(train_df))

    model = build_and_compile_model(normalizer)
    history = model.fit(
        train_df, train_labels,
        validation_split=0.2,
        verbose=0, epochs=300)

    plot_loss(history)

    results = model.evaluate(
        test_df, test_labels
    )

    test_predictions = DataFrame()
    test_predictions['pm10'] = test_labels
    test_predictions['pm10_prediction'] = model.predict(test_df)

    test_predictions.tail()

    return test_predictions


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [pm10]')
    plt.legend()
    plt.grid(True)
