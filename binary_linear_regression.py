import tensorflow as tf
import numpy as np
import pandas as pd


def create_model(my_learning_rate, feature_layer, metrics):
    """Create and compile a simple classification model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the feature layer (the list of features and how they are represented)
    # to the model.
    model.add(feature_layer)

    # Funnel the regression value through a sigmoid function.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,),
                                    activation=tf.sigmoid), )

    # Call the compile method to construct the layers into a model that
    # TensorFlow can execute.  Notice that we're using a different loss
    # function for classification than for regression.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    return model


def train_model(model, dataset, epochs, label_name,
                batch_size=None, shuffle=True):
    """Feed a dataset into the model in order to train it."""

    # The x parameter of tf.keras.Model.fit can be a list of arrays, where
    # each array contains the data for one feature.  Here, we're passing
    # every column in the dataset. Note that the feature_layer will filter
    # away most of those columns, leaving only the desired columns and their
    # representations as features.
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=shuffle)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the classification metric for each epoch.
    hist = pd.DataFrame(history.history)

    return epochs, hist
