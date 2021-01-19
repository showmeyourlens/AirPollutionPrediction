import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None, name='no_name'):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.batch_size = 70
        self.name = name
        self.overlap = 0
        self.samples = 10
        self.max_epochs = 10

    def __repr__(self):
        return '\n'.join([
            f'',
            f'Name: {self.name}',
            f'Total window size: {self.total_window_size}',
            f'Batches: {self.batch_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Input count: {self.input_indices.size}',
            f'Label count: {self.label_indices.size}',
            f'Shift: {self.shift}',
            f'Overlap: {self.overlap}',
            f'Samples: {self.samples}',
            f'Label column name(s): {self.label_columns}',
            f''
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def get_research_result(self, model, train_std, train_mean, plot_col='pm10'):
        inputs, labels = self.example
        self.overlap = max(0, max(self.input_indices) - min(self.label_indices) + 1)
        prediction_size = labels.shape[1] - self.overlap
        plot_col_index = self.column_indices[plot_col]
        predictions = model(inputs)
        output = DataFrame()
        rmse_coll = []
        rmse_by_hours = []
        for n in range(self.samples):
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            position = n * prediction_size
            input_vals = inputs[position, :, plot_col_index] * train_std + train_mean
            label_vals = labels[position, :, label_col_index] * train_std + train_mean
            pred_vals = predictions[position, :, label_col_index] * train_std + train_mean

            partial_rmse_col = np.sqrt((np.array(label_vals) - np.array(pred_vals)) ** 2)
            if self.overlap != 0:
                partial_rmse_col = partial_rmse_col[self.overlap:]
            rmse_coll = np.append(rmse_coll, partial_rmse_col)

            if self.overlap == 0:
                output['input_{}'.format(n)] = np.concatenate((np.array(input_vals), np.array(label_vals)), axis=0)
            else:
                output['input_{}'.format(n)] = np.concatenate((np.array(input_vals)[:-self.overlap], np.array(label_vals)), axis=0)

            output['prediction_{}'.format(n)] = np.concatenate((np.full(min(self.label_indices), 0.0), np.array(pred_vals)), axis=0)

        prediction_size = labels.shape[1] - self.overlap
        rmse_coll = rmse_coll.reshape((self.samples, prediction_size))
        for i in range(prediction_size):
            rmse_column = rmse_coll[:, i]
            rmse_by_hours = np.append(rmse_by_hours, np.sum(rmse_column) / rmse_column.size)

        return output, self.samples, rmse_by_hours

    def plot(self, model=None, plot_col='pm10', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                input_vals = inputs[n, :, plot_col_index]
                label_vals = labels[n, :, label_col_index]
                pred_vals = predictions[n, :, label_col_index]

                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        weather_code = data['weather_code']
        past_weather_code = data['past_weather_code']

        weather_unique = weather_code.unique()
        past_weather_unique = past_weather_code.unique()

        x = OneHotEncoder().fit_transform(weather_code.values.reshape(-1, 1)).toarray()
        y = OneHotEncoder().fit_transform(past_weather_code.values.reshape(-1, 1)).toarray()

        data['weather_code'] = x
        data['past_weather_code'] = y

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        # No example batch was found, so get one from the `.train` dataset
        iterator = iter(self.train)
        result1, result2 = next(iterator)
        result3, result4 = next(iterator)
        res1 = tf.concat([result1, result3], 0)
        res2 = tf.concat([result2, result4], 0)
        # And cache it for next time
        return res1, res2
