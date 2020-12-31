import numpy as np
import pandas as pd
import run_params as RP
import tensorflow as tf
import input_data_preparation as data_prep
import binary_linear_regression as blr
import linear_regression as lr
import plotting as plot

from tensorflow.keras import layers

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    run_params = RP.RunParams
    run_params.training_years = ["2018"]
    # training_years = ["2018", "2015", "2016", "2017"]
    run_params.testing_years = ["2019"]
    run_params.hour_resolution = "1"
    run_params.predicted_labels = ["pm10", "pm25"]
    run_params.learning_rate = 0.01
    run_params.batch_size = 400
    run_params.epochs = 200

    run_params.pretty_print()
    out1 = lr.run_algorithm(run_params)

    # run_params.hour_resolution = "3"
    # run_params.batch_size = 300
    #
    # run_params.pretty_print()
    # out2 = lr.run_algorithm(run_params)
    #
    # run_params.hour_resolution = "6"
    # run_params.batch_size = 200
    #
    # run_params.pretty_print()
    # out3 = lr.run_algorithm(run_params)

    out1.to_csv('C:\\Users\\sliwk\\Downloads\\Dane\\Output\\' + run_params.hour_resolution + 'h out.csv', sep=';')
    # Plot a graph of the metric(s) vs. epochs.
    # list_of_metrics_to_plot = ['loss', 'root_mean_squared_error']
    # plot.plot_curve(epochs, hist, list_of_metrics_to_plot)
    # list_of_metrics_to_plot = ['precision', 'recall']
    # plot.plot_curve(epochs, hist, list_of_metrics_to_plot)

    print('finished')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
