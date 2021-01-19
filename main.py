import numpy as np
import pandas as pd
from pandas import DataFrame
import run_params as RP
import NeutralNetwork as NN
import tensorflow as tf
import input_data_preparation as data_prep
import binary_linear_regression as blr
import linear_regression as lr
import plotting as plot
import TimeSeriesForecast
import SVR2


from tensorflow.keras import layers

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # data_prep.create_prepared_files(["2019", "2018", "2017", "2016", "2015"], "6")
    # data_prep.create_prepared_files(["2019", "2018", "2017", "2016", "2015"], "3")
    # data_prep.create_prepared_files(["2019", "2018", "2017", "2016", "2015"], "1")

    run_params = RP.RunParams
    # run_params.training_years = ["2019", "2018"]
    run_params.training_years = ["2019", "2017", "2016", "2015", "2018"]
    # run_params.testing_years = ["2019"]
    run_params.hour_resolution = "1"
    # run_params.predicted_labels = ["pm10", "pm25"]
    run_params.predicted_label = "pm10"
    run_params.learning_rate = 0.005
    run_params.batch_size = 100
    run_params.epochs = 40

    # out, samples, rmse_by_hour = TimeSeriesForecast.forecast(run_params)
    # out.to_csv('C:\\Users\\sliwk\\Downloads\\Dane\\Output\\{}_{}h out.csv'.format(run_params.predicted_label, run_params.hour_resolution), sep=';')
    # data = lr.do_research(run_params)
    #
    # plot.plot_curve(data['index'], data, ['pm10', 'pm10_copy'])
    # nn_result = NN.run_nn(run_params)
    #
    # svr_result = SVR2.run(run_params).reset_index(drop=True)
    #
    # run_params.pretty_print()
    lr_result = lr.run_algorithm(run_params)
    #
    out1 = DataFrame()
    # out1["svr_input"] = svr_result["pm10"]
    out1['datetime'] = lr_result['datetime']
    out1["lr_input"] = lr_result["pm10"]
    # out1["nn_input"] = nn_result["pm10"]
    # out1["svr_prediction"] = svr_result["pm10_prediction"]
    out1["lr_predictions"] = lr_result["pm10_prediction"]
    # out1["nn_predictions"] = nn_result["pm10_prediction"]

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
