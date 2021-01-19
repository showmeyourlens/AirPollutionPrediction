import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import input_data_preparation as idp
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from pandas import DataFrame


def run(run_params):
    train_df, _, test_df, _ = idp.prepare_data(
        run_params.training_years,
        run_params.testing_years,
        run_params.hour_resolution,
        False
    )

    label_vector = train_df['pm10'].iloc[:].values.reshape(-1, 1)
    test_values = test_df['pm10']
    train_df.drop(['pm10', 'pm25'], axis=1, inplace=True)
    test_df.drop(['pm10', 'pm25'], axis=1, inplace=True)

    input_vector = train_df.iloc[:, 1:].values
    test_vector = test_df.iloc[:, 1:].values

    sc_S = StandardScaler()
    sc_t = StandardScaler()
    S2 = sc_S.fit_transform(input_vector)
    t2 = sc_t.fit_transform(label_vector)

    regressor = SVR(kernel='rbf')
    regressor.fit(S2, t2.ravel())

    zp = DataFrame()
    zp["pm10"] = test_values
    zp["pm10_prediction"] = sc_t.inverse_transform(regressor.predict(sc_S.transform(test_vector)))

    return zp
