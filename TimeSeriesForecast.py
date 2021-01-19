import input_data_preparation
import numpy as np
import matplotlib.pyplot as plt
import WindowGenerator
import Baseline
import tensorflow as tf


def forecast(run_params):
    df = input_data_preparation.prepare_timeseries_data(run_params.training_years, run_params.hour_resolution)

    a = np.full((30, 3), 1)
    z = np.zeros(3)
    b = np.insert(a, 0, np.zeros(3))
    leng = a.size + 3
    b = b.reshape(31, 3)

    if run_params.predicted_label == 'pm10':
        df = df.drop('pm25', axis=1)
    if run_params.predicted_label == 'pm25':
        df = df.drop('pm10', axis=1)

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[int(n * 0.05):int(n * 0.7)]
    val_df = df[int(n * 0.63):int(n * 0.80)]
    test_df = df[int(n * 0.80):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()
    print(val_df.head())
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    OUT_STEPS = 20
    INPUT_STEPS = 20
    SHIFT = 10
    val_performance = {}
    performance = {}

    window = WindowGenerator.WindowGenerator(input_width=INPUT_STEPS, label_width=OUT_STEPS, shift=SHIFT,
                                             label_columns=['pm10'], test_df=test_df, train_df=train_df,
                                             val_df=val_df, name='Linear')

    baseline = Baseline.Baseline(label_index=column_indices['pm10'])

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])
    #

    # # val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    # # performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
    # #
    # # print('Input shape:', single_step_window.example[0].shape)
    # # print('Output shape:', baseline(single_step_window.example[0]).shape)
    # #
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])
    #
    history = compile_and_fit(model, window)

    val_performance['Linear'] = model.evaluate(window.val)
    performance['Linear'] = model.evaluate(window.test, verbose=0)

    plt.bar(x=range(len(train_df.columns)),
            height=model.layers[0].kernel[:, 0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(train_df.columns)))
    _ = axis.set_xticklabels(train_df.columns, rotation=90)
    plt.show()

    # single_step_window.plot(baseline)
    #
    # CONV_WIDTH = 5
    # LABEL_WIDTH = 10
    # INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    # window = WindowGenerator.WindowGenerator(
    #     input_width=INPUT_WIDTH,
    #     label_width=LABEL_WIDTH,
    #     shift=5,
    #     label_columns=['pm10'],
    #     train_df=train_df,
    #     test_df=test_df,
    #     val_df=val_df)
    #
    # # wide_conv_window.plot()
    # # print(wide_conv_window)
    #
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv1D(filters=32,
    #                            kernel_size=(CONV_WIDTH,),
    #                            activation='relu'),
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #     tf.keras.layers.Dense(units=1),
    # ])
    #
    # history = compile_and_fit(model, window)
    #
    # val_performance['Multi step dense'] = model.evaluate(window.val)
    # performance['Multi step dense'] = model.evaluate(window.test, verbose=0)

    # Mutli steps prediction

    # print("Output steps: " + OUT_STEPS)
    # print("Input Steps: " + INPUT_STEPS)
    # window = WindowGenerator.WindowGenerator(input_width=INPUT_STEPS,
    #                                          label_width=OUT_STEPS,
    #                                          shift=SHIFT,
    #                                          label_columns=[run_params.predicted_label],
    #                                          train_df=train_df,
    #                                          test_df=test_df,
    #                                          val_df=val_df,
    #                                          name='LSTM'
    #                                          )
    #
    # class MultiStepLastBaseline(tf.keras.Model):
    #     def call(self, inputs):
    #         return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])
    #
    # last_baseline = MultiStepLastBaseline()
    # last_baseline.compile(loss=tf.losses.MeanSquaredError(),
    #                       metrics=[tf.metrics.MeanAbsoluteError()])
    #
    # multi_val_performance = {}
    # multi_performance = {}
    #
    # model = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, lstm_units]
    #     # Adding more `lstm_units` just overfits more quickly.
    #     tf.keras.layers.LSTM(32, return_sequences=False),
    #     # Shape => [batch, out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features,
    #                           kernel_initializer=tf.initializers.zeros),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])
    #
    # history = compile_and_fit(model, window)
    #
    # IPython.display.clear_output()
    #
    # multi_val_performance['LSTM'] = model.evaluate(window.val)
    # multi_performance['LSTM'] = model.evaluate(window.test, verbose=0)

    # multi_window.plot(multi_lstm_model)
    output, samples, rmse = window.get_research_result(model, train_std[run_params.predicted_label],
                                                       train_mean[run_params.predicted_label])

    print(window)
    print(rmse)
    # prediction = multi_lstm_model.predict(multi_window.val)
    # prediction = prediction[:, :, 1]
    # prediction = prediction * np.full((1, OUT_STEPS), train_std[run_params.predicted_label]) + np.full((1, OUT_STEPS), train_mean[run_params.predicted_label])
    #
    # for _ in range(OUT_STEPS - 1 + INPUT_STEPS):
    #     inputArr = np.zeros(OUT_STEPS, dtype=float)
    #     prediction = np.insert(prediction, 0, inputArr)
    #
    # leng = int(prediction.size / 3)
    # prediction = prediction.reshape(leng, OUT_STEPS)
    # shifted_input = multi_window.val_df[run_params.predicted_label] * train_std[run_params.predicted_label] + train_mean[run_params.predicted_label]
    # output = DataFrame()
    # output['a_{}_1'.format(run_params.predicted_label)] = shifted_input
    #
    # for i in range(OUT_STEPS):
    #     output['p_{}_{}'.format(run_params.predicted_label, i + 1)] = prediction[:, i]

    return output, samples, rmse


def compile_and_fit(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=window.max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history
