from dataclasses import dataclass


class RunParams:
    training_years = []
    testing_years = []
    hour_resolution = "0"
    predicted_label = "err"
    learning_rate = float
    epochs = int
    batch_size = int

    def __init__(self):
        pass

    @classmethod
    def pretty_print(cls):
        print("Training years: {}".format(' '.join([str(elem) for elem in cls.training_years])))
        print("Testing years: {}".format(' '.join([str(elem) for elem in cls.testing_years])))
        print("Hour resolution: {}h".format(cls.hour_resolution))
        print("Predicted labels: {}".format(cls.predicted_label))
        print("Learning rate : {}".format(cls.learning_rate))
        print("Epochs: {}".format(cls.epochs))
        print("Batch size: {}".format(cls.batch_size))
