from dataclasses import dataclass


class RunParams:
    training_years = []
    testing_years = []
    hour_resolution = "0"
    predicted_labels = []
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
        print("Predicted labels: {}".format(' '.join([str(elem) for elem in cls.predicted_labels])))
        print("Learning rate : {}".format(cls.learning_rate))
        print("Epochs: {}".format(cls.epochs))
        print("Batch size: {}".format(cls.batch_size))
