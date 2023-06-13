from abc import ABC, abstractmethod

import numpy
import numpy as np


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

        self.inputs_train, self.targets_train = None, None
        self.inputs_valid, self.targets_valid = None, None
        self.inputs_test, self.targets_test = None, None,
        self.train_mean, self.train_std = None, None,

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        inputs = self.inputs
        targets = self.targets
        rows, _ = inputs.shape
        train_amount = int(rows * self.train_set_percent)
        validation_amount = int(rows * self.valid_set_percent)
        test_amount = rows - train_amount - validation_amount
        self.inputs_train = inputs[0:train_amount]
        self.targets_train = targets[0:train_amount]
        self.inputs_valid = inputs[train_amount:train_amount + validation_amount]
        self.targets_valid = targets[train_amount:train_amount + validation_amount]
        self.inputs_test = inputs[-test_amount:]
        self.targets_test = targets[-test_amount:]

    def normalization(self):
        # TODO write normalization method BONUS TASK
        pass

    def get_data_stats(self):
        self.train_mean = self.inputs_train.mean(axis=0)
        self.train_std = self.inputs_train.std(axis=0)
        return {"mean": self.train_mean, "std": self.train_std}

    def standardization(self):
        std = self.train_std
        std[np.where(std == 0)] = 1
        self.inputs_train = (self.inputs_train - self.train_mean) / std
        self.inputs_test = (self.inputs_test - self.train_mean) / std
        self.inputs_valid = (self.inputs_valid - self.train_mean) / std

class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        matrix = np.zeros((targets.size, number_classes))
        matrix[np.arange(targets.size), targets] = 1
        return matrix
