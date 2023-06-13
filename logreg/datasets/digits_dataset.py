import numpy as np
from easydict import EasyDict
from sklearn.datasets import load_digits

from logreg.datasets.base_dataset_classes import BaseClassificationDataset
from logreg.utils.enums import DataProcessTypes, SetType


class Digits(BaseClassificationDataset):

    def __init__(self, cfg: EasyDict):

        super(Digits, self).__init__(cfg.train_set_percent, cfg.valid_set_percent)
        digits = load_digits()

        # define properties
        self.inputs = digits.data
        self.targets = digits.target
        self.k = np.max(self.targets) + 1
        self.d = self.inputs.shape[1]

        # divide into sets
        self.divide_into_sets()
        self.test_set_images = digits.images[self.inputs_train.shape[0]+self.inputs_valid.shape[0]:]

        # preprocessing
        if cfg.data_preprocess_type.value == DataProcessTypes.standardization.value:
            self.get_data_stats()
            self.standardization()
        elif cfg.data_preprocess_type.value == DataProcessTypes.normalization.value:
            self.get_data_stats()
            self.normalization()
        else:
            raise Exception('No such preprocessing function!')

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = value

    def __call__(self, set_type: SetType) -> dict:
        inputs, targets = getattr(self, f'inputs_{set_type.name}'), getattr(self, f'targets_{set_type.name}')
        return {'inputs': inputs,
                'targets': targets,
                'onehotencoding': self.onehotencoding(targets, self.k)}
