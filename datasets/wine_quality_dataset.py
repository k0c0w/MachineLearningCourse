import numpy
import numpy as np
import pandas

from easydict import EasyDict
from datasets.base_dataset_classes import BaseClassificationDataset
from utils.enums import SetType

class WineQuality(BaseClassificationDataset):

    def __init__(self, cfg: EasyDict, shuffle_required=False, regression=False):
        super(WineQuality, self).__init__(cfg.train_set_percent, cfg.valid_set_percent)
        wine_params = pandas.read_csv(cfg.data_set_path)
        self._inputs = wine_params.drop('type', axis=1).to_numpy(dtype='float')
        classes = wine_params['type'].to_numpy()
        int_classes = np.zeros(shape=classes.shape, dtype='int')
        unique_classes = np.unique(classes)
        self._k = unique_classes.size
        _, self.d = self.inputs.shape
        for i in np.arange(unique_classes.size):
            value = unique_classes[i]
            int_classes[classes == value] = i
        self._targets = int_classes
        if regression:
            inputs = wine_params.drop('type', axis=1).drop('quality', axis=1).to_numpy(dtype='float')
            self._inputs = numpy.c_[int_classes, inputs]
            self._targets = wine_params['quality'].to_numpy()

        if shuffle_required:
            self.__shuffle_collection()
        self.divide_into_sets()

    def __shuffle_collection(self):
        size = self.targets.size
        inputs = self._inputs
        targets = self._targets
        for i in np.arange(size):
            new_index = np.random.randint(size)
            inputs[i], inputs[new_index] = inputs[new_index], inputs[i]
            targets[i], targets[new_index] = targets[new_index], targets[i]


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