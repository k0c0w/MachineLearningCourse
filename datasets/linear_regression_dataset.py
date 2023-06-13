import numpy as np
from utils.common_functions import read_dataframe_file
from easydict import EasyDict
class LinRegDataset():

    def __init__(self, cfg: EasyDict):
        advertising_dataframe = read_dataframe_file(cfg.dataframe_path)
        inputs, targets = np.asarray(advertising_dataframe['inputs']), np.asarray(advertising_dataframe['targets'])
        self.__divide_into_sets(inputs, targets, cfg.train_set_percent, cfg.valid_set_percent)

    def __divide_into_sets(self, inputs: np.ndarray, targets: np.ndarray, train_set_percent: float = 0.8,
                           valid_set_percent: float = 0.1) -> None:
        #inputs, targets = LinRegDataset.shuffle(inputs, targets)
        train_amount = int(inputs.size * train_set_percent)
        validation_amount = int(inputs.size * valid_set_percent)
        test_amount = inputs.size - train_amount - validation_amount
        self.inputs_train = inputs[0:train_amount]
        self.targets_train = targets[0:train_amount]
        self.inputs_valid = inputs[train_amount:train_amount+validation_amount]
        self.targets_valid = targets[train_amount:train_amount+validation_amount]
        self.inputs_test = inputs[-test_amount:]
        self.targets_test = targets[-test_amount:]

    def __call__(self) -> dict:
        return {'inputs': {'train': self.inputs_train,
                           'valid': self.inputs_valid,
                           'test': self.inputs_test},
                'targets': {'train': self.targets_train,
                            'valid': self.targets_valid,
                            'test': self.targets_test}
                }

    @staticmethod
    def shuffle(inputs: np.ndarray, targets: np.ndarray) -> (np.ndarray, np.ndarray):
        max_int = inputs.size
        indexes = np.random.randint(0, max_int, size=2*max_int)
        for i in range(max_int):
            old_index = indexes[i]
            new_index = indexes[i+1]
            inputs[old_index], inputs[new_index] = inputs[new_index], inputs[old_index]
            targets[old_index], targets[new_index] = targets[new_index], targets[old_index]
        return inputs, targets
