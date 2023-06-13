from easydict import EasyDict
import numpy as np
cfg = EasyDict()
cfg.dataframe_path = '.\datasets\linear_regression_dataset(shuffled).csv'

basis_functions_length = 100
cfg.base_functions = [lambda x: x**i for i in range(0, basis_functions_length)]
cfg.regularization_coeff = 0
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1