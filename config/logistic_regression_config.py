from easydict import EasyDict

from utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization

# training
#cfg.weights_init_type = WeightsInitType.normal
#cfg.weights_init_kwargs = {'sigma': .05}
#cfg.gamma = .01

cfg.weights_init_kwargs = {'n_in': 64, 'n_out': 10}
cfg.weights_init_type = WeightsInitType.xavier
cfg.gamma = 0.005

#cfg.weights_init_type = WeightsInitType.uniform
#cfg.weights_init_kwargs = {'epsilon': 10}
#cfg.gamma = 0.01

#cfg.weights_init_type = WeightsInitType.chi
#cfg.weights_init_kwargs = {'n_in': 2}
#cfg.gamma = 0.0359

cfg.gd_stopping_criteria = GDStoppingCriteria.epoch
cfg.nb_epoch = 300


