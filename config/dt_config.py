from utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria
from easydict import EasyDict

cfg = EasyDict()

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_set_path = 'datasets/wine-quality-white-and-red.csv'

cfg.train_repetition = 100
cfg.max_depth = 17
cfg.max_min_entropy = 10
cfg.max_min_elem = 2