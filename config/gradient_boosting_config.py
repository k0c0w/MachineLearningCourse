from easydict import EasyDict

cfg = EasyDict()

cfg.train_set_percent = 0.9
cfg.valid_set_percent = 0
cfg.data_set_path = 'datasets/wine-quality-white-and-red.csv'

cfg.regressors_amount = 50
cfg.alpha = .5
cfg.shuffle = True