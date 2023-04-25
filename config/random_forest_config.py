from easydict import EasyDict

cfg = EasyDict()

cfg.bagging_train_method = False

#random search bounds
cfg.max_amount_of_trees = 20
cfg.min_amount_of_trees = 3
cfg.max_dimensions_to_check = 10
cfg.min_dimensions_to_check = 2
cfg.max_nb_thresholds = 15
cfg.min_nb_thresholds = 2

#dt_params
cfg.max_depth = 5
cfg.min_entropy = .1
cfg.min_elem = 20