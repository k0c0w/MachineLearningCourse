import enum

import numpy as np

import utils.metrics
from models.adaboost import Adaboost

if __name__ == "__main__":
    from models.gradientboost import Gradientboost
    from datasets.wine_quality_dataset import WineQuality
    from config.gradient_boosting_config import cfg

    dataset = WineQuality(cfg, shuffle_required=cfg.shuffle, regression=True)
    gradientBoost = Gradientboost(dataset.inputs_train, dataset.targets_train, cfg.regressors_amount, cfg.alpha)
    predictions = gradientBoost(dataset.inputs_test)

    print("MSE")
    print(utils.metrics.MSE(predictions, dataset.targets_test))
    
