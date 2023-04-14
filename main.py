import numpy as np
import utils.metrics
from utils.visualisation import Visualisation
from models.decision_tree import BinaryDTRegression, BinaryDTClassification

def map_predictions(old_prediction):
    return np.fromiter(map(np.argmax, old_prediction), dtype='int')


def train_model_with_random_values(cfg, classes, inputs, targets) -> BinaryDTClassification:
    max_depth = np.random.randint(cfg.max_depth)
    min_elem = np.random.randint(cfg.max_min_elem)
    min_entropy = np.random.randint(cfg.max_min_entropy)
    dt = BinaryDTClassification(classes, max_depth, min_elem=min_elem, min_entropy=min_entropy)
    dt.train(inputs, targets)
    return dt


def get_best_model(cfg, classes, dataset):
    from utils.metrics import accuracy
    best_accuracy, best_model = None, None
    for i in range(cfg.train_repetition):
        model = train_model_with_random_values(cfg, classes, dataset.inputs_train,
                                                                     dataset.targets_train)
        model_accuracy = accuracy(map_predictions(model(dataset.inputs_valid)), dataset.targets_valid)
        if not best_accuracy or model_accuracy > best_accuracy:
            best_accuracy = model_accuracy
            best_model = model

    return best_model

def print_model_specs(cfg, dataset, classes):
    from utils.metrics import accuracy, confusion_matrix

    model = get_best_model(cfg, classes, dataset)
    valid_predictions = map_predictions(model(dataset.inputs_valid))
    test_predictions = map_predictions(model(dataset.inputs_test))
    print(f'depth:{model.max_depth} entropy:{model.min_entropy} minimal elem: {model.min_elem}')
    print(f'validation accuracy:{accuracy(valid_predictions, dataset.targets_valid)}')
    print(f'test accuracy:{accuracy(test_predictions, dataset.targets_test)}')
    print(f'validation confusion:\n{confusion_matrix(classes, valid_predictions, dataset.targets_valid)}')
    print(f'test confusion:\n{confusion_matrix(classes, test_predictions, dataset.targets_test)}')

def wine_quality():
    from datasets.wine_quality_dataset import WineQuality
    from config.dt_config import cfg
    dataset = WineQuality(cfg)
    print_model_specs(cfg, dataset, 2)

def digits():
    from datasets.digits_dataset import Digits
    from config.logistic_regression_config import cfg
    #31 1 4
    # 44 1 1 ~78%
    cfg.train_repetition = 100
    cfg.max_depth = 50
    cfg.max_min_entropy = 15
    cfg.max_min_elem = 3
    dataset = Digits(cfg)
    print_model_specs(cfg, dataset, 10)

if __name__ == "__main__":
    digits()
    #d:7 entropy:0 elem: 2

