import numpy as np
class ModelSpecs:
    def __init__(self, model, valid_accuracy, test_accuracy):
        self.model = model
        self.valid_accuracy = valid_accuracy
        self.test_accuracy = test_accuracy

def map_predictions(old_prediction):
    return np.fromiter(map(np.argmax, old_prediction), dtype='int')

def get_accuracy(model, inputs, targets):
    from utils.metrics import accuracy
    return accuracy(map_predictions(model(inputs)), targets)

def random_search(models_number, dataset, nb_of_classes):
    from models.random_forest import RandomForestClassification
    from config.random_forest_config import cfg
    m = []
    for i in range(models_number):
        max_dims = np.random.randint(cfg.min_dimensions_to_check, cfg.max_dimensions_to_check + 1)
        treshold = np.random.randint(cfg.min_nb_thresholds, cfg.max_nb_thresholds + 1)
        trees = np.random.randint(cfg.min_amount_of_trees, cfg.max_amount_of_trees + 1)

        model = RandomForestClassification(trees, cfg.max_depth, cfg.min_entropy, cfg.min_elem, max_dims, treshold)
        model.train(dataset.inputs_train, dataset.targets_train, nb_of_classes, cfg.bagging_train_method)

        accuracy = get_accuracy(model, dataset.inputs_valid, dataset.targets_valid)
        m.append((model, accuracy))
    m.sort(key=lambda x: x[1], reverse=True)
    return m

if __name__ == "__main__":
    from datasets.digits_dataset import Digits
    from config.logistic_regression_config import cfg
    from utils.visualisation import Visualisation
    from utils.metrics import confusion_matrix
    dataset = Digits(cfg)
    models = random_search(30, dataset, 10)
    best_models = []
    for model, valid_accuracy in models[:10]:
        test_accuracy = get_accuracy(model, dataset.inputs_test, dataset.targets_test)
        best_models.append(ModelSpecs(model, valid_accuracy, test_accuracy))
    Visualisation.visualize_plot(list(map(lambda x: f'({x.model.nb_trees},{x.model.max_nb_dim_to_check},{x.model.max_nb_thresholds})', best_models)),
                                 list(map(lambda x: x.valid_accuracy, best_models)),
                                 list(map(lambda x: f'test accuracy: {x.test_accuracy:.2f}', best_models)))

    best_test_model = sorted(best_models, key=lambda x: x.test_accuracy, reverse=True)[0].model

    print(confusion_matrix(10, map_predictions(best_test_model(dataset.inputs_test)), dataset.targets_test))
    print(f'test accuracy:{get_accuracy(best_test_model, dataset.inputs_test, dataset.targets_test) * 100 :.2f}%')
