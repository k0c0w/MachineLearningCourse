import numpy as np
import logreg.utils.metrics
from logreg.utils.visualisation import Visualisation
from logreg.models.logistic_regression_model import LogReg

def visualize_plots(digits, model):
    Visualisation.metric_changing_per_iteration(model.train_set_accuracy_history, "Train set accuracy")
    Visualisation.metric_changing_per_iteration(model.validation_set_accuracy_history, "Validation set accuracy")
    Visualisation.metric_changing_per_iteration(model.E_history, "Target function (E)")
    Visualisation.visualize_most_predicted_pictures(model, digits.inputs_test, digits.targets_test, digits.test_set_images)

def execute(digits, model):
    model.train(digits.inputs_train, digits.targets_train, digits.inputs_valid, digits.targets_valid)
    print(f'Validation set accuracy: {model.validation_set_accuracy_history[-1] * 100:.2f}%')
    print(f'Validation set confusion matrix:\n{model.validation_set_confusion_matrix}')
    out = model(digits.inputs_test)
    print(logreg.utils.metrics.confusion_matrix(10, out[0], digits.targets_test))
    print(logreg.utils.metrics.accuracy(np.argmax(out[1], axis=1), digits.targets_test))

def save_model(model: LogReg):
    model.save_weights()
    print("model has been saved")

def load_model(filePath, config):
    return LogReg.load_from_file(filePath, config)

if __name__ == "__main__":
    from logreg.datasets.digits_dataset import Digits
    from config.logistic_regression_config import cfg

    digits = Digits(cfg)
    model = LogReg(cfg, digits.k, digits.d)

    execute(digits, model)
    save_model(model)
    visualize_plots(digits, model)