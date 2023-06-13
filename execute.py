import numpy as np
from configs.linear_regression_cfg import cfg
from models.linear_regression_model import LinearRegression
from datasets.linear_regression_dataset import LinRegDataset
from utils.metrics import MSE
from utils.visualisation import Visualisation
from utils.dtos import ModelDescription


def experiment(poly, coeff):
    dataset = LinRegDataset(cfg)()
    model = LinearRegression([lambda x, pow=i: x ** pow for i in range(poly)], coeff)
    model.train_model(dataset['inputs']['train'], dataset['targets']['train'])

    inputs = dataset['inputs']['test']
    targets = dataset['targets']['test']
    
    indexes = np.argsort(inputs)
    inputs = inputs[indexes]
    targets = targets[indexes]

    predictions = model(inputs)
    targets = dataset['targets']['test']
    error = MSE(predictions, targets)
    Visualisation.visualise_predicted_trace(predictions, inputs, targets,
                        plot_title=f'Полином степени {poly}; MSE = {error}')

def random_search(repetitions: int, polynomial_degree_range: (int, int), reg_coeff_range: (int, int)):
    result = []
    dataset = LinRegDataset(cfg)()
    for _ in range(repetitions):
        coeff = np.random.randint(reg_coeff_range[0], reg_coeff_range[1] + 1)
        poly = np.random.randint(polynomial_degree_range[0], polynomial_degree_range[1] + 1)
        base_functions = [lambda x, i=i: x**i for i in range(poly)]

        model = LinearRegression(base_functions, coeff)
        model.train_model(dataset['inputs']['train'], dataset['targets']['train'])

        valid_mse = MSE(model(dataset['inputs']['valid']), dataset['targets']['valid'])
        test_mse = MSE(model(dataset['inputs']['test']), dataset['targets']['test'])
        result.append(ModelDescription(poly, coeff, valid_error=valid_mse, test_error=test_mse))

    result.sort(key=lambda x: x.ErrorOnValid)
    return result

def visualise_top_10_from_random(repetitions: int, polynomial_degree_range: (int, int), reg_coeff_range: (int, int)):
    best = random_search(repetitions, polynomial_degree_range, reg_coeff_range)[:10]
    Visualisation.visualise_error(best)

def visualise_top_10():
    best = [
        ModelDescription(12, 0, 99.79, 99.68),
        ModelDescription(14, 0, 100.27, 102.84),
        ModelDescription(50, 0, 100.23, 106.65),
        ModelDescription(51, 0, 100.31, 106.66),
        ModelDescription(21, 0, 100.33, 104.47),
        ModelDescription(18, 0, 100.37, 104.86),
        ModelDescription(23, 0, 100.43, 104.45),
        ModelDescription(34, 0, 100.92, 104.82),
        ModelDescription(10, 0, 105.3, 98.6),
        ModelDescription(11, 0, 105.33, 98.5),
    ]
    Visualisation.visualise_error(best)

if __name__ == '__main__':
    visualise_top_10()
    experiment(12, 1e-5)
    experiment(12, 0)
