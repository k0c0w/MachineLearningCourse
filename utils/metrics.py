import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return round(np.mean(np.square(np.subtract(targets, predictions))), 2)


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return sum(map(lambda x: 1 if x[0] == x[1] else 0, zip(predictions, targets))) / targets.size

def confusion_matrix(classes_number: int, predicted_classes: np.ndarray, targets: np.ndarray):
    # Ox predicted Oy granted truth
    matrix = np.zeros((classes_number, classes_number))
    for i in range(predicted_classes.size):
        x = predicted_classes[i]
        y = targets[i]
        matrix[y, x] += 1
    return matrix
