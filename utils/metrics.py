import numpy
import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    pass


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

def precision(predictions, targets):
    true_pos = np.logical_and(predictions == targets, np.logical_or(predictions > 0, predictions == True))
    false_pos = np.logical_and(np.logical_or(predictions > 0, predictions == True), np.logical_or(targets < 0, targets == False))
    true_pos_amount = predictions[true_pos].size
    false_neg_amount = predictions[false_pos].size

    return true_pos_amount / (true_pos_amount + false_neg_amount)

def recall(predictions, targets):
    true_pos = np.logical_and(predictions == targets, np.logical_or(predictions > 0, predictions == True))
    false_neg = np.logical_and(np.logical_or(predictions < 0, predictions == False), np.logical_or(targets > 0, targets == True))

    true_pos_amount = predictions[true_pos].size
    false_neg_amount = predictions[false_neg].size

    return true_pos_amount / (true_pos_amount + false_neg_amount)

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)
