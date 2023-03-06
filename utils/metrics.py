import numpy as np

def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return round(np.mean(np.square(np.subtract(targets, predictions))), 2)
