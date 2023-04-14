from typing import Union
from enum import Enum
import numpy
import utils.metrics as metrics
import numpy as np
from easydict import EasyDict
from datasets.base_dataset_classes import BaseClassificationDataset

class WeightsDto:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int):
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        self.__weights = None
        self.__bias = None
        self.E_history = []
        self.train_set_accuracy_history = []
        self.validation_set_accuracy_history = []
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)

    @property
    def weights(self):
        return self.__weights

    @property
    def bias(self):
        return self.__bias

    def weights_init_normal(self, sigma):
        self.__bias = np.random.normal(0, sigma, (1, self.k))
        self.__weights = np.random.normal(0, sigma, (self.k, self.d))

    def weights_init_uniform(self, epsilon):
        self.__bias = np.zeros((1, self.k))
        self.__weights = np.random.uniform(0, epsilon, (self.k, self.d))

    def weights_init_xavier(self, n_in, n_out):
        sigma = np.sqrt(1 / (n_in + n_out))
        weights = np.random.normal(0, sigma, (n_out, n_in))
        self.__bias = np.random.normal(0, sigma, (1, self.k))
        self.__weights = weights

    def weights_init_chi(self, n_in):
        self.__bias = np.random.chisquare(n_in, (1, self.k))
        self.__weights = np.random.chisquare(n_in, (self.k, self.d))

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        b = np.amax(model_output, axis=1)
        model_output -= b.reshape((b.shape[0], 1))
        exp_powered = np.exp(model_output)
        exp_sum = np.sum(exp_powered, axis=1)
        return exp_powered / exp_sum.reshape((exp_sum.shape[0], 1))

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        rows = inputs.shape[0]
        z = np.zeros((rows, self.k))
        for i in range(rows):
            z[i] = self.__get_model_output(inputs[i])
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        return self.__weights @ inputs + self.__bias

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        onehotencoding = BaseClassificationDataset.onehotencoding(targets, self.k)
        return (model_confidence - onehotencoding).transpose() @ inputs

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        onehotencoding = BaseClassificationDataset.onehotencoding(targets, self.k)
        return (model_confidence - onehotencoding).transpose() @ np.ones((targets.shape[0],))

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        gamma = self.cfg.gamma
        self.__bias -= gamma * self.__get_gradient_b(targets, model_confidence)
        self.__weights -= gamma * self.__get_gradient_w(inputs, targets, model_confidence)

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None):

        predictions = np.empty((inputs_train.shape[0], self.k))
        for i in range(inputs_train.shape[0]):
            predictions[i] = self.__get_model_output(inputs_train[i])
        self.__log_metrics(epoch, inputs_train, targets_train, train_predictions=predictions,
                           targets_valid=targets_valid, inputs_valid=inputs_valid)
        self.__weights_update(inputs_train, targets_train, self.__softmax(predictions))
        """
        :param targets_train: onehot-encoding
        :param epoch: number of loop iteration
        """

    def __log_metrics(self, epoch, inputs_train, targets_train, train_predictions=None, inputs_valid=None, targets_valid=None):
        E = self.__target_function_value(inputs_train, targets_train)
        self.E_history.append(E)
        train_accuracy, train_confusion_matrix = self.__validate(inputs_train, targets_train, train_predictions)
        self.train_set_accuracy_history.append(train_accuracy)
        self.validation_set_accuracy_history.append(self.__validate(inputs_valid, targets_valid)[0])
        if epoch % 10 == 0:
            self.__validate(inputs_train, targets_train)
            print(
                f'Epoch {epoch}:\nAccuracy:{train_accuracy * 100:.2f}%\nConfusion matrix:\n{train_confusion_matrix}')
            print(f'E(W,b): {E}\n')

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        for epoch in range(self.cfg.nb_epoch):
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with gradient norm stopping criteria BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with stopping criteria - norm of difference between ￼w_k-1 and w_k;￼BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with stopping criteria - metric (accuracy, f1 score or other) value on validation set is not growing;￼
        #  BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                inputs_valid,
                                                                                targets_valid)
        if not (inputs_valid is None or targets_valid is None):
            _, confusion_matrix = self.__validate(inputs_valid, targets_valid)
            self.validation_set_confusion_matrix = confusion_matrix


    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                model_confidence: Union[np.ndarray, None] = None) -> float:
        ohe = BaseClassificationDataset.onehotencoding(targets, self.k)
        target_function_value = 0
        if not model_confidence:
            for i in range(inputs.shape[0]):
                prediction = self.__get_model_output(inputs[i])
                zk = np.where(ohe[i] == 1)
                target_function_value += np.log(np.sum(np.exp(prediction), axis=1)) - prediction[0, zk]
        else:
            for i in range(inputs.shape[0]):
                target_function_value -= np.log(model_confidence[np.where(ohe[i] == 1)])
        return target_function_value[0][0]

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        predicted_classes = np.argmax(model_confidence, axis=1)
        return metrics.accuracy(predicted_classes, targets), metrics.confusion_matrix(self.k, predicted_classes, targets)

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=1)
        return predictions, model_confidence

    def save_weights(self, directory=None):
        import pickle
        import datetime
        if not directory:
            directory = ""
        directory += f'{datetime.datetime.now().strftime("%I-%M%p-%B-%d-%Y")}_model.pickle'
        dto = WeightsDto(self.__weights, self.__bias)
        with open(directory, 'wb') as file:
            pickle.dump(dto, file)

    @staticmethod
    def load_from_file(path, model_config):
        import pickle
        with open(path, 'rb') as file:
            dto = pickle.load(file)
        if not dto:
            raise ValueError("Could not read weights")
        sample = np.ndarray([])
        if type(dto.weights) != type(sample) or type(dto.bias) != type(sample):
            raise TypeError("Read values are not np.ndarrays")
        K, D = dto.weights.shape
        if dto.bias.shape != (1, K):
            raise ValueError("Weights and bias shape mismatch")
        model = LogReg(model_config, K, D)
        model.__weights = dto.weights
        model.__bias = dto.bias
        return model
