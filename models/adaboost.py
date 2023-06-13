import numpy as np

from datasets.dataset_titanik import Titanic
from models.decision_tree import DecisionStump

class Adaboost():
    def __init__(self, M, error_treshold_delta=0.001):
        self.M = M
        self.__error_treshold_delta = error_treshold_delta
        self.__weak_classifiers = []
        self.__weak_classifiers_weights = []

    def __init_weights(self,N):
        return np.ones(N) / N

    def update_weights(self,gt,predict,weights,weight_weak_classifiers):
        pows = np.zeros(shape=weights.shape)
        pows[predict != gt] = 1
        new_weights = weights * weight_weak_classifiers * np.exp(pows)

        return new_weights / np.sum(new_weights)

    def calculate_error(self, targets, predict, weights):
        wrong_answers = targets != predict

        return np.sum(weights[wrong_answers])

    def claculate_classifier_weight(self,error):
        return np.log((1 - error) / error)

    def train(self, inputs, targets):
        weights = self.__init_weights(inputs.shape[0])
        for i in range(self.M):
            weak_classifier = self.__train_weak_classifier(inputs, weights, targets)
            predictions = weak_classifier(inputs)
            error = self.calculate_error(targets, predictions, weights)
            if 0.5 - error < self.__error_treshold_delta:
                break
            elif error < self.__error_treshold_delta:
                raise Exception("Ideal classifier")

            alpha = self.claculate_classifier_weight(error)
            weights = self.update_weights(targets, predictions, weights, alpha)
            self.__weak_classifiers.append(weak_classifier)
            self.__weak_classifiers_weights.append(alpha)

    def __call__(self,vectors):
        return np.sign(sum(map(lambda x: x[0] * x[1](vectors), zip(self.__weak_classifiers_weights, self.__weak_classifiers))))

    @staticmethod
    def __train_weak_classifier(inputs, inputs_weights, targets):
        return DecisionStump(inputs, targets, inputs_weights)
