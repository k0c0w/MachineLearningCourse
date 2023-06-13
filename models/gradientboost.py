import numpy as np
from models.decision_tree import RegressionDecisionStump


class Gradientboost():
    def __init__(self, inputs, targets, M, alpha):
        self._M = M
        self._alpha = alpha
        self._regressors = []
        self._train(inputs, targets)


    def _init_zero_learner(self, targets):
        class A:
            def __init__(self, targets):
                self._mean = np.mean(targets)
                self._predictions = np.ones(shape=targets.shape) * self._mean
            def get_predictions(self):
                return self._predictions
            def __call__(self):
                return self._mean

        a = A(targets)
        return a


    @property
    def M(self):
        return self._M

    def _train(self, inputs, targets):
        w = self._init_zero_learner(targets)
        y = w.get_predictions()
        self._regressors.append(w)
        for i in range(self.M - 1):
            r = targets - y
            w = RegressionDecisionStump(inputs, r)
            predictions = w(inputs)
            y += self._alpha * predictions
            self._regressors.append(w)

    def __call__(self, inputs):
        init = self._regressors[0]()
        prediction = np.ones(shape=inputs.shape[0]) * init
        sum = np.zeros(shape=inputs.shape[0])
        for w in self._regressors[1:]:
            sum += w(inputs)

        return prediction + self._alpha * sum