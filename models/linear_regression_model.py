import numpy as np


class LinearRegression():

    def __init__(self, base_functions: list, reg_coeff: float):
        self.weights = np.random.normal(loc=0, scale=1, size=(len(base_functions), 1))
        self.base_functions = base_functions
        self.reg_coeff = reg_coeff


    def __pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        U, s, V_transposed = np.linalg.svd(matrix)
        S_inversed = self.__inverse_singular_matrix(s, U.shape[0], V_transposed.shape[0], self.reg_coeff)

        return V_transposed.transpose() @ S_inversed @ U.transpose()

    def __inverse_singular_matrix(self, diagonal_values: np.ndarray, n, m_plus_one, regularisation_coeff=0):
        epsilon = np.finfo(float).eps * max(n, m_plus_one) * max(diagonal_values)
        singular_matrix = np.zeros((n, m_plus_one))
        elem = diagonal_values[0]
        singular_matrix[0][0] = 1 / elem if elem > epsilon else 0
        for i in range(1, min(n, m_plus_one)):
            elem = diagonal_values[i]
            if elem > epsilon:
                if regularisation_coeff == 0:
                    singular_matrix[i][i] = 1 / elem
                else:
                    singular_matrix[i][i] = elem / (elem * elem + regularisation_coeff)

        return singular_matrix.transpose()

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        functions = self.base_functions
        n = len(functions)
        design_matrix = np.ones((inputs.size, 0))
        for i in range(0, n):
            vectorized = np.vectorize(functions[i])
            applied = vectorized(inputs)
            design_matrix = np.hstack((design_matrix, applied.reshape((applied.size, 1))))

        return design_matrix

    def __calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        self.weights = pseudoinverse_plan_matrix @ targets

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        return plan_matrix @ self.weights

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        # prepare data
        plan_matrix = self.__plan_matrix(inputs)
        pseudoinverse_plan_matrix = self.__pseudoinverse_matrix(plan_matrix)

        # train process
        self.__calculate_weights(pseudoinverse_plan_matrix, targets)


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions
