from abc import abstractmethod, ABC

import numpy as np
from typing import Optional

class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_ind = None
        self.split_val = None
        self.terminal_node_value = None

    @property
    def is_terminal(self):
        return self.terminal_node_value is not None


class BinaryDTBase(ABC):

    def __init__(self, max_depth, min_entropy=0, min_elem=0):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.root = Node()

    def train(self, inputs, targets, random_mode:Optional[tuple]=None):
        entropy_val = self.__function_for_information_gain(targets, len(targets))
        self.__nb_dim = inputs.shape[1]
        self.__all_dim = np.arange(self.__nb_dim)

        if random_mode:
            self.max_nb_dim_to_check, self.max_nb_thresholds = random_mode
            self.__get_axis, self.__get_threshold = self.__get_random_axis, self.__generate_random_threshold
        else:
            self.__get_axis, self.__get_threshold = self.__get_all_axis, self.__generate_all_threshold
        self.__build_tree(inputs, targets, self.root, 1, entropy_val)

    def __get_random_axis(self):
        return np.random.choice(self.__nb_dim, self.max_nb_dim_to_check)

    def __get_all_axis(self):
        return self.__all_dim

    def __generate_all_threshold(self, inputs):
        return np.unique(inputs)

    def __generate_random_threshold(self, inputs):
        n = min(self.max_nb_thresholds, len(inputs))
        min_value, max_value = np.min(inputs), np.max(inputs)
        return np.random.uniform(min_value, max_value, size=n)
        """
        :param inputs: все элементы обучающей выборки(дошедшие до узла) выбранной оси
        :return: пороги, выбранные с помощью равномерного распределения.
        Количество порогов определяется значением параметра self.max_nb_thresholds
        """

    def __inf_gain(self, targets_left, targets_right, root_value, N):
        left_value = self.__function_for_information_gain(targets_left, targets_left.size)
        right_value = self.__function_for_information_gain(targets_right, targets_right.size)
        Ni0 = targets_left.size
        Ni1 = targets_right.size
        return root_value - Ni0 * left_value / N - Ni1 * right_value / N, left_value, right_value
        """
        :param targets_left: targets для элементов попавших в левый узел
        :param targets_right: targets для элементов попавших в правый узел
        :param root_value: энтропия или дисперсия узла-родителя
        :param N: количество элементов, дошедших до узла родителя
        :return: information gain, энтропия/дисперсия для левого узла, энтропия/дисперсия для правого узла
        """

    def __build_splitting_node(self, inputs, targets, entropy, N):
        d, t, left_Ifunc_value, right_Ifunc_value = self.__get_max_inf_gain_dimension_tau_pair(inputs, targets, entropy, N)
        right_ind = inputs[:, d] > t
        left_ind = ~right_ind
        return d, t, left_ind, right_ind, left_Ifunc_value, right_Ifunc_value

    def __get_max_inf_gain_dimension_tau_pair(self, inputs, targets, root_value, N):
        chosen_dimension, chosen_t, left_Ifunc_value, right_Ifunc_value = None, None, None, None
        max_information_gain = None
        for d in self.__get_axis():
            inputs_d_column = inputs[:, d]
            for t in self.__get_threshold(inputs_d_column):
                targets_left, targets_right = self.__split_targets_by_tau_and_d_value(inputs_d_column, targets, t)
                current_inf_gain, left_I_or_V, right_I_or_V = self.__inf_gain(targets_left, targets_right, root_value, N)
                if not max_information_gain or current_inf_gain > max_information_gain:
                    max_information_gain = current_inf_gain
                    chosen_dimension, chosen_t = d, t
                    left_Ifunc_value, right_Ifunc_value = left_I_or_V, right_I_or_V
        return chosen_dimension, chosen_t, left_Ifunc_value, right_Ifunc_value

    def __split_targets_by_tau_and_d_value(self, inputs, targets, tau):
        mask = inputs > tau
        targets_left = targets[~mask]
        targets_right = targets[mask]

        return targets_left, targets_right

    def __build_tree(self, inputs, targets, node, depth, entropy):
        N = len(targets)
        if depth >= self.max_depth or entropy <= self.min_entropy or N <= self.min_elem:
            node.terminal_node_value = self.__get_terminal_node_prediction(targets)
        else:
            ax_max, tay_max, ind_left, ind_right, disp_left_max, disp_right_max = self.__build_splitting_node(inputs, targets, entropy, N)
            node.split_ind = ax_max
            node.split_val = tay_max
            node.left = Node()
            node.right = Node()
            self.__build_tree(inputs[ind_left], targets[ind_left], node.left, depth + 1, disp_left_max)
            self.__build_tree(inputs[ind_right], targets[ind_right], node.right, depth + 1, disp_right_max)

    def __find_prediction(self, input, node: Node):
        current = node
        while not current.is_terminal:
            if input[current.split_ind] > current.split_val:
                current = current.right
            else:
                current = current.left
        return current.terminal_node_value

    def __call__(self, inputs):
        result = np.ndarray(shape=len(inputs), dtype='object')
        for i, input in enumerate(inputs):
            prediction = self.__find_prediction(input, self.root)
            result[i] = prediction
        return result
        """
        :param inputs: вектора характеристик
        :return: предсказания целевых значений
        """

    @abstractmethod
    def __function_for_information_gain(self, targets, N) -> float:
        '''
            whether Variance or Entropy
            :return: float - value of H(S) or V(S)
        '''
        pass

    @abstractmethod
    def __get_terminal_node_prediction(self, targets):
        pass

class BinaryDTClassification(BinaryDTBase):

    def __init__(self, classes_amount, max_depth, min_entropy=0, min_elem=0):
        self.total_classes = classes_amount
        super().__init__(max_depth, min_entropy=min_entropy, min_elem=min_elem)

    @property
    def classes_amount(self):
        return self.total_classes

    def _BinaryDTBase__get_terminal_node_prediction(self, target):
        """
            :param target: классы элементов обучающей выборки, дошедшие до узла
            :return: уверенность
        """
        unique, counts = np.unique(target, return_counts=True)
        predictions = np.zeros(self.total_classes)
        for i in np.arange(unique.size):
            predictions[unique[i]] = counts[i]

        return predictions / target.size

    def _BinaryDTBase__function_for_information_gain(self, targets, N):
        '''
            Shanon entropy
        '''
        _, counts = np.unique(targets, return_counts=True)
        counts = counts / N
        log = np.log2(counts)
        return -np.sum(counts * log)

class BinaryDTRegression(BinaryDTBase):

    def __init__(self, max_depth, min_entropy=0, min_elem=0):
        super().__init__(max_depth, min_entropy=min_entropy, min_elem=min_elem)

    def _BinaryDTBase__get_terminal_node_prediction(self, target):
        return np.mean(target)

    def _BinaryDTBase__function_for_information_gain(self, targets, N):
        '''
            Variance function
        '''
        mean = np.mean(targets)
        normalized = targets - mean

        return np.sum(np.square(normalized)) / N

class RegressionDecisionStump:
    def __init__(self, inputs, targets):
        self.root = Node()
        self.__all_dim = np.arange(inputs.shape[1])
        N = len(targets)
        variance = self.__variance(targets)
        self.__build_tree(inputs, targets, self.root, 0, variance, N)

    def __variance(self, targets):
        if (len(targets) == 0):
            return 0
        return np.var(targets)

    def __build_tree(self, inputs, targets, node, depth, variance, N):
        if depth == 3:
            node.terminal_node_value = self.__get_terminal_node_prediction(targets)
        else:
            node.split_ind, node.split_val, ind_left, ind_right, variance_l, variance_r = self.__build_splitting_node(inputs, targets, variance, N)
            node.left, node.right = Node(), Node()
            self.__build_tree(inputs[ind_left], targets[ind_left], node.left, depth + 1, variance_l, len(ind_left))
            self.__build_tree(inputs[ind_right], targets[ind_right], node.right, depth + 1, variance_r, len(ind_right))

    def __build_splitting_node(self, inputs, targets, variance, N):
        chosen_dimension, chosen_t, left_Ifunc_value, right_Ifunc_value = None, None, None, None
        max_information_gain = None
        for d in self.__all_dim:
            inputs_d_column = inputs[:, d]
            for t in np.unique(inputs_d_column):
                targets_left_ind, targets_right_ind = self.__split_targets_by_tau_and_d_value(inputs_d_column, t)
                current_inf_gain, left_I, right_I = self.__inf_gain(targets, targets_left_ind, targets_right_ind, variance, N)
                if not max_information_gain or current_inf_gain > max_information_gain:
                    max_information_gain = current_inf_gain
                    chosen_dimension, chosen_t = d, t
                    left_Ifunc_value, right_Ifunc_value = left_I, right_I

        right_ind = inputs[:, chosen_dimension] > chosen_t
        left_ind = ~right_ind
        return chosen_dimension, chosen_t, left_ind, right_ind, left_Ifunc_value, right_Ifunc_value

    def __split_targets_by_tau_and_d_value(self, inputs, tau):
        mask = inputs > tau

        return ~mask, mask

    def __inf_gain(self, targets, left_ind, right_ind, variance, N):
        total = len(left_ind)
        left = len(np.where(left_ind==True)[0])
        Ni0 = left
        Ni1 = total - left
        varianceLeft = self.__variance(targets[left_ind])
        varianceRight = self.__variance(targets[right_ind])

        return variance - (Ni0 * varianceLeft + Ni1 * varianceRight) / N, varianceLeft, varianceRight

    def __get_terminal_node_prediction(self, targets):
        if(len(targets) == 0):
            return 0
        return np.mean(targets)

    def __call__(self, inputs):
        def __get_prediction(input):
            current = self.root
            while not current.is_terminal:
                if input[current.split_ind] > current.split_val:
                    current = current.right
                else:
                    current = current.left
            return current.terminal_node_value

        predictions = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            predictions[i] = __get_prediction(inputs[i])

        return predictions

#old name DecisionStump
class ClassificationDecisionStump:
    def __init__(self, inputs, targets, weights):
        self.root = Node()
        self.__all_dim = np.arange(inputs.shape[1])
        self.__classes = np.unique(targets)

        N = np.sum(weights)
        entropy_val = self.__entropy(targets, weights, N)
        self.__build_tree(inputs, targets, weights, self.root, 0, entropy_val, N)

    def __entropy(self, targets, weights, N):
        mask = targets == self.__classes[0]
        if N == 0:
            first, second = 1, 1
        else:
            first = np.sum(weights[mask]) / N
            second = np.sum(weights[~mask]) / N
            if first == 0:
                first = 1
            if second == 0:
                second = 1
        return -(first * np.log(first) + second * np.log(second))

    def __build_tree(self, inputs, targets, weights, node, depth, entropy, N):
        if depth == 1:
            node.terminal_node_value = self.__get_terminal_node_prediction(targets, weights)
        else:
            node.split_ind, node.split_val, ind_left, ind_right, entropy_left, entropy_right = self.__build_splitting_node(weights, inputs, targets, entropy, N)
            node.left, node.right = Node(), Node()
            left_weights = weights[ind_left]
            right_weights = weights[ind_right]
            self.__build_tree(inputs[ind_left], targets[ind_left], left_weights, node.left, depth + 1, entropy_left, np.sum(left_weights))
            self.__build_tree(inputs[ind_right], targets[ind_right], right_weights, node.right, depth + 1, entropy_right, np.sum(right_weights))

    def __build_splitting_node(self, weights, inputs, targets, entropy, N):
        chosen_dimension, chosen_t, left_Ifunc_value, right_Ifunc_value = None, None, None, None
        max_information_gain = None
        for d in self.__all_dim:
            inputs_d_column = inputs[:, d]
            for t in np.unique(inputs_d_column):
                targets_left_ind, targets_right_ind = self.__split_targets_by_tau_and_d_value(inputs_d_column, t)
                current_inf_gain, left_I, right_I = self.__inf_gain(weights, targets, targets_left_ind, targets_right_ind, entropy, N)
                if not max_information_gain or current_inf_gain > max_information_gain:
                    max_information_gain = current_inf_gain
                    chosen_dimension, chosen_t = d, t
                    left_Ifunc_value, right_Ifunc_value = left_I, right_I

        right_ind = inputs[:, chosen_dimension] > chosen_t
        left_ind = ~right_ind
        return chosen_dimension, chosen_t, left_ind, right_ind, left_Ifunc_value, right_Ifunc_value

    def __split_targets_by_tau_and_d_value(self, inputs, tau):
        mask = inputs > tau

        return ~mask, mask

    def __inf_gain(self, weights, targets, left_ind, right_ind, entropy, N):
        left_weights = weights[left_ind]
        right_weights = weights[right_ind]
        Ni0 = np.sum(left_weights)
        Ni1 = np.sum(right_weights)
        entropyLeft = self.__entropy(targets[left_ind], left_weights, N)
        entropyRight = self.__entropy(targets[right_ind], right_weights, N)

        return entropy - (Ni0 * entropyLeft  + Ni1 * entropyRight) / N, entropyLeft, entropyRight

    def __get_terminal_node_prediction(self, targets, weights):
        mask = targets == self.__classes[0]
        k1_weights = np.sum(weights[mask])
        k2_weights = np.sum(weights[~mask])

        return self.__classes[0] if k1_weights > k2_weights else self.__classes[1]

    def __call__(self, inputs):
        def __get_prediction(input):
            current = self.root
            while not current.is_terminal:
                if input[current.split_ind] > current.split_val:
                    current = current.right
                else:
                    current = current.left
            return current.terminal_node_value

        predictions = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            predictions[i] = __get_prediction(inputs[i])

        return predictions
