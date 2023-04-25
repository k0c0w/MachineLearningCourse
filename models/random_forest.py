import numpy as np

from models.decision_tree import BinaryDTClassification

class RandomForestClassification():

    def __init__(self, nb_trees, max_depth, min_entropy, min_elem, max_nb_dim_to_check, max_nb_thresholds):
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.max_nb_dim_to_check = max_nb_dim_to_check
        self.max_nb_thresholds = max_nb_thresholds

    def train(self, inputs, targets, nb_classes, bagging=False):
        self.trees = []
        params = (self.max_nb_dim_to_check, self.max_nb_thresholds)
        max_inputs = inputs.shape[0]
        for i in range(self.nb_trees):
            tree = BinaryDTClassification(nb_classes, self.max_depth, self.min_entropy, self.min_elem)
            if bagging:
                indexes = np.unique(np.random.randint(max_inputs, size=np.random.randint(self.min_elem, max_inputs)))
                tree.train(inputs[indexes], targets[indexes])
            else:
                tree.train(inputs, targets, params)
            self.trees.append(tree)

    def __call__(self, inputs):
        predictions = None
        for tree in self.trees:
            result = tree(inputs)
            if predictions is None:
                predictions = result
            else:
                predictions += result

        return predictions / len(self.trees)