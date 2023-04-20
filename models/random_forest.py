from models.decision_tree import BinaryDTClassification

class RandomForestClassification():

    def __init__(self, nb_trees, max_depth, min_entropy, min_elem, max_nb_dim_to_check, max_nb_thresholds):
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.max_nb_dim_to_check = max_nb_dim_to_check
        self.max_nb_thresholds = max_nb_thresholds

    def train(self, inputs, targets, nb_classes):
        self.trees = []
        params = (self.max_nb_dim_to_check, self.max_nb_thresholds)
        #todo: распараллелить
        for i in range(self.nb_trees):
            tree = BinaryDTClassification(nb_classes, self.max_depth, self.min_entropy, self.min_elem)
            tree.train(inputs, targets, params)
            self.trees.append(tree)

    #todo: распараллелить
    def get_prediction(self, inputs):
        predictions = None
        for tree in self.trees:
            result = tree(inputs)
            if predictions is None:
                predictions = result
            else:
                predictions += result

        return predictions / len(self.trees)