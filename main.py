import numpy as np
from models.adaboost import Adaboost

if __name__ == "__main__":
    from datasets.dataset_titanik import Titanic
    from utils.metrics import recall, precision, f1_score, confusion_matrix
    dataset = Titanic("./datasets/titanik_train_data.csv", "./datasets/titanik_test_data.csv")()
    adaBoost = Adaboost(15)
    adaBoost.train(dataset['train_input'], dataset['train_target'])

    predictions = adaBoost(dataset['test_input'])
    targets = dataset['test_target']

    m_recall = recall(predictions, targets)
    m_precision = precision(predictions, targets)
    m_f1_score = f1_score(m_precision, m_recall)

    def map_for_conf_matrix(array):
        a = np.ones(shape=array.shape, dtype='int')
        mask = array == -1.0
        a[mask] = 0
        return a


    conf_matrix = confusion_matrix(2, map_for_conf_matrix(predictions), map_for_conf_matrix(targets))

    print(f'CONFUSION MATRIX:\n{conf_matrix}')
    print(f'RECALL: {m_recall}')
    print(f'PRECISION: {m_precision}')
    print(f'F1 SCORE: {m_f1_score}')