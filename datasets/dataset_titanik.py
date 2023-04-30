import pandas as pd

class Titanic():
    def __init__(self,path2train,path2test):
        self.train_set_csv = pd.read_csv(path2train)
        self.test_set_csv = pd.read_csv(path2test)

    def __call__(self):
        return {'train_input':self.train_set_csv.values[:,3:],
                'train_target': 2 * self.train_set_csv.values[:,2] - 1,
                'test_input': self.test_set_csv.values[:, 3:],
                'test_target': 2 * self.test_set_csv.values[:, 2] - 1
                }



