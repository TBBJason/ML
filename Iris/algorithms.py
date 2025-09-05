from sklearn.model_selection import train_test_split


class Validation:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset[:, 0:4]
        self.Y = dataset[:, 4]

    def harness(self, percent=0.2):
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X, self.Y, test_size=percent, random_state=1)
    
