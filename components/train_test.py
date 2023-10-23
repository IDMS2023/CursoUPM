from sklearn.model_selection import train_test_split


class TrainTest:
    def __init__(self, test_size=0.3, seed=42):
        self.test_size = test_size
        self.seed = seed

    def split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )
        return X_train, X_test, y_train, y_test
