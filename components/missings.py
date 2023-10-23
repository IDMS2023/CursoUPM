from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Missings:
    def __init__(self, categorical_method="most_frequent", numerical_method="mean"):
        self.categorical_method = categorical_method
        self.numerical_method = numerical_method

    def fit(self, X, y):
        numerical_imputer = SimpleImputer(strategy=self.numerical_method)
        categorical_imputer = SimpleImputer(strategy=self.categorical_method)
        self.imputer = Pipeline(
            [("numerical", numerical_imputer), ("categorical", categorical_imputer)]
        )
        self.imputer.fit(X, y)
        return self

    def transform(self, X):
        return self.imputer.transform(X)
