from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Missings:

    def __init__(self, cat_method, num_method):
        self.cat_method = cat_method
        self.num_method = num_method

    def fit(self, X, y=None):
        numerical_imputer = SimpleImputer(strategy=self.num_method)
        categorical_imputer = SimpleImputer(strategy=self.cat_method)
        self.imputer = Pipeline([
            ('numerical', numerical_imputer),
            ('categorical', categorical_imputer)
        ])
        self.imputer.fit(X, y)
        return self
    
    def transform(self, X):
        return self.imputer.transform(X)
    