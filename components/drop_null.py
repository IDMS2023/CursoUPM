class DropNull:
    def __init__(self, percentage_of_nulls=0.15):
        self.percentage_of_nulls = float(percentage_of_nulls)

    def fit(self, X, y):
        missings_dict = dict(X.isna().sum() / len(X.index))
        self.missings_variables_list = [
            variable
            for variable, missings_percentage in missings_dict.items()
            if float(missings_percentage) > self.percentage_of_nulls
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.missings_variables_list)
