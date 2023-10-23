from category_encoders.ordinal import OrdinalEncoder


class Encoding:
    def __init__(self, method="ordinal"):
        self.method = method

    def fit(self, X, y):
        if self.method == "ordinal":
            self.encoder = OrdinalEncoder(return_df=True)
            self.encoder.fit(X, y)
        else:
            raise Exception("Not available encoder")
        return self

    def transform(self, X):
        return self.encoder.transform(X)
