from category_encoders.ordinal import OrdinalEncoder

class Encoding:

    def __init__(self, method):
        self.method = method
    
    def fit(self, X, y=None):
        if self.method == 'ordinal':
            self.encoder = OrdinalEncoder(return_df=True)
        else:
            raise Exception('Not a valid method')
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)