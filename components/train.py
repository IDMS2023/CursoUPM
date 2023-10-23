from sklearn.ensemble import RandomForestClassifier

class Train:

    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters

    def fit(self, X, y):
        if self.model == 'Random Forest':
            self.clasiffier = RandomForestClassifier(**self.parameters)
        else:
            raise Exception('Not a valid Model')
        self.clasiffier.fit(X, y)
        return self
    
    def transform(self, X):
        predictions = self.clasiffier.predict(X)
        probabilities = self.clasiffier.predict_proba(X)[:, 1]
        return predictions, probabilities