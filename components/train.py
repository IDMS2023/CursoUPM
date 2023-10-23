from sklearn.ensemble import RandomForestClassifier


class Train:
    def __init__(self, model="Random Forest", parameters={}):
        self.model = model
        self.parameters = parameters

    def fit(self, X, y):
        if self.model == "Random Forest":
            self.classifier = RandomForestClassifier(**self.parameters)
            self.classifier.fit(X, y)
        else:
            raise Exception("Not valid model")
        return self

    def transform(self, X):
        probabilities = self.classifier.predict_proba(X)[:, 1]
        predictions = self.classifier.predict(X)
        return probabilities, predictions
