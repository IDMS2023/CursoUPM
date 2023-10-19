from sklearn.datasets import load_breast_cancer


class Reader:

    def upload_data():
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        return X, y
