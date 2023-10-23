from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from components.reader import Reader
from components.train_test import TrainTest
from components.drop_null import DropNull
from components.missings import Missings
from components.encoding import Encoding
from components.train import Train


class ExecuteFlowAPI:
    def post(request):
        # Retrive data
        X, y = Reader.upload_data()
        # Start ml process
        # Divide train test
        train_test = TrainTest(test_size=request["test_size"], seed=request["seed"])
        X_train, X_test, y_train, y_test = train_test.split(X, y)
        # Start pipeline
        # Drop nulls
        stages = []
        stages.append(("dropnull", DropNull(request["percentage_of_nulls"])))
        # Missings
        stages.append(
            (
                "missings",
                Missings(
                    categorical_method=request["cat_method"],
                    numerical_method=request["num_method"],
                ),
            )
        )
        # Encoding
        stages.append(("encoding", Encoding(request["method"])))
        # Model training
        stages.append(("model", Train(request["model"], request["model_parameters"])))
        # Compile pipeline
        pipeline = Pipeline(stages)
        pipeline.fit(X_train, y_train)
        # Obtain predictions
        probabilities, predictions = pipeline.transform(X_test)
        # Get metrics
        metrics_dict = {
            "f1": f1_score(y_test, predictions),
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, probabilities),
        }
        return metrics_dict
