from components.reader import Reader
from components.train_test import TrainTest
from components.drop_null import DropNull
from components.missings import Missings
from components.encoding import Encoding
from components.train import Train

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

class API:
    def post(request):
        X, y = Reader.upload_data()
        train_test = TrainTest(request['test_size'], request['seed'])
        X_train, X_test, y_train, y_test = train_test.split(X, y)
        drop_null = DropNull(request['percentage_of_nulls'])
        missings = Missings(request['cat_method'], request['num_method'])
        encoding = Encoding(request['method'])
        train = Train(request['model'], request['model_parameters'])

        pipeline = Pipeline([
            ('drop_null', drop_null),
            ('missings', missings),
            ('encoding', encoding),
            ('model', train)
        ])

        pipeline.fit(X_train, y_train)

        y_pred, y_prob = pipeline.transform(X_test)
        
        metrics = {
            'AUC': roc_auc_score(y_test, y_prob),
            'F1 Score': f1_score(y_test, y_pred)
        }
        return metrics


