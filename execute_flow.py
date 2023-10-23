from componentsAPIs.API import ExecuteFlowAPI

data = {
    'test_size': 0.3,
    'seed': 42, 
    'percentage_of_nulls': 0.15,
    'cat_method': 'most_frequent',
    'num_method': 'mean',
    'method': 'ordinal',
    'model': 'Random Forest',
    'model_parameters': {
        'n_estimators': 200,
        'max_depth': 4
    }

}

results = ExecuteFlowAPI.post(data)
print(results)