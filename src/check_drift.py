import pandas as pd
from sklearn.metrics import root_mean_squared_error
import joblib
import mlflow



def check_for_drift(new_data_path, model_path, threshold=0.1):
    new_data = pd.read_csv(new_data_path)
    new_X = new_data[['DC', 'DMC', 'FFMC']]
    new_y = new_data["area"]

    model = joblib.load(model_path)
    predictions = model.predict(new_X)
    new_rmse = root_mean_squared_error(new_y, predictions)

    experiment_name = "Burned Area Estimator v1"
    best_run_df = mlflow.search_runs(order_by=['metrics.training_root_mean_squared_error ASC'], max_results=1)
    if len(best_run_df.index) == 0:
        raise Exception(f"Found no runs for experiment '{experiment_name}'")

    best_run = mlflow.get_run(best_run_df.at[0, 'run_id'])
    best_model_uri = f"{best_run.info.artifact_uri}/model"
    # best_model = mlflow.sklearn.load_model(best_model_uri)

    # print best run info
    #print("Best run info:")
    #print(f"Run id: {best_run.info.run_id}")
    #print(f"Run parameters: {best_run.data.params}")
    #print("Run score: RMSE = {:.4f}".format(best_run.data.metrics['training_root_mean_squared_error']))
    #print(f"Run model URI: {best_model_uri}")
    
    old_rmse = best_run.data.metrics['training_root_mean_squared_error']

    drift = abs(new_rmse - old_rmse) / old_rmse
    return drift > threshold

