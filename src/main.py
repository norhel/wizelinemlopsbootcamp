import os
import yaml
from preprocessing import load_and_split_data
from train import train_model
from evaluation import evaluate_model
#import mlflow

#data_path = "data/forestfires.csv"


# Main function
def main():
    with open("src/params.yaml", "r") as f:

        params = yaml.safe_load(f)    
    # Step 1: Loading and spliting data
    data_path = params["data"]["path"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    n_estimators = params["model"]["n_estimators"]

    #mlflow.set_tracking_uri("http://localhost:5000")
    #mlflow.set_experiment("Burned Area Estimator v1")
    #with mlflow.start_run():
    #    mlflow.log_param("data_path", data_path)
    #    mlflow.log_param("test_size", test_size)
    #    mlflow.log_param("random_state", random_state)
    #    mlflow.autolog()



    X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size, random_state)

    # Step 2: Training Model
    model = train_model(X_train, y_train, n_estimators, model_path = "models/base_model.pkl")

    # Step 3: Evaluating Model
    mae, mse, rmse, y_pred = evaluate_model(model, X_test, y_test)
    print("Predicted Area:", y_pred)
    #print("Mean Absolute Error:", mae)
    #print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    


# Executing main function
if __name__ == "__main__":
    main()
