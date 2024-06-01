import pytest
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import os
import yaml

with open("../src/params.yaml", "r") as f:

    params = yaml.safe_load(f)   
data_path = params["data"]["path"]
new_data_path = params["data"]["new_path"]
test_size = params["data"]["test_size"]
random_state = params["data"]["random_state"]
n_estimators = params["model"]["n_estimators"]

@pytest.fixture
def load_data():
    data = pd.read_csv(data_path)
    return data

@pytest.fixture
def load_model():
    model = joblib.load("models/base_model.pkl")
    return model

def test_data_shape(load_data):
    data = load_data
    assert data.shape[0] > 0, "Data should have rows"
    assert data.shape[1] > 1, "Data should have features and target column"

def test_model_prediction(load_model, load_data):
    model = load_model
    data = load_data

    X = data[['DC', 'DMC', 'FFMC']]
    y = data["area"]

    predictions = model.predict(X)
    assert len(predictions) == len(y), "Number of predictions should match number of observations"

def test_model_performance(load_model, load_data):
    model = load_model
    data = load_data

    X = data[['DC', 'DMC', 'FFMC']]
    y = data["area"]

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    # Define a threshold for acceptable model performance
    threshold_mse = 100
    assert mse < threshold_mse, f"Model MSE should be less than {threshold_mse}, but got {mse}"
