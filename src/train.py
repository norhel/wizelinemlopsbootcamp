import joblib
from sklearn.ensemble import RandomForestRegressor

#Training Model
def train_model(X_train, y_train, n_estimators, model_path):
    # Using a Random Forest regression model
    model = RandomForestRegressor(n_estimators = n_estimators)

    # Training the model using training data
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    return model
