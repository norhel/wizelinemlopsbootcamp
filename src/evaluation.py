from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

# Evaluating Model
def evaluate_model(model, X_test, y_test):
    # Predicting burned areas
    y_pred = model.predict(X_test)

    # Mean Squared Error and R-squared score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)


    return mae, mse, rmse, y_pred
