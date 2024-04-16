from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluating Model
def evaluate_model(model, X_test, y_test):
    # Predicting burned areas
    y_pred = model.predict(X_test)

    # Mean Squared Error and R-squared score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2, y_pred
