from preprocessing import load_and_split_data
from train import train_model
from evaluation import evaluate_model

data_path = "wizelinemlopsbootcamp/data/forestfires.csv"


# Main function
def main():
    # Step 1: Loading and spliting data
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)

    # Step 2: Training Model
    model = train_model(X_train, y_train)

    # Step 3: Evaluating Model
    mae, mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print("Predicted Area:", y_pred)
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)


# Executing main function
if __name__ == "__main__":
    main()
