# Importing libraries
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


#Loading and spliting data
def load_and_split_data(data_path, test_size, random_state):
    # Loading data
    df = pd.read_csv(data_path)
    df.drop_duplicates()

    # Selected features (DC, DMC, FFMC) and target variable (area)
    X = df[['DC', 'DMC', 'FFMC']]  # Selected features
    y = df['area']  # Target variable

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test

