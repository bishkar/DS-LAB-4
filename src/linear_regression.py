import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns its content as a DataFrame.
    Raises ValueError if the file is empty.
    """
    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError("The CSV file is empty.")
    return data


def visualize_data(data: pd.DataFrame, x_column: str, y_column: str):
    """
    Visualizes the relationship between the feature and the target variable.
    """

    plt.show()


def linear_regression_with_metrics(data: pd.DataFrame, x_column: str, y_column: str):
    """
    Performs linear regression, evaluates metrics, and visualizes the regression line.
    """
    # Prepare the data


    # Split the data


    # Train the model


    # Make predictions


    # Evaluate the model


    print(f"Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Visualize the regression line
    plt.scatter(X_test, y_test, color='blue', label='Actual data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    csv_path = "../data.csv"  # Path to the dataset
    data = read_csv(csv_path)
    print("Data Loaded Successfully!")

    # Visualize the data
    visualize_data(data, x_column="Feature", y_column="Target")

    # Run Linear Regression and Evaluate Metrics
    linear_regression_with_metrics(data, x_column="Feature", y_column="Target")
