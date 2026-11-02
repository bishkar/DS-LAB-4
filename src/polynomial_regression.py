import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns its content as a pandas DataFrame.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError("The CSV file is empty.")
    return data


def train_polynomial_regression(X: np.array, y: np.array, degree: int):
    """
    Trains a polynomial regression model.

    Parameters:
    X (np.array): Independent variable (e.g., house sizes).
    y (np.array): Dependent variable (e.g., house prices).
    degree (int): Degree of the polynomial.

    Returns:
    tuple: Trained model and polynomial feature transformer.
    """

    return model, poly


def predict_prices(model: LinearRegression, poly: PolynomialFeatures, X: np.array):
    """
    Predicts prices using the trained polynomial regression model.

    Parameters:
    model (LinearRegression): Trained polynomial regression model.
    poly (PolynomialFeatures): Polynomial feature transformer.
    X (np.array): Independent variable for prediction.

    Returns:
    np.array: Predicted prices.
    """

    return model.predict(X_poly)


def evaluate_model(y_true: np.array, y_pred: np.array):
    """
    Evaluates the model using various metrics.

    Parameters:
    y_true (np.array): Actual prices.
    y_pred (np.array): Predicted prices.

    Returns:
    dict: Evaluation metrics.
    """

    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def visualize_data(X: np.array, y: np.array, model: LinearRegression, poly: PolynomialFeatures):
    """
    Visualizes the original data and the regression curve.

    Parameters:
    X (np.array): Independent variable (e.g., house sizes).
    y (np.array): Dependent variable (e.g., house prices).
    model (LinearRegression): Trained polynomial regression model.
    poly (PolynomialFeatures): Polynomial feature transformer.
    """
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_plot = predict_prices(model, poly, X_plot)

    plt.scatter(X, y, color="blue", label="Actual Prices")
    plt.plot(X_plot, y_plot, color="red", label="Polynomial Regression Fit")
    plt.xlabel("House Size")
    plt.ylabel("House Price")
    plt.title("Polynomial Regression: House Size vs Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Step 1: Load data
    file_path = "../house_data.csv"  # Path to dataset
    data = read_csv(file_path)

    # Step 2: Extract features and target
    X = data["Size"].values
    y = data["Price"].values

    # Step 3: Train polynomial regression model
    degree = 3
    model, poly = train_polynomial_regression(X, y, degree)

    # Step 4: Predict prices
    predictions = predict_prices(model, poly, X)

    # Step 5: Evaluate model
    metrics = evaluate_model(y, predictions)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    # Step 6: Visualize results
    visualize_data(X, y, model, poly)
