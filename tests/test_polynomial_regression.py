import pytest
import numpy as np
import pandas as pd
from src.polynomial_regression import (
    read_csv,
    train_polynomial_regression,
    predict_prices,
    evaluate_model,
)


def test_read_csv(tmp_path):
    """
    Test if the read_csv function correctly loads data.
    """
    # Create a temporary CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("Size,Price\n1000,150000\n2000,300000")

    # Load the data
    data = read_csv(csv_file)
    assert not data.empty, "CSV file should not be empty."
    assert list(data.columns) == ["Size", "Price"], "CSV file must have 'Size' and 'Price' columns."




def test_train_polynomial_regression():
    """
    Test if the train_polynomial_regression function runs without errors.
    """
    X = np.array([1000, 1200, 1500])
    y = np.array([150000, 170000, 200000])

    model, poly = train_polynomial_regression(X, y, degree=2)
    assert model is not None, "Model should be trained successfully."
    assert poly is not None, "PolynomialFeatures should be created successfully."


def test_predict_prices():
    """
    Test if predictions are correctly generated.
    """
    X = np.array([1000, 1200, 1500])
    y = np.array([150000, 170000, 200000])

    model, poly = train_polynomial_regression(X, y, degree=2)
    predictions = predict_prices(model, poly, X)

    assert len(predictions) == len(X), "Predictions should match the input size."


def test_evaluate_model():
    """
    Test the evaluate_model function for correct outputs.
    """
    y_true = np.array([150000, 170000, 200000])
    y_pred = np.array([151000, 169000, 201000])

    metrics = evaluate_model(y_true, y_pred)
    assert "MAE" in metrics, "MAE should be in evaluation metrics."
    assert "MSE" in metrics, "MSE should be in evaluation metrics."
    assert "RMSE" in metrics, "RMSE should be in evaluation metrics."


if __name__ == "__main__":
    pytest.main()
