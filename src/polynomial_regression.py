from functools import partial
from typing import Dict, Tuple

import numpy as np
from pydantic import ConfigDict, PositiveInt, validate_call
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src import common
from src.common import ROOT, Metrics, PlotConfig, as_column, plot

SIZE, PRICE = "Size", "Price"

read_csv = partial(common.read_csv, required_columns=(SIZE, PRICE))


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_polynomial_regression(
    X: np.ndarray, y: np.ndarray, degree: PositiveInt
) -> Tuple[LinearRegression, PolynomialFeatures]:
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(as_column(X)), y)
    return model, poly


def predict_prices(model: LinearRegression, poly: PolynomialFeatures, X: np.ndarray) -> np.ndarray:
    return model.predict(poly.transform(as_column(X)))


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return Metrics.from_predictions(y_true, y_pred).model_dump()


def visualize_data(X: np.ndarray, y: np.ndarray, model: LinearRegression, poly: PolynomialFeatures) -> None:
    X_plot = np.linspace(X.min(), X.max(), 100)
    plot(
        X,
        y,
        PlotConfig(
            x_label="House Size",
            y_label="House Price",
            title="Polynomial Regression: House Size vs Price",
            data_label="Actual Prices",
            fit_label="Polynomial Regression Fit",
        ),
        fit=(X_plot, predict_prices(model, poly, X_plot)),
    )


if __name__ == "__main__":
    data = read_csv(ROOT / "house_data.csv")
    X, y = data[SIZE].to_numpy(), data[PRICE].to_numpy()

    model, poly = train_polynomial_regression(X, y, degree=3)
    Metrics(**evaluate_model(y, predict_prices(model, poly, X))).report()
    visualize_data(X, y, model, poly)
