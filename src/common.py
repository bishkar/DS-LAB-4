from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent


class Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    MAE: float = Field(ge=0)
    MSE: float = Field(ge=0)

    @computed_field
    @property
    def RMSE(self) -> float:
        return float(np.sqrt(self.MSE))

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "Metrics":
        return cls(MAE=mean_absolute_error(y_true, y_pred), MSE=mean_squared_error(y_true, y_pred))

    def report(self) -> None:
        print("Model Evaluation Metrics:")
        for name, value in self.model_dump().items():
            print(f"{name}: {value:.2f}")


class PlotConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_label: str
    y_label: str
    title: str
    data_label: str = "Actual data"
    fit_label: str = "Regression line"


def as_column(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1, 1)


def read_csv(file_path, required_columns: Tuple[str, ...] = ()) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError("The CSV file is empty.")
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def plot(x, y, config: PlotConfig, fit: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    plt.scatter(x, y, color="blue", label=config.data_label)
    if fit is not None:
        fit_x, fit_y = map(np.ravel, fit)
        order = np.argsort(fit_x)
        plt.plot(fit_x[order], fit_y[order], color="red", linewidth=2, label=config.fit_label)
    plt.xlabel(config.x_label)
    plt.ylabel(config.y_label)
    plt.title(config.title)
    plt.legend()
    plt.show()
    plt.close()
