import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.common import ROOT, Metrics, PlotConfig, as_column, plot, read_csv


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_size: float = Field(0.2, gt=0, lt=1)
    random_state: int = 42


def visualize_data(data: pd.DataFrame, x_column: str, y_column: str) -> None:
    plot(
        data[x_column],
        data[y_column],
        PlotConfig(x_label=x_column, y_label=y_column, title=f"{y_column} vs {x_column}", data_label="Data"),
    )


def linear_regression_with_metrics(
    data: pd.DataFrame, x_column: str, y_column: str, split: SplitConfig = SplitConfig()
) -> Metrics:
    X_train, X_test, y_train, y_test = train_test_split(
        as_column(data[x_column]), data[y_column].to_numpy(), **split.model_dump()
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = Metrics.from_predictions(y_test, y_pred)
    metrics.report()

    plot(
        X_test,
        y_test,
        PlotConfig(x_label=x_column, y_label=y_column, title="Linear Regression Fit"),
        fit=(X_test, y_pred),
    )
    return metrics


if __name__ == "__main__":
    data = read_csv(ROOT / "data.csv", required_columns=("Feature", "Target"))
    print("Data Loaded Successfully!")
    visualize_data(data, x_column="Feature", y_column="Target")
    linear_regression_with_metrics(data, x_column="Feature", y_column="Target")
