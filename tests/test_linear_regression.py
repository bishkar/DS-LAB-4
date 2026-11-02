import pytest
from src.linear_regression import read_csv, visualize_data, linear_regression_with_metrics


def test_read_csv_function_exists():
    """
    Test if the read_csv function exists in the module.
    """
    assert callable(read_csv), "The read_csv function is not defined."


def test_csv_not_empty(tmp_path):
    """
    Test if the CSV reading function correctly identifies non-empty files.
    """
    # Create a temporary CSV file
    csv_file = tmp_path / "../data.csv"

    # Write valid content to the CSV file
    csv_file.write_text("Feature,Target\n1,2\n3,4")

    # Read the CSV and verify it's not empty
    data = read_csv(csv_file)
    assert not data.empty, "The CSV file should not be empty."



def test_visualize_data_function_exists():
    """
    Test if the visualize_data function exists in the module.
    """
    assert callable(visualize_data), "The visualize_data function is not defined."


def test_visualize_data_no_errors(tmp_path, monkeypatch):
    """
    Test the visualize_data function to ensure it runs without errors.
    """
    # Create a temporary CSV file with valid data
    csv_file = tmp_path / "../data.csv"
    csv_file.write_text("Feature,Target\n1,2\n3,4\n5,6")

    # Read the CSV file
    data = read_csv(csv_file)

    # Mock `plt.show()` to prevent the test from displaying the plot
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    # Ensure visualize_data runs without throwing errors
    try:
        visualize_data(data, x_column="Feature", y_column="Target")
    except Exception as e:
        pytest.fail(f"visualize_data raised an exception: {e}")


def test_linear_regression_with_metrics_no_errors(tmp_path, capsys):
    """
    Test the linear_regression_with_metrics function to ensure it runs and outputs metrics correctly.
    """
    # Create a temporary CSV file with valid data
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("Feature,Target\n1,2\n2,4\n3,6\n4,8\n5,10")

    # Read the CSV file
    data = read_csv(csv_file)

    # Ensure linear_regression_with_metrics runs without throwing errors
    try:
        linear_regression_with_metrics(data, x_column="Feature", y_column="Target")
    except Exception as e:
        pytest.fail(f"linear_regression_with_metrics raised an exception: {e}")

    # Capture the output printed by the function
    captured = capsys.readouterr()
    assert "Model Evaluation Metrics:" in captured.out, "The output should contain evaluation metrics."


if __name__ == "__main__":
    pytest.main()
