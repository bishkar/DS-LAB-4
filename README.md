# Practical Labs 4
# Supervised Learning - Regression

## Setup

Before performing the task, you need to set up the development environment.

### Creating a Python virtual environment

1. For Windows:
    ```powershell
    python -m venv venv
    ```

2. For Linux/macOS:
    ```bash
    python3 -m venv venv
    ```

### Activating the virtual environment

1. For Windows:
    ```powershell
    .\venv\Scripts\activate
    ```

2. For Linux/macOS:
    ```bash
    source venv/bin/activate
    ```

> [!NOTE]
> After activating the virtual environment, you will see the `(venv)` prefix in front of the command line.

### Installing the required packages from the `requirements.txt` file

1. For both systems:
    ```bash
    pip install -r requirements.txt
    ```

## Task 1. Linear Regression with CSV Data and Model Evaluation

### Problem Description

You are tasked with creating a Python script that:
1. Implements **Linear Regression** to predict a target variable from features in a CSV file.
2. Evaluates the model using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**.
3. Reads data from a CSV file, ensuring the file isn't empty.
4. Visualizes the relationship between the feature and target variable using **Matplotlib**.

### Example CSV File (`data.csv`)
```csv
Feature,Target
1,2
2,4
3,6
4,8
5,10
```

### Folder Structure
```
project/
│
├── src/
│   ├── linear_regression.py
│
├── data.csv  # Example dataset
```

### Explanation of the Code

#### **`src/linear_regression.py`**
1. **`read_csv(file_path)`**:
   - Reads a CSV file using `pandas`.
   - Ensures the file isn't empty, raising a `ValueError` if it is.

2. **`visualize_data(data, x_column, y_column)`**:
   - Visualizes the scatter plot of the data using **Matplotlib**.

3. **`linear_regression_with_metrics(data, x_column, y_column)`**:
   - Trains a Linear Regression model using `scikit-learn`.
   - Evaluates the model using metrics: MAE, MSE, RMSE.
   - Visualizes the regression line over the test data.

## Task 2. Polynomial Regression for Predicting House Prices

### **Problem Description**
You are tasked with implementing **Polynomial Regression** to predict house prices based on house sizes. Your solution should:  
1. Read a dataset (`house_data.csv`) containing house sizes and prices.  
2. Train a **Polynomial Regression** model to capture non-linear relationships between house size and price.  
3. Evaluate the model using metrics:  
   - **Mean Absolute Error (MAE)**  
   - **Mean Squared Error (MSE)**  
   - **Root Mean Squared Error (RMSE)**  
4. Visualize the results using **Matplotlib**, plotting the original data points and the regression curve.  

### Example Input Dataset: `house_data.csv`
```csv
Size,Price
1000,150000
1200,170000
1500,200000
1800,250000
2000,300000
```

### Folder Structure
```
project/
│
├── src/
│   ├── polynomial_regression.py
│
├── house_data.csv  # Dataset containing house sizes and prices
```

### Explanation of the Code

#### **`src/polynomial_regression.py`**
1. **`read_csv(file_path)`**:
   - Reads the house dataset from CSV using `pandas`.
   - Validates the data format and presence of required columns.
   - Returns the data as a pandas DataFrame.

2. **`train_polynomial_regression(X, y, degree)`**:
   - Trains a polynomial regression model using `LinearRegression`.
   - Takes the degree of polynomial as a parameter.
   - Returns the trained model and polynomial feature transformer.

3. **`predict_prices(model, poly, X)`**:
   - Makes predictions using the trained polynomial regression model.
   - Transforms input features using polynomial features.
   - Returns predicted prices.

4. **`evaluate_model(y_true, y_pred)`**:
   - Calculates and returns model evaluation metrics:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)

5. **`visualize_data(X, y, model, poly)`**:
   - Creates a scatter plot of original data points.
   - Plots the polynomial regression curve.
   - Adds labels, title, and legend for better visualization.
