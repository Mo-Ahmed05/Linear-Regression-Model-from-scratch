# Linear Regression - Non-Linear Regression "from Scratch"

This project implements a Linear Regression model algorithms, logic and math from **scratch** (without using machine learning libraries like Scikit-learn for the core algorithm) to predict sales based on advertising budgets for TV, Radio, and Newspaper. It explores Advertising Sales Prediction with Linear Regression - Non-Linear Regression relationships and includes regularization techniques (Lasso and Ridge) to improve model performance.


## Model Features

- **LinearRegression Class**: A custom Python class Regression that supports:

    - Gradient Descent Algorithm.

    - Custom number of **Iteration** and **Learning Rate**.

    - Custom **Epsilon** value (very tiny number) that stops the learning process if the change of the cost function is smaller than Epsilon.

    - If the model overshoot it sends a warning massage and stops the process.

    - **Lasso** and **Ridge** regularization, with custom **lambda** value.

    - **Min-Max normalization** for feature scaling.

    - Option for rescaling the **coefficients** (weights - bias term) after the learing process to work on the real world values.

    View the full code: [Linear Regression class](linear_regression.py)

- **Model Statistics**: Helper functions for calculating $R^2$, MAE, RMSE, and correlation.

    View the full code: [Model Statistics](model_statistics.py)


## Dataset
The dataset used is advertising.csv, which contains the following columns:

- **TV**: Advertising budget for TV.

- **Radio**: Advertising budget for Radio.

- **Newspaper**: Advertising budget for Newspaper.

- **Sales**: Sales of the product.


## Model Testing

- **Linear Modeling**: with visualizing the features and results.
    
    View the Notebook: [Linear Model](linear_model.py)
- **Non-Linear Modeling**: Enhancement of the linear model by introducing interaction terms (e.g., TV * Radio) and polynomial features (e.g., TV^2).

    View the Notebook: [None-Linear Model](non_linear_model.py)