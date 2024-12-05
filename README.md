# SalesPrediction-LinearRegression
This repository demonstrates linear regression for sales prediction based on TV advertising spend. It includes simple and multiple regression models, evaluation metrics (MSE, RMSE, R²), and cross-validation. The project also visualizes model performance and shows sales predictions based on different ad spends.

<img src="https://github.com/akay35/image/blob/main/1.jpg" alt="This is my world" width="580" height="500"/>   

# Advertising Sales Prediction with Linear Regression

This project demonstrates the use of linear regression to predict sales based on advertising expenditures across TV, Radio, and Newspaper channels. It includes both simple and multiple linear regression models, model evaluation with metrics like MSE, RMSE, and R-squared, and performance testing using training/test splits and cross-validation.

## Methodology

### 1. Simple Linear Regression

Using **scikit-learn's LinearRegression** method, the model is trained to predict **sales** based on **TV advertising expenditure** as the independent variable. The prediction formula is:

\[
y = b + w * x
\]

Where:
- \( y \) = predicted sales
- \( b \) = intercept
- \( w \) = weight (coefficient) for TV spend
- \( x \) = TV advertising expenditure

### 2. Model Evaluation

- **Metrics Used**:
  - **MSE (Mean Squared Error)**: Measures the average squared difference between actual and predicted values.
  - **MAE (Mean Absolute Error)**: Measures the average absolute differences between actual and predicted values.
  - **RMSE (Root Mean Squared Error)**: Takes the square root of MSE to bring error metrics back to the original scale of the dependent variable.
  - **R-squared**: Indicates the percentage of variance in the dependent variable explained by the independent variable(s).

### 3. Multiple Linear Regression

In this model, **TV**, **Radio**, and **Newspaper** advertising spend are treated as independent variables. The weights (coefficients) for each variable are calculated to predict sales.

### 4. Model Performance

- The dataset is split into **80% training** and **20% testing** sets.
- **Cross-validation** is performed using **10-fold** to evaluate the model’s generalizability.

### 5. Results

- The **R-squared** value indicates that TV advertising explains **61%** of the variation in sales.
- **Model Performance**:
  - **RMSE** on training set: 1.73
  - **RMSE** on test set: 1.41
  - **Cross-validation RMSE**: 1.69

These results suggest the model performs well, with no overfitting.

## Conclusion

This project effectively demonstrates how advertising expenditures (specifically on TV) can predict sales. The linear regression models have shown good predictive accuracy and can be further refined for better performance.


