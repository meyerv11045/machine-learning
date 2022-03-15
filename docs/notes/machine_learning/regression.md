# Regression

## Simple Linear Regression

## Multiple/Polynomial Linear Regression

### Scikit Learn

- Provides two approaches

1. `LinearRegression` object- uses ordinary least squares solver from scipy to compite the closed form solution 
    - If enough memory for the matrices and inversions, this method is faster and easier
2. `SGDRegressor` object- generic implementation of stochastic gradient descent so must set
    - `loss` to `L2` for linear regression
    - `penality` to `none` for linear regression or `L2` for ridge regression (this is the regularization mode)
    - behaves better if loss function can be decomposed into additive terms



## Probabilistic Linear Regression

## Locally Weighted Linear Regression