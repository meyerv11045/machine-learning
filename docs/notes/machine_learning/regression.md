# Regression

## Simple Linear Regression

### Probabilistic Interpretation

## Multiple Linear Regression

## Polynomial Regression

### Scikit Learn

- Provides two approaches

1. `LinearRegression` object- uses ordinary least squares solver from scipy to compite the closed form solution 
    - If enough memory for the matrices and inversions, this method is faster and easier
2. `SGDRegressor` object- generic implementation of stochastic gradient descent so must set
    - `loss` to `L2` for linear regression
    - `penality` to `none` for linear regression or `L2` for ridge regression (this is the regularization mode)
    - behaves better if loss function can be decomposed into additive terms

## Locally Weighted Linear Regression

- For a given query point $x \in \mathbb{R}^n$, we compute the weights of all other training points and use these weights to compute the optimal parameters $\theta \in \mathbb{R}^n$. 
- Each training sample has a weight $w^{(i)} = \exp (-\frac{(x^{(i)} - x)^2}{2 \tau^2})$ 
- The weights for the each training sample are placed along the diagonal of a matrix of zeros $W \in \mathbb{R}^{m \times m}$ 
- Optimal parameters can be calculated using the closed form equation for ordinary least squares:
    - $$\theta = (X^T W X )^{-1} X^T Wy$$
    - Must be recomputed for each new query point since the weights change for each query point (computationally expensive)
- $h_\theta(x) = \theta^T x = x^T \theta$ where $x \in \mathbb{R}^n$
- Note that the intercept term of $x_0 = 1$ is not used in LWLR
- $\tau$ is the bandwith parameter
    - Increasing $\tau$ gives more equal weighting to all training samples, resulting in more underfitting
    - Decreasing $\tau$ gives more weighting to nearby training samples, resulting in more overfitting

