# Bisection Method

- Uses only $g$ 
    - Uses only $g = f'$ where $f$ is an objective function in an optimization problem
- Roughly equivalent to binary search over a continuous domain instead of over a discerete list 

## Method

1. Choose a starting interval $[a_0, b_0]$ such that $g(a_0) g(b_0) < 0$ (alternatively $x^* \in [a_0, b_0]$ where $g(x^*) = 0$)

    - interval should contain the root
    - different signs at the endpoints of the interval implies by the intermediate value theorem that a root exists in between them due to continuity of $g$

2. Compute $g(c_k)$ where $c_k = \frac{a_k + b_k}{2}$ (i.e. the midpoint of the interval)

3. Determine the next subinterval $[a_{k+1}, b_{k+1}]$:

    - If $g(a_k) g(c_k) < 0$ then $a_{k+1} = a_k$ and $b_{k+1} = c_k$

    - If $g(b_k) g(c_k) < 0$ then $a_{k+1} = c_k$ and $b_{k+1} = b_k$

## Convergence

- Actual error is not gauranteed to go down in any iteration
- Only the bound on error is gauranteed to go down 
- Therefore, we have R-linear convergence with $c = 1/2$ since the bound on the error is cut in half at each iteration 

## Pros

- Requires no derivative information of $g$ (only need $f'$ of objective function)
- Reliable- guaranteed to converge in a predictable way given an initial interval that contains a root 
- Can calculate the min number of iterations required to guarantee an error less than $\epsilon > 0$ ([Link](https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/bisection/))

## Cons

- Slow