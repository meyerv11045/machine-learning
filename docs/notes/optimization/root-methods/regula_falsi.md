# Regula Falsi

- Known as the method of false position 
- Only uses $g$ 
- Bracketing method like Bisection to guaranteed to converge

## Method

1. Construct a secant line using the the two interval endpoints

    - Approximates $g$ 
    - $l_k(x) = g(a_k) + \frac{g(b_k) - g(a_k)}{b_k - a_k}(x - a_k)$

2. Find the root of the line

    - $c_k = \frac{a_k g(b_k) - b_k g(a_k)}{g(b_k) - g(a_k)}$

3. Determine the next subinterval $[a_{k+1}, b_{k+1}]$:

    - If $g(a_k) g(c_k) < 0$ then $a_{k+1} = a_k$ and $b_{k+1} = c_k$

    - If $g(b_k) g(c_k) < 0$ then $a_{k+1} = c_k$ and $b_{k+1} = b_k$

## Convergence

- Q-Linear convergence
- $c$ depends on the curvature of $g$ and how close $a_0$ is to $x^*$
    - Since we are using a linear approximation of $g$, less curvature = faster convergence

## Pros

- Reliable given the interval encloses $x^*$

## Cons

- Slow convergence with a constant that depends on the curvature of $g$ 