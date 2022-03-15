# Secant Method

- Finds the root of a function $g$ 
    - i.e. finds $x$ such that $g(x) = 0$
- Uses only $g$

## Method

1. Use the two most recent points to construct a secant line $l_k(x)$
2. Compute the root of the secant line to find the next point
    - $x_{k+1} = \frac{x_{k-1}g(x_k) - x_kg(x_{k-1})}{g(x_k) - g(x_{k-1})}$
    - equivalent to $x_{k+1} = x_k - \frac{g(x_k)(x_{k-1} - x_k)}{g(x_k) - g(x_{k-1})}$
        - numerically less likely to lose preceision

## Convergence 

- Superlinear convergence with a q-rate of 1.618 (golden ratio)
- Slower than newton's but faster than bisection and regula falls
- The constant $C > \frac{g''(x^*)}{2g'(x^*)}$
    - This tells us that that curvature of $g$ around $x^*$ in a scale invariant way 
    - Less curvature -> smaller $C$ -> faster convergence.

## Pros

- Relatively fast
- Doesn't require $g'$

## Cons

- May diverge or converge to something unexpected

## Other Notes

Since we are using a *linear* approximation of $g$ to find its roots, the actual curvature of $g$ affects how fast Newton's method converges. For low curvature, the method is very fast. 



