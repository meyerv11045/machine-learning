
# Newton's Method

- Finds the root of a function $g$ 
    - i.e. finds $x$ such that $g(x) = 0$
- Uses $g$ and $g'$ 
    - In the context of optimization where $g = f'$, we would need first and second derivatives of the cost function $f$ 

## Method

Given $x_k$  we will approximate $g(x)$ by the tangent line $l_k(x)$ at $x_k$. This tangent line is the 1st order taylor polynomial of $g$ at $x_k$ and represents a linear approximation of $g$.

$$ l_k(k) = g(x_k) + g'(x_k)(x - x_k)$$

If we let $x_{k+1}$ solve $l_k(x) = 0$ we get the equation for the next step in the iterative process of finding the root:

$$x_{k + 1} = x_k - \frac{g(x_k)}{g'(x_k)}$$

## Convergence 

- Q-quadratic convergence 
- The constant $C > \frac{g''(x^*)}{2g'(x^*)}$
    - This tells us that that curvature of $g$ around $x^*$ in a scale invariant way 
    - Less curvature -> smaller $C$ -> faster convergence.

## Pros

- Very fast 

## Cons

- May diverge or converge to something unexpected
- Can get misdirected when $g'$ is small (i.e. $g$ is flat)
    - when the cost function $f$ doesn't change very rapidly, $g$ will be flat and newton's method will not perform well
    - due to  $g'(x_k) \neq 0$ in the denominator of the update step 
- Requires $g'$ which is the second derivative of the cost function in an optimization context, so this can be unfeasible for certain problems or computationally expensive to compute 

## Other Notes

Since we are using a *linear* approximation of $g$ to find its roots, the actual curvature of $g$ affects how fast Newton's method converges. For low curvature, the method is very fast. 



