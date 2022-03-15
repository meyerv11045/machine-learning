# Convergence Analysis

We are interested in how fast the error goes to zero for a given iterative optimization method where $\lim_{k \to \infty} x_k = x^*$. Define the error as $e_k = ||x_k - x^*||$.

## Q-Rate Convergence

Q-rate tells us how fast a sequence convergences to a value.

$x_k \to x^*$ with q-rate $r$ if there exists a constant $c$  such that for any sufficiently large $k$: 

$$||x_{k+1} - x^*|| \leq c ||x_{k+1} -x_k||^r$$

Also can be represented as $||e_{k+1}|| \leq c||e_k||^r$ for sufficiently large $k$ where $e$ is the error btw the current point and optimal point at a given iteration. We can alternatively write that $||e_{k_1}|| = O(e_k)$ using big-o notation to represent asymptotic growth. 

Taking the negative logarithm of the error term will give us the number of decimal places of accuracy for that given iteration's approximation. For instance, let $e_k = 10^{-6}$, then the number of decimal places of accuracy is $-\log_{10}10^{-6} = 6$. We can take the negative logarithm of both sides of the above equation to get:

$$ -\log_{10} e_{k+1} \geq r (- \log_{10} e_k) - \log_{10} c$$

This expression tells us how fast the number of decimal places of accuracy is increasing.

### Linear

- rate $r = 1$ and $c < 1$ 
- Ex: 1, 1/2 1/4, 1/8, ... -> 0 for $r = 1$ and $c = \frac{1}{2}$ 

### Superlinear

- $r = 1$
- The constant $c$ changes over time: $c_k \to 0$ as $k \to 0$

### Quadratic

-  $r = 2$ 
    - $r$ is large enough that it swamps the effect of $c$ no matter its value
- Ex: $10^{-1}, 10^{-2}, 10^{-4}, 10^{-8}, \cdots \to 0$ for $r= 2$ and $c = 1$

### Q-rate Limit Lemma

Suppose $r \geq 1$ and $L$ exists where ($L < 1$  if $r = 1$)

$$L = \lim_{k \to \infty} \frac{||x_{k+1} - x^* ||}{||x_k - x^* ||^r}$$

Then $x_k \to x^*$ with Q-rate $r$ and a constant $C > L$



## R-Rate Convergence

- Tells us how fast the bounds on the error converges to 0. 
    - The error has a bounds that has a q-rate convergence

- $x_k \to x^*$ with r-rate $r$  if $||e_k || \leq b_k$ for all $k$ where $b_k \to 0$ with q-rate $r$



## Summary

- Linear:
    - moderate to large $c$ is slow
    - small $c$ is fast
- Superlinear: fast
- Quadratic: very fast