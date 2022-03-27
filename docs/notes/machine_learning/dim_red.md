# Dimensionality Reduction

- Similar to feature selection in that it reduces the complexity of features being learned
- different from feature selection in that it finds a smaller set of new features, each being a combination of input features, that contain the same information as the input features (compare to only keeping the most relevant features from the input features like in feature selection)

## Principal Components Analysis

- Can model data $x \in \mathbb{R}^n$ as approximately lying in some $k$-dimension subspace where $k << d$
- Prior to running PCA, we preprocess the data by normalizing each feature to have mean 0 and variance 1
    - can be ommmitted if we know different features are on the same scale (e.g. pixels in a grayscale img)
- Want to compute the *major axis of variation* (i.e. the direction on which the data approximately lies)
    - find the direction of maximum variance
- For a given unit vector $u$ and a point $x$, the length of the porjection of $x$ onto. $u$ Is given by their dot product (e.g. $x^T u$).
    - $$\textit{proj}_u(x)$$
    - if $x$ is a point in our dataset, then $x^Tu$ is a distance from the origin
- we want to maximize:

$$\begin{align}\frac{1}{n} \sum_{i=1}^n (x^T u)^2 &= \frac{1}{n}\sum_{i=1}^n (x^T u)^T (x^T u) \\ &= \frac{1}{n}\sum_{i=1}^n u^T x x^T u \\ &= u^T (\frac{ 1}{n} \sum_{i=1}^n x x^T)u \\ &= u^T \Sigma u \end{align}$$

- subject to $u^T u = 1$ (e.g. $||u||_2 = 1$)
    - gaurantees all basis vectors are orthogonal
- Def of variance: $\frac{ 1}{n} \sum_{i=1}^n x x^T = \sigma^2$
- We can define the lagragian of this optimization problem as 
- $$\mathcal{L}(u, \lambda) = u^T \Sigma u - \lambda (u^T u - 1)$$
- $\Sigma \in \mathbb{R}^{n \times n}$, $u \in \mathbb{R}^n$
- $$\nabla_u \mathcal{L}( \cdot) = \Sigma u - \lambda u = 0$$ yields $\Sigma u = \lambda u$
    - n solutions every time
    - eigenvalues are the solutions $\lambda_1, \cdots, \lambda_n$ to $\det (\lambda - \Sigma) = 0$
- principal eigenvector take the largest $|\lambda_i|$ and find $u_1$ s.t. $\Sigma u_1 = \lambda_1 u_1$
    - repeat to find the 2nd, 3rd, etc. principal eigenvectors
-  $z$ is the new data poiint in reduced space:
    - $z = (x^T u_1, \cdots, x^T u_k)^T$ where $k << n
    - to cast z back to original space $x_{approx} = (z_1 u_1) + \cdots + z_k u_k = $ u_k z where each $u_i \in \mathbb{R}^n$

