# Singular Value Decomposition

- Allows us to decompose a data matrix $X \in \mathbb{R}^{n \times m}$ into matrices $U \Sigma V^T$.
-  $U$ and $V$ are orothonormal matrices (unitary so $UU^T = U^TU = I_{n \times n}$)  
    - Columns of $U$ are "eigen" and give us a basis (that retains the variance of $X$) that we can represent the column vectors in $X$ in
- $\Sigma$ is a diagonal matrix ($\sigma_1 \geq \cdots \geq \sigma_m \geq 0$).
- $X$- each column is a sample
- $U$- left singular vectors
- $V$- right singular vectors. $V^T$ columns give the mixture of the basis columns in $U$ needed to get the corresponding columns in the data matrix $X$ 
- $\Sigma$- singular values (sigmas are ordered by importance)
- Guaranted to exist and be unique 
- Matlab: `[u, s, v] = svd(x);`

## Resources

-  [SVD Overview Video](https://www.youtube.com/watch?v=nbBvuuNVfco)

# Principal Component Analysis

- Common dimensionality reduction technique 

## Background Math Refresher

### Variance & Covariance

- Variance measures the spread of the data
    - variance = sum of (dist from mean)^2 / n
    - $Var(X) = \sigma_X^2 = \frac{\sum_{i=1}^n ( x_i - \mu_X)^2}{n-1}$ (this is sample variance, population variance has $n$)
- Std Deviation is the square root of variance $\sigma_X = \sqrt{\sigma_X^2}$
- For data $x \in \mathbb{R}^n$, you can measure the variance along each dimension by calculating the variance of the components in that dimenson
- However, variance does not show the whole structure of how multi-dimensional data is distributed so we also use covariance to describe multidimensional data
    - $Cov(X,Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$
    - Subtracting by the expected values of $X$ and $Y$ centers the data at $(0,0)$
    - Can be interpreted as taking the avg of the product of coordinates
- We define a covariance matrix $\Sigma = \begin{bmatrix} cov(X,X) & cov(X,Y) \\cov(Y,X) & cov(Y,Y) \\ \end{bmatrix} = \begin{bmatrix} var(X) & cov(X,Y) \\cov(Y,X) & var(Y) \\ \end{bmatrix} $
    - Symmetric since property of covariance is that $cov(X,Y) = cov(Y,X)$
    - can be $n \times n$ where covariance of each dimension with itself
    - for a zero-mean matrix $X$, we $\Sigma = \frac{1}{n-1} XX^T$ 

### Eigenvectors & Eigenvalues

- Eigenvectors of a matrix/linear transform are vectors who maintain the same direction before and after the transformation
- Eigenvalue is the scalar amount eigenvectors are stretched by
    - imaginary eigenvalues indicate rotation
- $Av = \lambda v$
- eigenvectors of $A$- direction of stretch of LT from $A$
- eigenvalues of $A$- magnitude/amt of stretch of LT from A
- Symmetric matrices have orthogonal eigenvectors and real eigenvalues 
    - covariance matrices are always symmetric

### Tieing them Together

- We can view the covariance matrix as a linear transformation
- The eigenvectors of $\Sigma$ indicate the directions of the variance of the data
- The eigenvalues of $\Sigma$ indicate the magnitude of the variance of the data along the direction of the eigenvectors
    - Larger eigenvalues will indicate more variance is captured along the direction of the corresponding eigenvector

## Covariance Perspective

1. Calculate the covariance matrix $\Sigma$ of the data 
2. Find the eigenvalues and corresponding eigenvectors of the covariance matrix
3. Select the eigenvectors with the largest eigenvalues to be the basis of a smaller subspace to represent the data 
    - Eigenvectors w/ largest eigenvalues represent the directions in which the most variance of the data is capture 
    - In order to accurately represent the data in lower dimensional space, we want to preserve as much of the original variance as possible
    - Can select a new basis based on desired dimension of the subspace or based on how much variance you want preserved (e.g. 80%)
        - sum of eigenvalues of current selected eigenvectors / sum of all eigenvalues = variance preserved in current subspace

## SVD Perspective

- Statistical Interpretation of SVD
- Gives us a heirirachial coordiante system (based on data)
    - want to uncover dominant combinations of features that describe the data as much as possible
- $X$ - each row is a sample
- assumption of gaussian distribution for the data

1. compute mean row and make a giant mean matrix $X'$ out of it that is the same size as $X$
2. subtract the mean from the data matrix to get the mean centered matrix $\Beta = X - X'$
    - centers data at the origin
3. Covariance matrix ($\Sigma$ in SVD) of rows of $\Beta$: $C = \Beta^T \Beta$

- Takeaway: Can compute principal components and loadings from the SVD of your mean subtracted data 
    - principal components: $T = U \Sigma$

## Optimization Perspective

- We want compute the major axis of variation $u$ of a mean normalized dataset
- $\max_u \frac{1}{n} \sum_{i=1}^n (x^{(i)^T} u)^2$ subject to $||u||_2 = 1$
    - $x^Tu$ is the distance/variation from the origin (assumes the data's mean is at the origin) so we want to maximize it for all samples, subject to the constraint of $u$ being a unit vector 
    - the problem expands to: $\max_u \frac{1}{n} u^T \Sigma u$ subject to $u^Tu = 1$ 
        - Can be solved using the method of lagrange multiplier to yield $\Sigma u = \lambda u$
- This means the top $k$ eigenvectors of $\Sigma \in \mathbb{R}^{n \times n}$ are the principal components that form a new orthogonal basis for the data in $\mathbb{R}^k$
- Given $x \in \mathbb{R}^n$ where $n >> k$, we can represent it in $\mathbb{R}^k$ using the $k$ principal components $u_i \in \mathbb{R}^n$
    - $\begin{bmatrix} u_1^Tx \\ \vdots \\ u_k^T x\end{bmatrix} \in \mathbb{R}^k$

## Applications

- Noise Reduction
- Image Compression 

## Resources

- [PCA using SVD Video](https://www.youtube.com/watch?v=fkf4IBRSeEc)
- [PCA Intuition Video](https://www.youtube.com/watch?v=g-Hb26agBFg)
- [Covariance Matrix](https://datascienceplus.com/understanding-the-covariance-matrix/)
- [Relationship btw SVD & PCA](https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca)
- [Tutorial on PCA](https://arxiv.org/pdf/1404.1100.pdf)
- [CS 229 Notes](https://cs229.stanford.edu/notes2021fall/cs229-notes10.pdf)