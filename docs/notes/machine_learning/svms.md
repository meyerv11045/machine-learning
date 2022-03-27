# Kernel Methods

- **Attributes**- original input values
- **Features**- new set of quantities from the attributes
    - A feature map $\phi$ maps attributes to features
- GD update becomes computationally expensive when the features are high dimensional since its computing $\theta := \theta - \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)}))\phi(x^{(i)})$

## Kernel Trick

- Avoids computing the explicit mapping needed to allow linear learning algorithms to learn a nonlinear decision boundary 
- Dot products in the high-dimensional feature space can be computed as a kernel function $K(x, x') = \phi (x)^T \phi(x')$ thus avoiding the need to actually compute $\phi (x)$

## Kernels

- Dot products measure similarity of two vectors (tells us how parallel two vectors are)
    - dot product projects a vector onto another and is the length of that projected vector 
- Since kernels replace dot products, they are a similarity measure
- They are a class of functions (defined by mercer's theorem) that provably correspond to the dot product of some higher dimensional mapping 
    - Allow us to replace all dot products in the dual SVM computation with the kernel function that represents the data in a higher dimensional (and linearly separable) feature space

### Linear

- Just the basic dot product (no feature mapping)
- Can be used when the data is already linearly separable

### Polynomial

$$K(x, x') = (x^Tx' + c )^d$$

where $c \geq 0$ and $d \geq 1$ are parameters of the kernel. 

- When $c = 0$ the kernel is called homogeneous 
- Commonly applied to NLP problems with $d = 2$ (higher $d$ usually results in overfitting on NLP problems) 
- The mapped to feature space is equivalent to that of polynomial regression however the number of parameters to learn is decreased drastically

### Radial Basis Function / Gaussian

$$K(x, x') = \exp(-\gamma || x - x'||^2)$$

where $\gamma = \frac{1}{2\sigma^2}$ is a parameter of the kernel that controls the spread.

- The RBF kernel projects features into an infinite dimensional Euclidean space  which is where it is able to linearly separate any originally nonlinearly seperable data (this is why it performs so well and is so often used)
    - [RBF Kernel as Projection into Infinite Dimensional Space](https://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/svms/RBFKernel.pdf) 
- Need to do feature scaling before using gaussian kernel 

### Other

- Specialized to data type:
    - String, Graph, Tree, Wavelet, etc. 
- chi-square
- cosine similarity
- Sigmoid: $$K(x, x') = \tanh (\beta x^T x' + a)$$



## Support Vector Machines

- Formulated as an optimization problem to find the optimal parameters 
- The dual form of the optimization problem depends only on the dot product of datapoints (similarity of the training examples)
    - Replacing this dot product with a kernel function allows the algorithm to learn a nonlinear decision boundary in the original input space by projecting the low dimensional features into a higher dimensional feature space where they are linearly separable 
    - Using the kernel trick allowsus to leverage the advantages of the feature mapping without the computational costs of computing the mapping for each datapoint
    - The cost function is convex so it guarantees the existance of a global extremum



### Resources

- [AI Master Kernel SVM](https://ai-master.gitbooks.io/kernel-svm/content/)

- [MIT Lecture (Great- builds up from basics!)](https://www.youtube.com/watch?v=_PwhiWxHK8o)

    