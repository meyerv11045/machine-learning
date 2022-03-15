# Kernel Methods

- **Attributes**- original input values
- **Features**- new set of quantities from the attributes
    - A feature map $\phi$ maps attributes to features
- GD update becomes computationally expensive when the features are high dimensional since its computing $\theta := \theta - \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)}))\phi(x^{(i)})$

## Kernel Trick

- Initialize $\theta = 0$
- Theta can be represented as a linear combination of vectors
- $\theta = \sum_{i=1}^n \beta_i \phi(x^{(i)})$



## radial basis function kernel

- when $|| x - z||^2 $ is large, the value of the kernel is small

    - far away -> small kernel values

    