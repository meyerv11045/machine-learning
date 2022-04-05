# Neural Networks

- network can be viewed as a function we want to learn that perfectly maps from inputs to outputs
    - given a dataset, how we define our loss function determines how well we learn the mapping (how well we approximate the function)
- logistic regression can be seen as a special instance of a neural network
- A neuron is an operation that is a composite of a *linear* operation with an *activation/thresholding* operation
    - Classification Tasks:
        - Neurons in early layers of a network become feature detectors for things such as edges
            - 1st layer can only detect edges or similar basic features since it is given only pixel information in a simple fully connected network  
            - later layers can detect more complex features by combining information from earlier feature detectors indicating the presence/orientation of certain edges or other simple features
        - Neurons in the later layers build up more complex feature detectors using the feature detector neurons in the previous layer and the weights connecting the layers 
    - Regression Tasks:
        - Layers learn more complex features from the input features (e.g. relationships between features, the weighting of importance of certain features)
        - The learned features are often better than hand engineered features
            - Often difficult to understand what features are being learned and that's why NN's are often considered black boxes
- A model is an *architecture* with *parameters*
- *end-to-end* learning- we train based on inputs and outputs and don't care about whats happening on the inside (blackbox model) 
- Loss Function $\mathcal{L}$- for 1 example
- Cost Function $J$- for multiple examples
    - $J = \frac{1}{m}\sum_{i=1}^m \mathcal{L}^{(i)}$
    - Can find the derivative by just computing derivative of $\mathcal{L}$ since summation is a linear operation so we can just compute the derivative inside the summation
- Normalizing input to neural networks can help it learn better since certain features will not have outsized influence due to their scaling/units
    - normalize = center (subtract by mean) and standardize (divide by std deviation)
    - this results in circular contours to the cost function which result in a smoother descent for gradient descent compared to the unnormalized input which will have ellipsoidal contours
    - mean and standard deviation from the train set must also be applied to test set
- Smaller the batch, the more noisy the convergence
    - Full batch is the best approximation of the true cost function
    - Smaller batches serve as good enough approximations of the true cost function that are faster to compute, allowing more updates to the parameters in a given amount of time
    - More (less accurate) updates are typically preferred over fewer (very accurate) updates
        - the most accurate updates computed over the whole batch will have the smoothest convergence
    - Some research suggests the stochastity of smaller batches/single samples in gradient descent can help it escape local minima in the cost function that full batch gradient might get caught in



## Weight Initialization

- One way to avoid vanishing and exploding gradients is by initializing the weights to be near the ideal values
- The larger the number of input features to a layer, the smaller we want each individual weight to be so that the dot product of $w^Tx$ does not explode/become too lare
    - For sigmoid activation an initialization that works well: $w^{[\ell] } = \text{random}(shape) \times \frac{1}{\sqrt{n^{[\ell - 1]}}}$
    - For ReLU activation an initialization that works well: $w^{[\ell] } = \text{random}(shape) \times \frac{2}{\sqrt{n^{[\ell - 1]}}}$
- Xavier Initialization 
    - used for sigmoid and tanh activation
    - [Read More](https://cs230.stanford.edu/section/4/)
- He Initialization:
    - used for relu activation
    - takes into account number of inputs and outputs b/c of the back propagation step 
- Weights need to be randomly initialized or else network will suffer from the symmetry problem where all neurons start learning the same thing (need to be able to evolve independently)