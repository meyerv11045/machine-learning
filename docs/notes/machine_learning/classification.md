# Classification

## Perceptron

- Very early learning algorithm that performs simple linear classification between two classes
    - Never converges if training data is not linearly separable 
- $z = w^Tx + b$ 
- $g(z) = \begin{cases}1 & z \geq 0 \\ 0 & z < 0\end{cases}$
- Update rule:
    - $\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})x_j^{(i)}$
- Its predictions do not have meaningful probabilistic interpretations
- Cannot be derived as a maximum likelihood estimation algorithm
- Forms the basis for multilayer perceptrons (neural networks)
    - Backpropagation is used to train
    - Nonlinear activitaion functions that are part of the network need to be differentiable
- [Perceptron vs Logistic Regression](https://www.cs.utexas.edu/~gdurrett/courses/sp2021/perc-lr-connections.pdf)

## Logistic Regression

- Fisher Scoring- when the log likelihood function is optimizied using multi-dimensional Newton's method
    - $\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$
    - $H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$
    - Faster convergence than batch gradient descent but one iteration can be more expensive due to finding and inverting hessian
        - If number of params is small, and therefore hessian is small, then it is usually much faster overall
- [CS 229 Notes](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf)

## Softmax Regression

- aka Multinomial Logistic Regression
- Generalization of logistic regression to $C$ classes 
    - If $C = 2$ then softmax reduces to binary logistic regression

- Creates linear decision boundaries between classes
    - Adding hidden layers with nonlinear activation functions is what allows NNs to learn nonlinear decision boundaries

- Applies the softmax activation function to a vector of real numbers to produce a vector of probabilities
    - $S(x) = \frac{e^x}{\sum_{i =1}^C e^{x_i}}$
    - $S: \mathbb{R}^n \to \mathbb{R}^n$
    - The sum of the output probabilites = 1
    - Applies the exponetial elementwise to a vector 
    - Temperature parameter $t$ is a scalar that $x$ is multiplied by before being passed to the softmax activation
        - $0 < t < 1$ moves the probabilities closer together
        - $t > 1$ moves the probabilites further apart

- Uses cross entropy loss to train $\mathcal{L}(y, \hat y) = - \sum_{j=1}^C y_j \log \hat y_j$
    - assumes target $y$ uses one-hot encoding (e.g. $y^{(i)} = (0, 0, 1, 0)^T$)
    - In order to minimize $-\log \hat y$ we need to make $\hat y >> 0$ 

- $J(\theta) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(y^{(i)}, \hat y^{(i)})$
- [Stanford UFLDL](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)
- [DeepLearningAI Video on Softmax Regression](https://www.youtube.com/watch?v=LLux1SW--oM)

## Gaussian Discriminant Analysis

## Naive Bayes

- Features are discrete unlike the continue features in Gaussian Discriminant Analysis

- Ex: dictionary of words showing up in an email to classify whether it is spam 

    - mispelled words in spam messages to try to get past these dictionaries of possible spam values

    - the feature vector of words that show up in the desired dictionary is considered a bernoulli event model 

    - a multinomial event model takes into account the structure of the sentence or how the words appear in order to help prevent stuffing an email with a bunch of hidden good words that help the email get past the spam filter using a bernoulli event model
        - naive bayes is no longer valid
            - the feature vector will now depend on email length
            - laplace mothing based on dictionary size instead of the size of the feature vector

## Support Vector Machines

## Multiclass Classification

- To classify an input into one of $k$ classes, there are two techniques that can be applied to any of the classification methods (logistic regression, perceptron, SVM)
- [ML Mastery](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

### One-vs-All

- aka one-vs-rest
- For each $i = 1, \dots, k$ train a binary classifier to succesfully classify $y = i$. 
    - This yields a set of parameters $\theta^{(1)}, \dots, \theta^{(k)}$

- To make predictions for $x$, select the class $i$ that produces the highest value for $h_\theta^{(i)}(x) = (\theta^{(i)})^Tx$

### One-vs-One

- For each class $i = 1, \dots, k$ train a separate binary classifier to succesfully classify $y = i$ against each other class
    - Results in $\frac{k(k-1)}{2}$ Classifiers 
-  To make predictions for $x$:
    - Each model may predict a class label for $x$ and the class label with the most votes (most frequently occuring) amongst all the models is the predicted label
    - Each model may predict a probability of a class membership of $x$, so then the probabilites for each class membership are summed up and the class with the highest probability is the predicted label.
- Example w/ 4 classes: ‘*red*,’ ‘*blue*,’ and ‘*green*,’ ‘*yellow*.’ This would be divided into six binary classification problems:
    - 1: red vs. blue
    - 2: red vs. green
    - 3: red vs. yellow
    - 4: blue vs. green
    - 5: blue vs. yellow
    - 6: green vs. yellow