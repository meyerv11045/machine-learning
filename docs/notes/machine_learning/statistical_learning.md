# Statistical Learning

- Supervised Learning- building a statistical model for prediting/estimating an ouput based on some inputs. Basically is function approximation
  - Regression- predicting continuous or quantitative outputs
    - Least squares was the earliest form
  - Classifiction- predicting discrete/categorial or qualitative outputs
    - Linear discriminant analysis was the earliest form
- Unsupervised Learning- learn the relationships and strucuture from input data
  - Clustering- want to group datapoints according to similarity

- Generalized Linear Model- describes a class of stat learning methods including both linear and logistic regression
- Only linear methods were used till the 80s becuase non-linear relationships were too computationally expensive to fit at the time
- In the 80s, computational power was sufficient for nonlinear methods to flourish 
  - Classification and Regression trees
  - Generalized additive models
  - Neural Networks
  - Support Vector Machines (90s)
- In stat learning, inputs are referred to as predictors or independent vars (inputs/features in ML/CS)
  - outputs are responses or dependent variables (ouputs/target in ML)

# Fitting Linear Models

Let there be $n$ datapoints where each datapoint has $m$ features.

We want to find the optimal parameters for $f(x) = a_0 + \sum_{i=1}^m a_i x_i  $.

- Linear regression of single var: model for making predictions is a line in 2D

  - of 2 vars: model is a plane in 3D (for nonlinear models it would be a mesh/surface)

  - of m vars: model is a hyperplane in the m+1 dimensional input-output space



## Least Squares

- Makes huge assumptions about structure of data and yields stable but possibly inaccurate predictions
  - relies heavily on assumption that a linear model/decision boundary is appropriate
  - low variance, high bias
- We fit a linear model to a set of training data by minimizing a cost function defined by the residual sum of squares RSS
  - $RSS(\theta) = \sum_{i=1}^n(y_i - x_i^T\theta)^2$
  - Vectorized:
- We can take the gradient of the cost function, set equal to zero and then solve for the parameters
  - This way of computing the optimal paramters for a linear model is known as the normal equation
- Instead of the normal equation, could use gradient descent algorithm to iteratively find the minimum of the quadratic cost function

## k-nearest neighbor

- Makes very mild assumptions about structure of data and yields unstable but accurate predictions
  - no assumptions on the underlying data like in least squares resulting in wiggly, unstable decision boundaries that depend on only a handful of neighboring points
  - high variance, low bias 
- Uses observations in the training set closest to the input to create predictions
- $$h(x) = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i$$
    - $N_k(x)$ is the neighborhood of $x$ defined by the closest $k$ points $x_i$ in the training set
    - Prediction is just the average of the k-closest datapoints in the training set
- How do we define closeness in building the neighborhood?
    - Euclidean distance is nice



## Summary

- these two techniques are very popular (e.g. 1-nearest-neighbor is most used technique in low-dim problems)
- advancements/improvements:
    - kernel methods use weights that decrease smoothly to zero as the distance increases from the target point (rather than the abrupt 0/1 used by k-nearest neighbors)
    - distance kernels are modified to emphasize some variables more than others in high dimensional spaces
    - Local regression fits linear models by locally weighted least squares
    - Linear models fit to an expanded basis of the original inputs allow arbitrarily complex models
      - idk why you would ever want this
    - neural networks constist of sums of nonlinearly transformed linear models






## model selection & cross validation

- Given a good model for our training data, is the prediction's error on unseen data good or bad?
- k-fold cross validation
    - better than simple train test split for estimating model prediction error
    - divide dataset into k-sets 
    - For each subset:
      - use it once as a test set and the other k-1 sets for training and record the results
      - discard the model and start over again
    - End result is $k$ performance measures which can be averaged to get the models overall performance
    - gives insight into which $d$ is best for polynomial fit
    - Common $k = \{2, 5, 10\}$
    - better than hold-out 
- leave one out cross validation (k = m)
    - useful when m is small 
    - k = 5 or 10 is a good compromise
- select d based on training and validation best performer (hold out the test data)
    - train final model on training + validation data and then measure performance on testing data
- Training Set: fit model to best parameters
- Validation Set: evaluate model with MSE
- At the end, retrain and measure performance with test set

- split into train and test set 
    - hold out test set for final evaluation (never used for training/model updates)
    - variety of models with diff hyperparams train on training data and evaluate with validation data
    - once the best model hyperparams have been selected, train on the training and validation data
    - evaluate final model with test data that has never been seen before

- train, val, test = 60%, 20%, 20% of dataset 

- as you increase $d$ meaning the polynomial becomes more complex, it is able to fit the data better so the training loss will continue going down as $d$ increses since essentially you reach a point where the curve is overfitting the data
    - comparing the train loss to the validation loss and picking the $d$ with the minimum validation loss is the best
