# Supervised Learning

## Data Scaling for Model Stability

## Function Approximation
Function approximation is a technique for estimating an unknown underyling function using observations from the domain and their resulting image.


Given a dataset of inputs and outputs, we can assume there is an unkwnown underyling function that maps the inputs to the outputs. We can use supervised learning techniques to approximate this function.

Supervised learning achieves this by minimizing the error between predicted outputs and expected outputs from the domain by changing its parameters (the weights of the network).

Thus, simple feedforward neural networks can be better thought of as function approximators rather than models of how the brain works.

The true function mapping inputs to outputs is referred to as the **target** function since it is the target of the learning process.

Thus, intuitively we can better get a sense of what the function is by observing more input-output data points.

The universal approximation theorem states a feedforward NN with a linear output layer and at least 1 hidden layer with a nonlinear activation function can approximate any function mapping between two finite dimensional spaces with any desired non-zero amount of error (provided enough hidden units).

Below is an example of $y=x^2$ being approximated by a simple feedforward NN with 2 hidden layers of 10 nodes with relu activation trained on 100 observations:
![Example Approximation](https://machinelearningmastery.com/wp-content/uploads/2019/12/Scatter-Plot-of-Input-vs-Actual-and-Predicted-Values-for-the-Neural-Net-Approximation.png)


## Resources
- [Function Approximators](https://machinelearningmastery.com/neural-networks-are-function-approximators/)