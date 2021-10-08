# Activation Functions

- Mathematical equations that determine the output of a node/neuron in a neural network
- Determines whether each neuron should be activated based on the neuron's input and the specific activation function
- Normalize the output of a neuron to a range: [0,1] or [-1,1]
- Must be computationally efficient b/c they are calculated across thousands or millions of neurons for each pass
  - Having a computationally efficient derivative is good too for backpropagation
- Can be as simple as a step function that turns the neuron output on or off (e.g. 0 or 1) 
  - Binary Step Function (Doesn't allow multi-value outputs so it can't do multi-class classification)
- Can be a trasnformation that maps input signals to output signals
- Linear activations functions collapse a neural network into one layer b/c a linear combination of linear functions is still a linear function (not very useful)
  - Can't use backpropagation with linear activation functions b/c the derivative is a constant so its impossible to see the effect of changing weights
- Non-linear activation functions allow the neural network to learn more complex data like images, videos, and audio
  - Allow backpropagation b/c the derivative function is related to the inputs
  - Allow sacking of multiple hidden layers unlike linear activation functions
  - Most all processes can be modeled by a neural network w/ nonlinear activation functions

## Why do we need activation functions?
Matrix multiplication and vector addition are simply linear transformations so in order to learn any nonlinear representations of data, we need to introduce a non-linear function or else the most complex classification boundary it can learn will be linear (e.g. line, plane, etc. ) which isn't that useful for interesting problems.

In order to introduce non-linearity (otherwise matrix ops will result in only linear transformations), we apply non-linear activation functions pointwise to the results of the linear transformations at each layer.

## Sigmoid/Logistic 
$\sigma(x) = \dfrac{1}{1 + e^{-x}}$

$\dfrac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))$

Pros: 

- *Smooth Gradient* prevents jumps in the output values
- *Output vaues bound* btw 0 and 1, normalizing the each neuron's output
- *Clear predictions*- any values above 2 or below -2 are very close to the edge of the curve 

Cons:

- Vanishing Gradient- almost no change in the prediction for very high and very low input values
  - Can result in the network refusing to learn further or being too slow to reach an accurate prediction
- *Outputs not zero centered*
- *Computationally expensive*



Detailed Derivation:

$$ \begin{align}\dfrac{d}{dx} \sigma(x) &= \dfrac{d}{dx} \left[ \dfrac{1}{1 + e^{-x}} \right] \\ &= \dfrac{d}{dx} \left( 1 + \mathrm{e}^{-x} \right)^{-1} \\ &= -(1 + e^{-x})^{-2}(-e^{-x}) \\ &= \dfrac{e^{-x}}{\left(1 + e^{-x}\right)^2} \\ &= \dfrac{1}{1 + e^{-x}\ } \cdot \dfrac{e^{-x}}{1 + e^{-x}}  \\ &= \dfrac{1}{1 + e^{-x}\ } \cdot \dfrac{(1 + e^{-x}) - 1}{1 + e^{-x}}  \\ &= \dfrac{1}{1 + e^{-x}\ } \cdot \left( \dfrac{1 + e^{-x}}{1 + e^{-x}} - \dfrac{1}{1 + e^{-x}} \right) \\ &= \dfrac{1}{1 + e^{-x}\ } \cdot \left( 1 - \dfrac{1}{1 + e^{-x}} \right) \\ &= \sigma(x) \cdot (1 - \sigma(x)) \end{align} $$

---

## TanH/ Hyperbolic Tangent
Similar in advantages and disadvantages except it has the advantage of being *zero centered*, making it easier to models input that have strongly negative, neutral, and strongly positive values



## ReLU (Rectified Linear Unit)
Pros: 

- *Computationally efficient*- allows the network to converge quickly 
- *Non-linear*- although it seems lik e a lienar function, its derivative allows for backpropagation

Cons:

- *The Dying ReLU problem*- as inputs approach zero or are negative, the gradient becomes zero, so the network can't perform backpropagation and therefore cannot learn



## Leaky ReLU
Pros:

- *Prevents dying ReLU problem*- the smallpositive slope in the negative area enables backpropagation for negative inputs
- Otherwise like ReLU

Cons:

- *Inconsistent Results*- does't provide consistent predicitions for negative input values

https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/