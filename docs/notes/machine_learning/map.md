# Map of ML

## Supervised Learning

## Unsupervised Learning

### Auto-Encoders
num input layers = num output layers but smaller num hidden layers.

This forces the network to learn a compressed representation of the data. A simple autoencoder could take 100 dimensional data (100 input units) and represent it low dimensionally as 50 dimensions if using 50 hidden units.

Relies on the input data having features (completely random noise would be tough to learn anything from)

Mathematically, it can be thought of as trying to learn an approximation to the identity function so the output is similar to the input. Allows us to discover interesting structure about the data.

## Reinforcement Learning
A control policy $\pi$ maps the continuous state vector $x$ to a continuous action vector $u$ (often a control vector).

$u = \pi(x,t,\theta)$ where $\theta$ is the set of parameters to be learned by the control policy and $t$ is the specific timestep


## Model Predictive Control
Learning a control policy offline and deploying for online, realtime predictions:
- Reinforcement learning 
    - Difficulty lies in structuring rewards and exploration so that the agent learns the optimal control policy
    - Takes a lot of episodes to train and converge on an optimal control policy
- Supervised learning 
    - Use an an algorithm that is known to create good control trajectories to generate ground truth training data on a bunch of environment state inputs (including edge cases)
        - Data generateed: (environment/agent state vector, control trajectory ground truth)
        - Theoretically can then train a NN to approximate this planning algorithm given enough good data 
    - Difficulty lies in generating enough high quality training data (especially for things like robot control policies) 
