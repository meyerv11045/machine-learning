# Optimization Techniques

## Gradient Descent 

- Optimization technique
- **Stochastic**: cost function for single datapoint before doing a model parameter update
    - take step for each datapoint in dataset 
    - may get closer to minimum faster than batch but might never converge and just keep oscillating around the minmum (usually approximations of the minimum are good enough)
        - adaptive learning rate that decreases over time will ensure params converge and dont oscillate around the min
    - preferred when the training set is large
- **Batch**: cost function over the entire dataset before doing a model parameter update
    - scan entire dataset before taking step
- For convex cost functions, gradient descent always converges to the global minimum assuming learning rate is not too large


## Coordinate Descent

- [Wiki](https://en.wikipedia.org/wiki/Coordinate_descent)