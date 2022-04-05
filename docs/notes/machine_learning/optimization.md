# Optimization Techniques

## Gradient Descent 

- Optimization technique (aka steepest descent)

- **Stochastic**: cost function for single datapoint before doing a model parameter update
    - take step for each datapoint in dataset 
    - may get closer to minimum faster than batch but might never converge and just keep oscillating around the minmum (usually approximations of the minimum are good enough)
        - adaptive learning rate that decreases over time will ensure params converge and dont oscillate around the min
    - preferred when the training set is large
    
- **Batch**: cost function over the entire dataset before doing a model parameter update
    - scan entire dataset before taking step
    
- For convex cost functions, gradient descent always converges to the global minimum assuming learning rate is not too large

    - [MSE Cost is Convex Proof](https://thedatamint.github.io/Proving%20Convexity%20of%20Mean%20Squared%20Error%20Loss%20in%20a%20Regression%20Setting..html)

- Momentum- takes average of previous gradients into account

    - this results in less zigzagged trajectory throught the parameter space over the cost function

    - reduces changes in directions that are not useful in moving towards the min since previous gradients will all share a common direction towards the minimizer while less useful directions will tend to cancel, thus we want to generally follow the most followed direction of previous gradients

    - we don't update parameters directly, instead we update the velocity vector

        - very low memory (store 1 previous gradient which encapsulates the effect of gradients before it) and low computational cost but greatly helps convergence

    - ```
        v = Bv + (1 - B) gradient_w
        w = w - alpha * v
        ```



## Coordinate Descent

- [Wiki](https://en.wikipedia.org/wiki/Coordinate_descent)

## Advanced Techniques

- Conjugate Gradient
- BFGS- iterative method for solving unconstrained nonlinear optimization problems
    - determines descent direction by preconditioning gradient w/ curvature information
    - curvature from a gradually improved *approximation* of the hessian of the loss function (obtained from gradient evalations via a secant method
- L-BFGS