# Model Predictive Control
- Relies on a dynamic model to make predictions for future values of the controlled variables of the system. 
- It then solves a contrained optimization problem to calculate the optimal control action that minimizes the difference between the predicted controlled values and the desired/set control values.
- Main limitation: High computational cost since this optimization problem must be solved at each time step  
    - The computational cost increases even more as the time horizon over which to predict future control values increases.
    - The computational cost becomes intractable to run realtime as the number of states and contraints grows (NN approximation could fix this)
    - Explicit MPC (eMPC) and learning-based NN approaches have been proposed to overcome this
- Takes on many forms: one-norm, infinite norm, PWA model, tracking


## Explicit MPC
- Goal is to rewrite the optimization problem as a multi-parametric quadtratic programming (mpQP)
- This is possible for linear systems but cannot be extended to nonlinear systems
- An explicit solution to the mpQP has been demonstrated and it is of the form of a PWA function
- The PWA function defined over the state space will act as a fast prediction/lookup function for control policies given a input state meaning there is no need to solve the optimization problem online!
    - This result was achieved by Bemporad et al. (2002)
- Resources: ([Brief mention](http://www.bcamath.org/documentos_public/archivos/actividades_cientificas/TalkNUMERIWAVES20130412TP.pdf))


## Optimality:
- Not always feasible to run the optimization process to complete convergence for the most optimal control output (especially when running online) 
- Therefore, optimality is sometimes relaxed to only require a sufficient decrease in the cost/objective function
    - Then the optimization problem would be terminated at a feasible but sub-optimal stage

## Recursive Feasability:
- MPC controllers are not gauranteed to be stabilizing (linear quadratic control is gauranted to be stabilizing)
    - Stability must be built in when defining the optimization problem
- The optimization problem’s feasbility can be lost when the state is driven to a region in which the optimization problem has no solution
- typical approach to solve this is to append constrainst to the optimization problem to guarantee that loss of feasability cannot occur (that is what was done in the papers I have read so far)
    - When this is done, an MPC controller is said to be recursively feasible meaning it cannot put the state in a place with no solution to the optimization problem
- Constructing an MPC controller with the theoretical (a-priori) guarantee of recursive feasability is not always desirable or easy to do
- Sometimes we might have a situation where given an MPC controller and the goal is to determine if it is recursively feasbile
    - The goal is primarly to invalidate the controller by detecting problematic states where the recursive feasability is lost 
    - Second objective is to find certificates of guaranteed recursive feasability
- DEF: The MPC controller is recursively feasible if and only if for all ini- tially feasible x0 and for all optimal sequences of control inputs the MPC optimization problem remains feasible for all time. 
- DEF: The MPC controller is strongly recursively feasible if and only if for all initially feasible x0 and for all sequences of feasible control inputs the MPC optimization problem remains feasible for all time. 
- [Resource](https://users.isy.liu.se/johanl/2011_AUT_OOPS.pdf) (analysis tool for determining recursive feasability of an MPC controller)