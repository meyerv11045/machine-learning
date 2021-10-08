# Overview
Optimization is a huge field with a tremendous number of applications. It is also intertwined in ML so that is why its include in the ML notebook. Here is where I will try to keep a working glossary of areas I am being introduced to with a brief overview of what they are. Being able to map out how many pieces of a field fit together greatly helps me when diving deeper into any one area.

## What is Optimization?
Optimizing simply stands for finding the maximum or minimum of a function. This max or min is often subject to a series of constraints. The form of the function being optimized and the constraints being adhered to (e.g. linear, nonlinear, continuous, etc.) determines the type of problem and thus the technique used to solve them. 

Here is the mathematical definition of an optimization problem: 

$$\min f_0(x) $$
subject to the contraints:
$$ f_i(x) \leq b_i$$

[Here](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) is the textbook on convex optimization used for reference.

---
## Linear Programming (LP)
Optimizing a linear objective/cost function subject to linear equality/inequality constraints 

---
## Nonlinear Programming (NLP)
Optimizing an objective/cost function over a set of real variables subject to system of equations called constraints (more general class that includes LP)

### Quadtratic Programming (QP)
Methods for optimizing a quadtratic objective function subject to linear constraints

---
## Sequential Programming

### Sequential Quadtratic Programming (SQP)
Involves solving a series of subproblems where each is a quadratic program


### Sequential Linear Quadtratic Programming (SLQP)
Involves solving a linear program and equality constrained quadratic program at each step

---
## Dynamic Programming (DP)
Involves breaking down the main optimization problem into simpler sub-problems.

The **Bellman Equation** describes the relationship between the value of the larger problem and the values of the sub-problems

---
## Piece Wise Affine Functions (PWAs)
Approximates a function using a series of lines in 2D, planes in 3D, & hyperplanes in higher dimensional spaces. 

![PWA of Function](static/PWA-of-real.png)

While the picture above shows the PWA given a known function, PWAs are most often used when there is a set of datapoints representing the functions inputs and outputs. The task is then to construct a PWA from the data such that it best approximates the true function that is mapping the inputs to the outputs.
![PWA from Data](static/PWA-from-data.png)

Once you have a PWA made from your dataset, you can then make predictions on new inputs as to what the true function would map to.