import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multivariate_linear_regression import MultivariateLinReg
#from lin_reg_demo import plot_computeCost

def load_data(file_name,headers,n):
    x = pd.read_csv(file_name,usecols=[i for i in range(n)],header=None,names=headers)
    y = pd.read_csv(file_name,usecols=[n],header=None,names=['Output'])
    return x,y

def plot_computeCost(cost_history,iterations):
    _ , ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iterations), cost_history, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Number of Iterations')
    plt.show(block=False)
    input('Press <Enter> to continue')

if __name__ == '__main__':
    n = 2 #Number of features
    mlin_reg = MultivariateLinReg()
    X,y = load_data('ex1data2.txt',['X'+str(i) for i in range(1,n+1)],n)
    
    #Feature Scaling- Mean Normalization
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    theta = np.zeros((n+1,1))
    iterations = 1000
    alpha = 0.01
    X.insert(0,'X0',1)
    X = X.to_numpy()
    y = np.array(y)

    gradient, cost_history = mlin_reg.gradient_descent(X,y,theta,iterations,alpha) 
    plot_computeCost(cost_history,iterations)
   
    '''
    #Normal Equation Demonstration:
    init_cost = mlin_reg.computeCost(X,y,theta)
    print("Inital Cost: " + str(init_cost))
    gradient = mlin_reg.normal_equation(X,y)
    final_cost = mlin_reg.computeCost(X,y,gradient)
    print("Final Cost: " +str(final_cost))
    print("Thetas: ")
    print(gradient)
    '''