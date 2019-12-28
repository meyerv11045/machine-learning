import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def load_data(file_name):
    x = pd.read_csv(file_name,header=None,usecols=[0],names=['Population'])
    y = pd.read_csv(file_name,header=None,usecols=[1],names=['Profit'])
    return x,y 

def plot_data(x,y,xlabel,ylabel,title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x,y,marker='x')
    plt.show(block=False)
    input('Press <Enter> to continue')

def plot_trendline(x,y,theta,xlabel,ylabel,title): 
    x1 = np.linspace(x.min(),x.max(),len(y))
    f = theta[0,0] + (theta[1,0] * x1)

    _ , ax = plt.subplots(figsize=(12,8))
    ax.plot(x1, f, 'r', label='Prediction')
    ax.scatter(x,y, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show(block=False)
    input('Press <Enter> to continue')

def plot_computeCost(cost_history,iterations):
    _ , ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iterations), cost_history, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Number of Iterations')
    plt.show(block=False)
    input('Press <Enter> to continue')

if __name__ == '__main__':
    xlabel = "Population of City in 10,000s"
    ylabel = "Profit in $10,000s"
    title = "Population of City vs. Profit"
    
    x,y = load_data('ex1data1.txt')
    
    plot_data(x,y,xlabel,ylabel,title)
    
    theta = np.zeros((2,1))
    iterations = 1000
    alpha = 0.01
    x.insert(0,'X0',1)
    X = x.to_numpy()
    Y = np.array(y)

    lin_reg = LinearRegression()
    gradient, cost_history = lin_reg.gradient_descent(X,Y,theta,iterations,alpha)

    plot_trendline(x.iloc[:,1].to_numpy(),y.iloc[:,0].to_numpy(),gradient,xlabel,ylabel,title) #Ignore X0 = 1 in x dataframe
    plot_computeCost(cost_history,iterations)
    print(lin_reg.predict_intuition(3.5,gradient))
    print(lin_reg.predict_vectorized([1,3.5],gradient))