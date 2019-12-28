import numpy as np

class LinearRegression():

    def computeCost(self,X,y,theta):
        hx = X @ theta
        return np.sum(np.power(hx - y,2)) / (2 * len(X))

    def gradient_descent(self,X,y,theta,iterations,alpha):
        J_history = np.zeros((iterations,1)) 
        m = len(y)

        for i in range(iterations):
            hypothesis = X @ theta 
            error = hypothesis - y

            temp0 = theta[0] - ((alpha/m) * np.sum(error))
            temp1 = theta[1] - ((alpha/m) * (error.T @ X[:,1]))

            theta[0] = temp0
            theta[1] = temp1

            J_history[i] = self.computeCost(X,y,theta)
        return theta , J_history
    
    def normal_equation(self,X,y):
        inner = X.T @ X
        outside = X.T @ y 
        return np.linalg.inv(inner) @ outside

    def predict_intuition(self,x,theta):
        return theta[0] + theta[1] * x #y = mx + b where m = theta[1] and b = theta[0]
    
    def predict_vectorized(self,x,theta):
        x = np.array(x)
        return x @ theta