import numpy as np

class MultivariateLinReg():
    
    def computeCost(self,X,y,theta):
        m = len(y)
        return np.sum(np.power((X @ theta) - y,2)) / (2 * m)
    
    def gradient_descent(self,X,y,theta,iterations,alpha):
        m = len(y)
        J_history = np.zeros((iterations,1))
        for i in range(iterations):
            hx = X @ theta 
            error = X.T @ (hx - y) 
            theta -= (alpha/m) * error
            J_history[i] = self.computeCost(X,y,theta)
        return theta, J_history
    
    def normal_equation(self,X,y):
        inner = X.T @ X
        outside = X.T @ y 
        return np.linalg.inv(inner) @ outside
    
    def predict(self,x,theta): 
        x = np.array(x)
        return x @  theta #(1 x 3) * (3 x 1) = (1 x 1)