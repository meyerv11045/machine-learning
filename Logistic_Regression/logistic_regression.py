import numpy as np

class LogisticRegression():
    def computeCost(self,X,y,theta):
        m = len(y)
        hx = sigmoid(X @ theta)
        e = 1e-5
        return (((-y).T @ np.log(hx + e)) - ((1-y).T @ np.log(1-hx + e)))/m
    
    def gradient_descent(self,X,y,theta,iterations,alpha):
        J_history = np.zeros((iterations,1))
        m = len(y)

        for i in range(iterations):
            hx = sigmoid(X @ theta)
            error = hx - y 
            theta = theta - ((X.T @ error) * (alpha/m)) 
            J_history[i] = self.computeCost(X,y,theta)
        
        return theta, J_history

    def predict(self,X,theta):
        return np.round(sigmoid(X @ theta))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

    