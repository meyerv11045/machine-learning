import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

def load_data(file_name):
    return pd.read_csv(file_name,header=None,names=['X1','X2','y'])

def plot_data(data,xlabel,ylabel,title,pos_label,neg_label):
    pos = data[data['y'].isin([1])]
    neg = data[data['y'].isin([0])]

    _, ax = plt.subplots(figsize=(12,8))
    ax.scatter(pos['X1'],pos['X2'],s=20,c ='b',marker='o',label=pos_label)
    ax.scatter(neg['X1'],neg['X2'],s=20,c='r',marker='x',label=neg_label)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()

def plot_computeCost(cost_history,iterations):
    _ , ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iterations), cost_history, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Number of Iterations')
    plt.show(block=False)
    input('Press <Enter> to continue')

if __name__ == '__main__':    
    pos_label = 'Admitted'
    neg_label = 'Not Admitted'
    xlabel = 'Exam 1 Score'
    ylabel = 'Exam 2 Score'
    title = 'Admission Based on Exam Scores'
    
    data = load_data('ex2data1.txt')
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    plot_data(data,xlabel,ylabel,title,pos_label,neg_label)

    X.insert(0,'ones',1)
    X = X.to_numpy()
    y = y.to_numpy().reshape((100,1))
    theta = np.zeros((X.shape[1],1))
    iterations = 2000
    alpha = 0.00001

    classifier = LogisticRegression()
    gradient,cost_history = classifier.gradient_descent(X,y,theta,iterations,alpha)
    
    plot_computeCost(cost_history,iterations)

    predictions = classifier.predict(X,gradient)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) % len(correct)
    print('Accuracy: {0}%'.format(accuracy))