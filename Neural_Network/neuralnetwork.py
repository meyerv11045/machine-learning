''' Module implementing the stochastic gradient descent learning algorthim for a
    feedforward neural network
    Gradients are calculateed using backpropagation
'''

import numpy as np
from random import shuffle

class NeuralNetwork:
    def __init__(self,layer_sizes):
        self.num_layers = len(layer_sizes)
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/(s[1]**0.5) for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]
    
    def feedforward(self,a):
        '''Returns the output of the network given an input vector a'''
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,alpha,test_data=None):
        ''' Train the neral network using mini-batch stochastic gradient descent
            training_data is a list of (x,y) tuples representing the training inputs and the desired outputs
            epochs is the number of iterations of SGD
        '''
        if test_data: n_test = len(test_data)
        m = len(training_data)
        for i in range(epochs):
            shuffle(training_data)
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0,m,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,alpha)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(i,self.evaluate(test_data),n_test))
            else:
                print('Epoch {0} complete'.format(i))

    def update_mini_batch(self,mini_batch,alpha):
        ''' Updates the network's weights and biasses by applying GD using backpropagation to
            a sinlg mini-batch
            mini_batch is a list of tuples (x,y)
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w -(alpha/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)] #Add regularization at this step
        self.biases = [b - (alpha/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        ''' Return a tuple (nable_b,nabla_w) representing the gradient for the cost function C_x
            nabla_b and nabla_w are layer-by-layer lists of np arrays, similar to self.biases and self.weights
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #Feedforward
        activation = x
        activations = [x] #Stores all activations, layer by layer
        zs = [] #Stores all the z vectors, layer by leyer

        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #backward pass (uses negative indices to access last elements in list)
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].T)

        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T,delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].T)
        return (nabla_b,nabla_w)

    def cost_derivative(self,output_activations,y):
        return (output_activations - y)

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))