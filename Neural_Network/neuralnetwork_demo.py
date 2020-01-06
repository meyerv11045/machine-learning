import numpy as np
import neuralnetwork as nn
import matplotlib.pyplot as plt

def load_data(inpt,label):
    return list(zip(inpt,label)) #returns a list of (x,y) tuples where x = img & y = lbl
        
with np.load('mnist.npz') as data:
    #print(data.files) #Shows the different files & their names
    training_data = load_data(data['training_images'],data['training_labels'])
    test_data = load_data(data['test_images'],data['test_labels'])
    
'''
#Show an Image:
plt.imshow(training_data[0][0].reshape(28,28), cmap='gray')
plt.show()
input('Press <Enter> to continue')
'''

layer_sizes = (784,5,10)
net = nn.NeuralNetwork(layer_sizes)
net.SGD(training_data,30,10,3.0,test_data=test_data)