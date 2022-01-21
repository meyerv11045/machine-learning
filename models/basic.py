import torch

class FFNet(torch.nn.Module):
    """ Simple feed-forward neural network in PyTorch
        Consists of linear layers with a single activation function
    Attributes:
        layers: list of linear layers to be applied in forward pass
        activation: activation function to be applied between layers
    """

    def __init__(self, shape, activation = torch.nn.functional.relu):
        """ Constructor for FFNet
            
        Arguments:
            shape: list of number of nodes in each layer
            activation: pytorch function specifying the activation function for all the layers
        """
        super(FFNet, self).__init__()
        self.shape = shape 
        self.layers =  []
        self.activation = activation 

        for i in range(len(shape) - 1):
            self.layers.append(torch.nn.Linear(shape[i], shape[i + 1]))

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        """ Forward Pass through FFNet
        """

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        # no activation function on final layer    
        return self.layers[-1](x)