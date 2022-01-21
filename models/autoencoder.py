import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self, encdr_shape, dcdr_shape, act_fn = nn.functional.relu):
    super(AutoEncoder, self).__init__()
    self.shape = [encdr_shape, dcdr_shape]
    self.activation = act_fn

    encoder = [nn.Linear(encdr_shape[i], encdr_shape[i+1]) for i in range(len(encdr_shape) - 1)]
    decoder = [nn.Linear(dcdr_shape[i], dcdr_shape[i+1]) for i in range(len(dcdr_shape) - 1)]
    self.encoder = nn.ModuleList(encoder)
    self.decoder = nn.ModuleList(decoder)

  def encode(self,x):
    for i in range(len(self.encoder)):
      x = self.activation(self.layers[i](x))
    return x

  def decode(self,x):
    for i in range(len(self.decoder)):
      x = self.activation(self.layers[i](x))
    return x

  def forward(self,x):
    return self.decode(self.encode(x))