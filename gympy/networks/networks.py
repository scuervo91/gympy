from pydantic import BaseModel, Field
import numpy as np
from typing import List, Union, Callable

# local imports
from ..layers import Linear,Sigmoid, Tanh, Softmax,Relu
from ..optimizers import GradientDescent
from ..loss import cross_entropy

layers_types = Union[Linear,Sigmoid, Tanh, Softmax,Relu]
optimizers_types = Union[GradientDescent]

def gradients(dz, a):
    m = dz.shape[1]
    
    dw = (1/m) * np.dot(dz, a.T)
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    return dw, db

def linear_backward(w, dz, dev_z):
    return np.dot(w.T, dz) * dev_z

class NeuralNetwork(BaseModel):
    layers: List[layers_types] = Field(...)
    cache: List[np.ndarray] = Field(default=None)
    optimizer: optimizers_types = Field(...)
    cost_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = Field(cross_entropy)
    n_iter: int = Field(default=100)
    loss: List[float] = Field(None)
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        
    def forward(self, x):
        A = x
        cache_list = []
        cache_list.append(x)
        for layer in self.layers:
            A = layer.forward(A)
            cache_list.append(A)
        self.cache = cache_list
        return A
    
    def get_weigths(self):
        return [layer.weights for layer in self.layers]
    
    def train(self, x, y):
        n_layers = len(self.layers)
        for i in range(self.n_iter):
            grads_list = []
            dz_list = []
            AL = self.forward(x)
            cost = self.cost_function(AL, y)
            dz = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
            dz_list.append(dz)
            grads_list.append(gradients(dz, self.cache[-2]))

            for l in reversed(range(n_layers-1)):
                dev_z = self.layers[l].derivative()
                dz_ = linear_backward(
                    self.layers[l+1].weights,
                    dz_list[

                
            
            self.optimizer.update(self.layers, dz)    
    
            
        
    
