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

    def get_bias(self):
        return [layer.bias for layer in self.layers]
    
    def get_grads_dw(self):
        return [layer.grads_dw for layer in self.layers]

    def get_grads_db(self):
        return [layer.grads_db for layer in self.layers]
    
    def assing_weights(self,new_weights):
        for layer, weights in zip(self.layers,new_weights):
            layer.weights = weights
            
    def assing_bias(self,new_bias):
        for layer, bias in zip(self.layers,new_bias):
            layer.bias = bias
    
    def train(self, x, y):
        n_layers = len(self.layers) + 1
        for i in range(self.n_iter):
                       
            #Forward
            AL = self.forward(x)
            
            #Cost
            cost = self.cost_function(AL, y)
            
            #first dz
            self.layers[-1].dz = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
            
            #First grad
            dw, db = gradients(self.layers[-1].dz, self.cache[-2])
            self.layers[-1].grads_dw = dw
            self.layers[-1].grads_db = db

            for l in reversed(range(0,n_layers-2)):
                #Linear Backward
                self.layers[l].dz = linear_backward(
                    self.layers[l+1].weights,
                    self.layers[l+1].dz,
                    self.layers[l].derivative()
                )
                #Gradients
                dw_l,db_l = gradients(self.layers[l].dz,self.cache[l-1])
                self.layers[l].grads_dw = dw_l
                self.layers[l].grads_db = db_l
                
            #Get new Weights
            new_weights = self.optimizer.update(self.get_weigths(),self.get_grads_dw())
            new_bias = self.optimizer.update(self.get_bias(),self.get_grads_db())
            self.assing_weights(new_weights)
            self.assing_bias(new_bias)
            
            if i%100==0:
                print(f'{i} cost {cost}')