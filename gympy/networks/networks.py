from pydantic import BaseModel, Field
import numpy as np
from typing import List, Union, Callable

# local imports
from ..layers import Linear,Sigmoid, Tanh, Softmax,Relu
from ..optimizers import GradientDescent
from ..loss import CategoricalCrossEntropy

layers_types = Union[Linear,Sigmoid, Tanh, Softmax,Relu]
optimizers_types = Union[GradientDescent]

loss_types = Union[CategoricalCrossEntropy]

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
    loss: loss_types = Field(CategoricalCrossEntropy())
    cost: List[float] = Field(None)
    batch_size: int = Field(None, gt=0)
    seed: int = Field(None, gt=0)
    lambd: float = Field(0.0)
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        
    def forward(self, x):
        A = x
        cache_list = []
        cache_list.append(x)
        for i,layer in enumerate(self.layers):
            A = layer.forward(A)
            
            #Apply dropout
            if layer.dropout_rate > 0:
                dropout_array = layer.dropout_array()
                A *= dropout_array
                layer.dropout_cache = dropout_array
                
            #Append to cache list
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
            
    def get_regularization_loss(self):
        if self.lambd > 0.0:
            reg_cost = 0
            lambd = self.lambd
            for layer in self.layers:
                reg_cost += layer.regularization_loss(lambd)
            return reg_cost
        else:
            return 0.
    
    # def get_batch(self):
    #     if self.batch_size is None:
            
    
    def train(self, x, y, show=10, n_iter=100):
        n_layers = len(self.layers) + 1
        cost_list =[]
        for epoch in range(n_iter):
                       
            #Forward
            AL = self.forward(x)
            
            #Cost
            cost = self.loss(AL, y) + self.get_regularization_loss()
            cost_list.append(cost)
            
            #first dz
            #self.layers[-1].dz = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
            self.layers[-1].dz = AL - y
            
            #First grad
            dw, db = gradients(self.layers[-1].dz, self.cache[-2])
            
            # L2 Regularization
            if self.lambd > 0:
                dw += 2*self.lambd*self.layers[-1].weights
            
            #Save Gradients
            self.layers[-1].grads_dw = dw
            self.layers[-1].grads_db = db

            #For loop for layers
            for l in reversed(range(0,n_layers-2)):
                #Linear Backward
                self.layers[l].dz = linear_backward(
                    self.layers[l+1].weights,
                    self.layers[l+1].dz,
                    self.layers[l].derivative()
                )
                #Gradients
                dw_l,db_l = gradients(self.layers[l].dz,self.cache[l])
                if self.layers[l].dropout_rate > 0:
                    dw_l *= self.layers[l].dropout_cache
                
                #L2 Regularization
                if self.lambd > 0:
                    dw_l += 2*self.lambd*self.layers[l].weights
                self.layers[l].grads_dw = dw_l
                self.layers[l].grads_db = db_l
                
            #Get new Weights
            new_weights = self.optimizer.update(self.get_weigths(),self.get_grads_dw())
            new_bias = self.optimizer.update(self.get_bias(),self.get_grads_db())
            self.assing_weights(new_weights)
            self.assing_bias(new_bias)
            
            if epoch%show==0:
                print(f'{epoch} cost {cost}')
        
        self.cost = cost_list