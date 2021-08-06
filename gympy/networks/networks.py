from pydantic import BaseModel, Field
import numpy as np
from typing import List, Union

# local imports
from ..layers import Linear,Sigmoid, Tanh, Softmax,Relu

layers_types = Union[Linear,Sigmoid, Tanh, Softmax,Relu]

class NeuralNetwork(BaseModel):
    layers: List[layers_types] = Field(...)
    cache: List[np.ndarray] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        
    def forward(self, x):
        A = x
        cache_list = []
        for layer in self.layers:
            A = layer.forward(A)
            cache_list.append(A)
        self.cache = cache_list
        return A
            
        
    
