from pydantic import BaseModel, Field
import numpy as np
from typing import Type

#Local Imports
from .layers import layers_types

class RNN(BaseModel):
    n_input: int = Field(..., gt=0)
    n_output: int = Field(..., gt=0)
    n_hidden: int = Field(..., gt=0)
    layer_a: layers_types = Field(None)
    layer_y: layers_types = Field(None)
    
    def __init__(self,n_input:int, n_output:int,n_hidden:int,layer_y:Type, layer_a:Type):
        super().__init__(n_input=n_input,n_output=n_output,n_hidden=n_hidden)
        
        self.layer_a = layer_a(n_output=n_hidden,n_input=n_input+n_hidden)
        self.layer_y = layer_y(n_output=n_output,n_input=n_hidden)
        
    
    #for layer_y: n_output=self.n_output,n_input=self.n_hidden
    #layer_y: layers_types = Field(...)
    
    #for layer_a: n_output=self.n_hidden,n_input=self.n_input + self.n_hidden
    #layer_a: layers_types = Field(...)
   
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        
    def forward(self,x):
        a = np.zeros((self.n_hidden,1))
        y_output = np.zeros((self.n_output,x.shape[1]))
        a_output = np.zeros((self.n_hidden,x.shape[1]))
        for t in range(x.shape[1]):
            x_new_shape = np.expand_dims(x[:,t],axis=1)
            x_wa = np.concatenate([a,x_new_shape])
            
            a = self.layer_a.forward(x_wa)
            a_output[:,t] = np.squeeze(a)
            y_output[:,t] = np.squeeze(self.layer_y.forward(a))
            
        return y_output, a_output