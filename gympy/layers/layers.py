import numpy as np
from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import Union, Callable, Type
#local imports
from .functions import linear, relu, relu_derivative, sigmoid, sigmoid_derivative, softmax, tanh, tanh_derivative


class LayersEnum(str, Enum):
    linear = 'linear'
    relu = 'relu'
    sigmoid = 'sigmoid'
    tanh = 'tanh'
    softmax = 'softmax'


class Layer(BaseModel):
    n_input: int = Field(..., gt=0)
    n_output: int = Field(..., gt=0)
    weights: np.ndarray = Field(None)    
    bias: np.ndarray = Field(None)
    x: np.ndarray = Field(None)
    z: np.ndarray = Field(None)
    dz: np.ndarray = Field(None)
    dw: np.ndarray = Field(None)
    db: np.ndarray = Field(None)
    dropout_rate: float = Field(0, ge=0, lt=1)
    dropout_cache: np.ndarray = Field(None)
    
    @validator('x','z')
    def parse_values(v):
        return np.array(v, dtype=float)
    
    @validator('weights', always=True)
    def wights_shape(v,values):
        if v is None:
            return np.random.randn(values['n_output'],values['n_input'])*0.01
        arr = np.array(v, dtype=float)
        assert arr.shape == (values['n_output'],values['n_input'])
        return arr

    @validator('bias', always=True)
    def bias_shape(v,values):
        if v is None:
            return np.random.randn(values['n_output'],1)*0.01
        arr = np.array(v, dtype=float)
        assert arr.shape == (values['n_output'],1)
        return arr
        

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        
    def linear_forward(self,x):
        w = self.weights
        b = self.bias
        z = linear(w,x,b)
        self.x = x
        self.z = z
        return z
    
    def regularization_loss(self,lambd):
        return lambd * np.sum(np.square(self.weights))
    
    def dropout_array(self):
        dp_rate = 1 - self.dropout_rate
        size_output = self.n_output
        return np.random.binomial(1,dp_rate,(size_output,1)) / dp_rate
    
class Linear(Layer):
    type: LayersEnum = Field(LayersEnum.linear, const=True)
    
    def forward(self,x):        
        z = self.linear_forward(x)
        return z
    
    def derivative(self):
        return self.z
    
class Relu(Layer):
    type: LayersEnum = Field(LayersEnum.relu, const=True)
    
    def forward(self,x):
        z = self.linear_forward(x)
        return relu(z)
    
    def derivative(self):
        return relu_derivative(self.z)
    
class Sigmoid(Layer):
    type: LayersEnum = Field(LayersEnum.sigmoid, const=True)
    
    def forward(self,x):
        z = self.linear_forward(x)
        return sigmoid(z)

    def derivative(self):
        return sigmoid_derivative(self.z)
    
class Tanh(Layer):
    type: LayersEnum = Field(LayersEnum.tanh, const=True)
    
    def forward(self,x):
        z = self.linear_forward(x)
        return tanh(z)
    
    def derivative(self):
        return tanh_derivative(self.z)
    
class Softmax(Layer):
    type: LayersEnum = Field(LayersEnum.softmax, const=True)
    
    def forward(self,x):
        z = self.linear_forward(x)
        return softmax(z)


layers_types = Union[Linear,Sigmoid, Tanh, Softmax,Relu]

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
        for t in range(x.shape[1]):
            x_new_shape = np.expand_dims(x[:,t],axis=1)
            x_wa = np.concatenate([a,x_new_shape])
            
            a = self.layer_a.forward(x_wa)
            y_output[:,t] = np.squeeze(self.layer_y.forward(a))
            
        return y_output
