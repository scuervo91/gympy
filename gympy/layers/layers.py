import numpy as np
from pydantic import BaseModel, Field, validator
from enum import Enum
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
    grads_dw: np.ndarray = Field(None)
    grads_db: np.ndarray = Field(None)
    
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
    

    
    
