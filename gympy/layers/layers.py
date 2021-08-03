import numpy as np
from pydantic import BaseModel, Field, validator

#local imports
from .functions import linear, relu, sigmoid, softmax, tanh

class Layer(BaseModel):
    n_input: int = Field(..., gt=0)
    n_output: int = Field(..., gt=0)
    weights: np.ndarray = Field(None)    
    bias: np.ndarray = Field(None)
    x: np.ndarray = Field(None)
    z: np.ndarray = Field(None)
    
    @validator('x','z')
    def parse_values(v):
        return np.array(v, dtype=float)
    
    @validator('weights', always=True)
    def wights_shape(v,values):
        if v is None:
            return np.random.randn(values['n_output'],values['n_input'])
        arr = np.array(v, dtype=float)
        assert arr.shape == (values['n_output'],values['n_input'])
        return arr

    @validator('bias', always=True)
    def bias_shape(v,values):
        if v is None:
            return np.random.randn(values['n_output'],1)
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
    
    def forward(self,x):        
        z = self.linear_forward(x)
        return z
    
class Relu(Layer):
    
    def forward(self,x):
        z = self.linear_forward(x)
        return relu(z)
    
class Sigmoid(Layer):
    
    def forward(self,x):
        z = self.linear_forward(x)
        return sigmoid(z)
    
class Tanh(Layer):
    
    def forward(self,x):
        z = self.linear_forward(x)
        return tanh(z)
    
class Softmax(Layer):
    
    def forward(self,x):
        z = self.linear_forward(x)
        return softmax(z)
    

    
    
