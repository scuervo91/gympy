from matplotlib import rc_params_from_file
import numpy as np
from pydantic import BaseModel, Field

#local imports
from .functions import linear, relu, sigmoid, softmax

class Layer(BaseModel):
    n_features: int = Field(..., gt=0)
    
class Linear(Layer):
    
    def forward(w,x,b):
        return linear(w,x,b)
    
class Relu(Layer):
    
    def forward(w,x,b):
        z = linear(w,x,b)
        return relu(z)
    
class Sigmoid(Layer):
    
    def forward(w,x,b):
        z = linear(w,x,b)
        return sigmoid(z)
    
class Softmax(Layer):
    
    def forward(w,x,b):
        z = linear(w,x,b)
        return softmax(z)
    

    
    
