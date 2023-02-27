from .micrograd import Value
from random import uniform

class Neuron:
    def __init__(self,inputs,type='linear'):
        self.w = [Value(uniform(-1,1)) for _ in range(inputs)]
        self.b = Value(uniform(-1,1))
        self.type = type
        
    def forward(self,x):
        return sum((wi*xi for wi, xi in zip(self.w,x)),self.b)
    
    def __call__(self,x):
        f = self.forward(x)
        if self.type == 'linear':
            return f
        elif self.type == 'relu':
            return f.relu()
        elif self.type == 'tanh':
            return f.tanh()
        elif self.type == 'sigmoid':
            return f.sigmoid()
        else:
            raise ValueError(f'{self.type} is not allowed')
            
class Layer:
    def __init__(self,inputs,outputs,type='linear'):
        self.inputs = inputs
        self.outputs = outputs
        self.neurons = [Neuron(inputs,type=type) for _ in range(outputs)]
    
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs
    

class MLP:
    def __init__(self,inputs,layer_sizes, type_layers):
        sz = [inputs] + layer_sizes
        self.layers = [Layer(sz[i],sz[i+1],type=t) for i,t in zip(range(len(layer_sizes)),type_layers)]
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

        