from pydantic import BaseModel, Field
import autograd.numpy as np
from typing import List, Union, Type
#from autograd import elementwise_grad as egrad
from enum import Enum
# local imports
from ..layers import Linear,Sigmoid, Tanh, Softmax,Relu,layers_types
from ..optimizers import SGD, SGDMomentum, RMSprop, Adam
from ..loss import CategoricalCrossEntropy, LogisticLoss,MeanSquaredError


optimizers_types = Union[SGD, SGDMomentum, RMSprop, Adam]

loss_types = Union[CategoricalCrossEntropy,MeanSquaredError]

class RnnEnum(str, Enum):
    many_one = 'many_one'
    many_many = 'many_many'


def gradients(dz, a):
    m = dz.shape[1]
    
    dw = (1/m) * np.dot(dz, a.T)
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    return dw, db

def linear_backward(w, dz, dev_z):
    return np.dot(w.T, dz) * dev_z

class DataSet(BaseModel):
    x: np.ndarray = Field(None)
    y: np.ndarray = Field(None)
    batch_size: int = Field(None,gt=0)
    shuffle_index: List[np.ndarray] = Field(None)
    seed: int = Field(np.random.randint(1000), gt=0)
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        
    def get_shuffle_index(self):
        
        m = self.x.shape[1]
        shuffle_index = np.random.RandomState(self.seed).permutation(m)
        
        if self.batch_size:
            n_batches = m // self.batch_size
        else:
            n_batches = 1
        self.shuffle_index = np.array_split(shuffle_index, n_batches)
        
    def from_timeseries(self, x, len_seq=None):
        n_batches = x.shape[1] - len_seq
        X = []
        Y = []
        for i in range(n_batches):
            X.append(x[:,i:i+len_seq])
            Y.append(x[:,i+len_seq])
        
        X = np.array(X)
        Y = np.expand_dims(np.array(Y),axis=2) 
        self.x = X
        self.y = Y
        self.batch_size = n_batches    
        

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
                A = A * dropout_array
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
        return [layer.dw for layer in self.layers]

    def get_grads_db(self):
        return [layer.db for layer in self.layers]
    
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
                reg_cost = reg_cost + layer.regularization_loss(lambd)
            return reg_cost
        else:
            return 0.
    
    def get_cost(self, y_hat,y):
        return self.loss.forward(y_hat, y) + self.get_regularization_loss()
            
    
    def train(self, x, y, show=10, n_epochs=100):
        n_layers = len(self.layers) + 1
        cost_list =[]
        self.optimizer.init_vd(self.get_weigths(), self.get_bias())
        c = 1
        for epoch in range(n_epochs):
                       
            #Forward
            AL = self.forward(x)
            
            #Cost
            cost = self.get_cost(AL,y)
            cost_list.append(cost)
            
            #first dz
            #self.layers[-1].dz = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
            self.layers[-1].dz = self.loss.backward(AL,y)
            #self.layers[-1].dz = self.loss.backward(AL,y) * self.layers[-1].derivative()
            
            #First grad
            dw, db = gradients(self.layers[-1].dz, self.cache[-2])
            
            # L2 Regularization
            if self.lambd > 0:
                dw = dw + 2*self.lambd*self.layers[-1].weights
            
            #Save Gradients
            self.layers[-1].dw = dw
            self.layers[-1].db = db

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
                    dw_l = dw_l *self.layers[l].dropout_cache
                
                #L2 Regularization
                if self.lambd > 0:
                    dw_l = dw_l + 2*self.lambd*self.layers[l].weights
                self.layers[l].dw = dw_l
                self.layers[l].db = db_l
                
            #Get new Weights
            new_weights, new_bias = self.optimizer.update(self.get_weigths(),self.get_grads_dw(),self.get_bias(),self.get_grads_db(),c,epoch)
            c += 1
            self.assing_weights(new_weights)
            self.assing_bias(new_bias)
            
            if epoch%show==0:
                print(f'{epoch} cost {cost}')
        
        self.cost = cost_list
        
    def train_dataset(self, dataset:DataSet, show=10, n_epochs=50):
        n_layers = len(self.layers) + 1
        cost_list =[]
        dataset.get_shuffle_index()
        self.optimizer.init_vd(self.get_weigths(), self.get_bias())
        c = 1
        for epoch in range(n_epochs):
            
            for batch in dataset.shuffle_index:
                
                x = dataset.x[:,batch]
                y = dataset.y[:,batch]
  
                #Forward
                AL = self.forward(x)
            
                #Cost
                cost = self.get_cost(AL,y)
                cost_list.append(cost)
            
                #first dz
                #self.layers[-1].dz = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
                self.layers[-1].dz = self.loss.backward(AL,y)
                #self.layers[-1].dz = self.loss.backward(AL,y) * self.layers[-1].derivative()
                
                #First grad
                dw, db = gradients(self.layers[-1].dz, self.cache[-2])
                
                # L2 Regularization
                if self.lambd > 0:
                    dw = dw + 2*self.lambd*self.layers[-1].weights
                
                #Save Gradients
                self.layers[-1].dw = dw
                self.layers[-1].db = db

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
                        dw_l = dw_l *self.layers[l].dropout_cache
                    
                    #L2 Regularization
                    if self.lambd > 0:
                        dw_l = dw_l + 2*self.lambd*self.layers[l].weights
                    self.layers[l].dw = dw_l
                    self.layers[l].db = db_l
                
                #Get new Weights
                new_weights, new_bias = self.optimizer.update(self.get_weigths(),self.get_grads_dw(),self.get_bias(),self.get_grads_db(),c,epoch)
                self.assing_weights(new_weights)
                self.assing_bias(new_bias)
                c += 1
            
            if epoch%show==0:
                print(f'{epoch} cost {cost} iter {c}')
        
        self.cost = cost_list
        

class RNN(BaseModel):
    n_input: int = Field(..., gt=0)
    n_output: int = Field(..., gt=0)
    n_hidden: int = Field(..., gt=0)
    truncate: int = Field(5, gt=0)
    layer_hidden_w: layers_types = Field(None)
    layer_hidden_u: layers_types = Field(None)
    layer_output: layers_types = Field(None)
    optimizer: optimizers_types = Field(...)
    loss: loss_types = Field(CategoricalCrossEntropy())
    cost: List[float] = Field(None)
    type: RnnEnum = Field(RnnEnum.many_one)
    
    def __init__(self,**kwargs):
        layer_hidden = kwargs.pop('layer_hidden',None)
        layer_output = kwargs.pop('layer_output',None)
        super().__init__(**kwargs)
        
        n_hidden = kwargs['n_hidden']
        n_output = kwargs['n_output']
        n_input = kwargs['n_input']
        
        self.layer_hidden_w = layer_hidden(n_output=n_hidden,n_input=n_hidden)
        self.layer_hidden_u = layer_hidden(n_output=n_hidden,n_input=n_input)
        self.layer_output = layer_output(n_output=n_output,n_input=n_hidden)
        
    
    #for layer_output: n_output=self.n_output,n_input=self.n_hidden
    #layer_output: layers_types = Field(...)
    
    #for layer_hidden: n_output=self.n_hidden,n_input=self.n_input + self.n_hidden
    #layer_hidden: layers_types = Field(...)
   
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        
        
    def forward(self,x):
        a = np.zeros((self.n_hidden,1))
        y_output = np.zeros((self.n_output,x.shape[1]))
        a_output = np.zeros((self.n_hidden,x.shape[1]))
        for t in range(x.shape[1]):
            x_new_shape = np.expand_dims(x[:,t],axis=1)
            
            zw = self.layer_hidden_u.forward(x_new_shape)
            zu = self.layer_hidden_w.forward(a)
            a = zw + zu
            a_output[:,t] = np.squeeze(a)
            y_output[:,t] = np.squeeze(self.layer_output.forward(a))
            
        return y_output, a_output

    def get_weigths(self):
        return [layer.weights for layer in [self.layer_hidden_w,self.layer_hidden_u,self.layer_output]]

    def get_bias(self):
        return [layer.bias for layer in [self.layer_hidden_w,self.layer_hidden_u,self.layer_output]]
    
    def get_grads_dw(self):
        return [layer.dw for layer in [self.layer_hidden_w,self.layer_hidden_u,self.layer_output]]

    def get_grads_db(self):
        return [layer.db for layer in [self.layer_hidden_w,self.layer_hidden_u,self.layer_output]]

    def assing_weights(self,new_weights):
        for layer, weights in zip([self.layer_hidden_w,self.layer_hidden_u,self.layer_output],new_weights):
            layer.weights = weights
            
    def assing_bias(self,new_bias):
        for layer, bias in zip([self.layer_hidden_w,self.layer_hidden_u,self.layer_output],new_bias):
            layer.bias = bias
            
    def get_loss(self,x,y):
        y_hat, a = self.forward(x)
        
        #Cost
        if self.type == RnnEnum.many_one:
            y_hat = y_hat[:,-1].reshape(-1,1)
        print(y_hat, y)
        return self.loss.forward(y_hat, y)
    
    def predict(self,x):
        y_hat, _ = self.forward(x)
        return y_hat
    
    def predict_batch(self,x):
        y = []
        for batch in range(x.shape[0]):
            x_n = x[batch,:,:]
            #Forward
            y_hat, _ = self.forward(x_n)
            if self.type == RnnEnum.many_one:
                y_hat = y_hat[:,-1].reshape(-1,1)
            y.append(y_hat)
        return np.squeeze(y)
            


    def train_dataset(self,dataset:DataSet, show=10, n_epochs=50):
        cost_list = []

        self.optimizer.init_vd(self.get_weigths(), self.get_bias())
        c = 0
        for epoch in range(n_epochs):
            for batch in range(dataset.x.shape[0]):
                dLdV = np.zeros(self.layer_output.weights.shape)
                dLdbV = np.zeros(self.layer_output.bias.shape)
                dLdW = np.zeros(self.layer_hidden_w.weights.shape)
                dLdbW = np.zeros(self.layer_hidden_w.bias.shape)
                dLdU = np.zeros(self.layer_hidden_u.weights.shape)
                dLdbU = np.zeros(self.layer_hidden_u.bias.shape)
                x = dataset.x[batch,:,:]
                y = dataset.y[batch,:,:]

                #Forward
                y_hat, a = self.forward(x)
                
                #Cost
                if self.type == RnnEnum.many_one:
                    y_hat = y_hat[:,-1].reshape(-1,1)
                cost = self.loss.forward(y_hat, y)
                cost_list.append(cost)
                if c%150==0:
                    print(f'{epoch} cost {cost} iter {c}')
                dz = self.loss.backward(y_hat,y)
                for t in range(dataset.x.shape[2]):
                    dv, dbv =  gradients(dz,a[:,t].reshape(-1,1))
                    dLdV += dv
                    dLdbV += dbv
                    dza = linear_backward(
                        self.layer_output.weights,
                        dz,
                        self.layer_hidden_w.derivative()
                    )
                    for bptt_step in np.arange(max(0, t-self.truncate), t+1)[::-1]:
                        # print &quot;Backpropagation step t=%d bptt step=%d &quot; % (t, bptt_step)
                        # Add to gradients at each previous step
                        dw, dbw = gradients(dza,a[:,bptt_step-1].reshape(-1,1))
                        du, dbu = gradients(dza,x[:,bptt_step-1].reshape(-1,1))
                        dLdW += dw# np.outer(dza, a[:,bptt_step-1])   
                        dLdU += du
                        dLdbU += dbu
                        dLdbW += dbw
                        dza = linear_backward(
                            self.layer_hidden_w.weights,
                            dza,
                            self.layer_hidden_w.derivative()
                        )
                
                self.layer_hidden_w.dw = dLdW
                self.layer_hidden_w.db = dLdbW
                self.layer_hidden_u.dw = dLdU
                self.layer_hidden_u.db = dLdbU
                self.layer_output.dw = dLdV
                self.layer_output.db = dLdbV
                
                new_weights, new_bias = self.optimizer.update(self.get_weigths(),self.get_grads_dw(),self.get_bias(),self.get_grads_db(),c,epoch)
                self.assing_weights(new_weights)
                self.assing_bias(new_bias)
                
                c += 1
            
            if epoch%show==0:
                print(f'{epoch} cost {cost} iter {c}')
        self.cost = cost_list
                # COntinue to code
                    
                    
                    
                    
                
                
                