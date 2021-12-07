import autograd.numpy as np
from typing import List, Union
from pydantic import BaseModel, Field


class LearningDecayModel(BaseModel):
    rate: float = Field(0.0)
    
class TimeDecay(LearningDecayModel):
    
    def forward(self, lr:float,epoch:int):
        return (1/(1+self.rate*epoch))*lr
    
class ExponentialDecay(LearningDecayModel):
    def forward(self, lr:float,epoch:int):
        return lr*np.exp(-self.rate*epoch)

    

class SGD(BaseModel):
    learning_rate: float = Field(1e-3)
    decay: Union[TimeDecay, ExponentialDecay] = Field(TimeDecay())
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        
    def update(self, weigths, wgrads, bias, bias_grads,t,epoch):
        w_list =[]
        alpha = self.decay.forward(self.learning_rate,epoch)
        for w, g in zip(weigths, wgrads):
            params_w = w - alpha * g
            w_list.append(params_w)
            
        b_list =[]
        for b, gb in zip(bias, bias_grads):
            params_b = b - alpha * gb
            w_list.append(params_b)
        return w_list, b_list
    
    def init_vd(self, weights,bias):
        pass
        
class SGDMomentum(BaseModel):
    learning_rate: float = Field(1e-3)
    vdw: List[np.ndarray] = Field(None)
    vdb: List[np.ndarray] = Field(None)
    momentum: float = Field(0.9)
    decay: Union[TimeDecay, ExponentialDecay] = Field(TimeDecay())
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        
    def update(self, weigths, wgrads, bias, bias_grads,t,epoch):
        w_list =[]
        beta = self.momentum
        alpha = self.decay.forward(self.learning_rate,epoch)
        for i, (vdw,w, gw) in enumerate(zip(self.vdw,weigths, wgrads)):
            new_vdw = beta * vdw + (1-beta)*gw
            params_w = w - alpha * new_vdw
            w_list.append(params_w)
            self.vdw[i] = new_vdw
            
        b_list =[]
        for i,(vdb,b, gb) in enumerate(zip(self.vdb,bias, bias_grads)):
            new_vdb = beta * vdb + (1-beta)*gb
            params_b = b - alpha * new_vdb
            b_list.append(params_b)
            self.vdb[i] = new_vdb
        return w_list, b_list
    
    def init_vd(self, weights,bias):
        list_vdw = []
        list_vdb = []
        for w, b in zip(weights, bias):
            vdw = np.zeros_like(w)
            vdb = np.zeros_like(b)
            list_vdw.append(vdw)
            list_vdb.append(vdb)
            
        self.vdw = list_vdw
        self.vdb = list_vdb
        
class RMSprop(BaseModel):
    learning_rate: float = Field(1e-3)
    sdw: List[np.ndarray] = Field(None)
    sdb: List[np.ndarray] = Field(None)
    momentum: float = Field(0.9)
    epsilon: float = Field(1e-7)
    decay: Union[TimeDecay, ExponentialDecay] = Field(TimeDecay())
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        
    def update(self, weigths, wgrads, bias, bias_grads,t,epoch):
        w_list =[]
        beta = self.momentum
        alpha = self.decay.forward(self.learning_rate,epoch)
        for i, (sdw,w, gw) in enumerate(zip(self.sdw,weigths, wgrads)):
            new_sdw = beta * sdw + (1-beta)*np.square(gw)
            params_w = w - alpha * (gw/(np.sqrt(new_sdw)+self.epsilon))
            w_list.append(params_w)
            self.sdw[i] = new_sdw
            
        b_list =[]
        for i,(sdb,b, gb) in enumerate(zip(self.sdb,bias, bias_grads)):
            new_sdb = beta * sdb + (1-beta)*np.square(gb)
            params_b = b - alpha * (gb/(np.sqrt(new_sdb)+self.epsilon))
            b_list.append(params_b)
            self.sdb[i] = new_sdb
        return w_list, b_list
    
    def init_vd(self, weights,bias):
        list_sdw = []
        list_sdb = []
        for w, b in zip(weights, bias):
            sdw = np.zeros_like(w)
            sdb = np.zeros_like(b)
            list_sdw.append(sdw)
            list_sdb.append(sdb)
            
        self.sdw = list_sdw
        self.sdb = list_sdb
        
class Adam(BaseModel):
    learning_rate: float = Field(1e-3)
    vdw: List[np.ndarray] = Field(None)
    vdb: List[np.ndarray] = Field(None)
    sdw: List[np.ndarray] = Field(None)
    sdb: List[np.ndarray] = Field(None)
    beta1: float = Field(0.9)
    beta2: float = Field(0.99)
    epsilon: float = Field(1e-8)
    decay: Union[TimeDecay, ExponentialDecay] = Field(TimeDecay())
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        
    def update(self, weigths, wgrads, bias, bias_grads,t,epoch):
        w_list =[]
        beta1 = self.beta1
        beta2 = self.beta2
        alpha = self.decay.forward(self.learning_rate,epoch)
        for i, (vdw,sdw,w, gw) in enumerate(zip(self.vdw,self.sdw,weigths, wgrads)):
            if t == 0:
                print("t = 0")
            #Momentumn
            new_vdw = beta1 * vdw + (1-beta1)*gw
            new_vdw_corr = new_vdw/(1-np.power(beta1,t))
            
            #RMSprop
            new_sdw = beta2 * sdw + (1-beta2)*np.square(gw)
            new_sdw_corr = new_sdw/(1-np.power(beta2,t))
            
            #update
            params_w = w - (alpha * (new_vdw_corr/(np.sqrt(new_sdw_corr)+self.epsilon)))
            w_list.append(params_w)
            self.sdw[i] = new_sdw_corr
            self.vdw[i] = new_vdw_corr
               
            
        b_list =[]
        for i, (vdb,sdb,b, gb) in enumerate(zip(self.vdb,self.sdb,bias, bias_grads)):
            #Momentumn
            new_vdb = beta1 * vdb + (1-beta1)*gb
            new_vdb_corr = new_vdb/(1-np.power(beta1,t))
            
            #RMSprop
            new_sdb = beta2 * sdb + (1-beta2)*np.square(gb)
            new_sdb_corr = new_sdb/(1-np.power(beta2,t))
            
            #update
            params_b = b - (alpha * (new_vdb_corr/(np.sqrt(new_sdb_corr)+self.epsilon)))
            b_list.append(params_b)
            self.sdb[i] = new_sdb_corr
            self.vdb[i] = new_vdb_corr
        return w_list, b_list
    
    def init_vd(self, weights,bias):
        list_vdw = []
        list_vdb = []
        list_sdw = []
        list_sdb = []
        for w, b in zip(weights, bias):
            vdw = np.zeros_like(w)
            sdw = np.zeros_like(w)
            vdb = np.zeros_like(b)
            sdb = np.zeros_like(b)
            list_vdw.append(vdw)
            list_vdb.append(vdb)
            list_sdw.append(sdw)
            list_sdb.append(sdb)
            
        self.vdw = list_vdw
        self.vdb = list_vdb
        self.sdw = list_sdw
        self.sdb = list_sdb
    
# class SGDMomentum(BaseModel):
#     learning_rate: float = Field(1e-3)
#     momentum: float = Field(0.9)
#     def update(self, weigths, grads):
#         params_list =[]
#         for w, g in zip(weigths, grads):
#             params = w - self.learning_rate * g
#             params_list.append(params)
#         return params_list
    
    
# def exp_wgh_avg(y,beta):

#     yw = np.zeros(y.shape[0])
    
#     for i,r in enumerate(y):
#         if i>0:
#             yw[i] = (beta*yw[i-1] + (1-beta)*y[i])
            
#     bias_correction = 1 - np.power(beta,np.arange(len(yw)))
    
#     y = np.nan_to_num(yw / bias_correction)
    
#     return y

# def beta_from_days(days):
#     return 1 - (1/days)
